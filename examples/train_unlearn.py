import gc
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanorllm.agents.math_agent import MathAgent
from nanorllm.data.unlearn_split import split_tasks
from nanorllm.envs.math_env import MathEnv
from nanorllm.eval.metrics import compute_basic_eval_metrics
from nanorllm.policy.hf_causal import HFCausalPolicy
from nanorllm.policy.reference import ReferencePolicy
from nanorllm.rewards.math_reward import math_reward
from nanorllm.rollout.engine import RolloutEngine
from nanorllm.trainer.trainer import run_train_epoch, run_unlearn_epoch
from nanorllm.utils.util import build_parser, parse_dataclass, print_args_table, rollout_to_viewer_json, setup_file_logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Presets (same as train_math_grpo.py) ---

MODEL_PRESETS = {
    "llama3-8b-instruct": "Meta-Llama-3-8B-Instruct",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-0.5b-instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "tinyllama-1.1b-chat": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "smollm2-135m-instruct": "HuggingFaceTB/SmolLM2-135M-Instruct",
}

DATASET_PRESETS = {
    "simple-math": "simple_math",
    "gsm8k-jsonl": "gsm8k_jsonl",
    "arc-jsonl": "arc_jsonl",
    "mmlu-jsonl": "mmlu_jsonl",
    "humaneval-jsonl": "humaneval_jsonl",
}

MATH_SYSTEM_PROMPT = """You are a careful math problem solver. Think step by step when useful, and end with a clear final answer.

Follow these rules strictly:
1) Solve the question and return exactly one final answer wrapped in \\boxed{...}.
2) In \\boxed{...}, output only the final value/expression (no words, units, punctuation, or extra spaces).
3) Never output multiple boxed answers.
"""

MCQ_SYSTEM_PROMPT = """You are a careful multiple-choice question solver.
Follow these rules strictly:
1) Read the question and options.
2) Reply with ONLY one capital letter of your choice (e.g., A).
3) Do NOT output explanations, words, or punctuation.
"""

CODE_SYSTEM_PROMPT = """You are a helpful coding assistant.
Write valid Python code that defines the required function.
Only output the code (no backticks, no explanations).
"""


@dataclass
class UnlearnArgs:
    # --- Model ---
    model_name: str = "qwen2.5-3b-instruct"
    device: str = "cuda:0"

    # --- PPO / GRPO ---
    clip_eps: float = 0.2
    temperature: float = 0.9
    max_new_tokens: int = 256
    max_steps: int = 5
    num_samples_per_task: int = 6
    max_length: int = 1024
    max_turn: int = 5

    # --- Training ---
    lr: float = 5e-6
    unlearn_lr: float = 5e-7
    max_grad_norm: float = 0.1
    train_batch_size: int = 4
    loss_agg_mode: str = "seq-mean-token-mean"
    mode: str = "step"

    # --- Dataset ---
    dataset: str = "gsm8k-jsonl"
    dataset_path: str | None = None
    dataset_limit: int | None = 10

    # --- Dataset split ---
    split_mode: str = "ratio"
    forget_ratio: float = 0.3
    forget_task_ids: list[str] | None = None
    split_seed: int = 42

    # --- Learning phase ---
    learning_epochs: int = 3

    # --- Unlearn hyperparams ---
    lambda_kl: float = 1.0

    # --- UX ---
    show_progress: bool = True

    # --- Offline loading ---
    prefer_offline: bool = True
    offline_cache_dir: str = "models/"

    # --- Evaluation ---
    eval_before: bool = True
    eval_after: bool = True
    eval_temperature: float = 0.3
    eval_num_samples_per_task: int = 1
    eval_limit: int | None = None


def _run_eval(tasks, name, engine, agent, env, policy, args):
    """Evaluate policy on a named task subset and log metrics."""
    from nanorllm.rollout.collector import execute_tasks as _exec

    if not tasks:
        logger.info("Eval %s: no tasks, skipping", name)
        return
    limit = args.eval_limit
    eval_tasks = tasks[:limit] if limit else tasks
    eval_args = SimpleNamespace(
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        temperature=args.eval_temperature,
        num_samples_per_task=args.eval_num_samples_per_task,
        max_turn=args.max_turn,
        max_length=args.max_length,
    )

    def _rollout_fn(task):
        return engine.run_episode(agent, env, policy, task, eval_args)

    logger.info("Eval %s: %s tasks", name, len(eval_tasks))
    eval_rollouts = _exec(eval_tasks, eval_args.num_samples_per_task, _rollout_fn, show_progress=args.show_progress)
    metrics = compute_basic_eval_metrics(eval_rollouts)
    logger.info("Eval %s: %s", name, metrics)
    return eval_rollouts, metrics


if __name__ == "__main__":
    parser = build_parser(UnlearnArgs, description="Agentic RL Unlearning")
    args = parse_dataclass(parser, UnlearnArgs)
    setup_file_logging(args.model_name, args.dataset, logger)
    logger.info("Initializing unlearn run")
    print_args_table(args, logger, title="UnlearnArgs")

    engine = RolloutEngine()

    # --- Build agent / env / tasks (mirrors train_math_grpo.py) ---
    dataset_key = str(args.dataset).lower()
    if dataset_key == "simple-math":
        from nanorllm.data.simple_math import get_simple_math_tasks

        agent = MathAgent(system_prompt=MATH_SYSTEM_PROMPT)
        env = MathEnv(reward_fn=math_reward, max_turn=args.max_turn)
        tasks = get_simple_math_tasks()
        logger.info("Loaded %s simple-math tasks", len(tasks))
    elif dataset_key == "gsm8k-jsonl":
        from nanorllm.data.gsm8k_jsonl import get_gsm8k_tasks_from_jsonl
        from nanorllm.datasets_auto import ensure_local_jsonl

        agent = MathAgent(system_prompt=MATH_SYSTEM_PROMPT)
        env = MathEnv(reward_fn=math_reward, max_turn=args.max_turn)
        dataset_jsonl = args.dataset_path or str(
            ensure_local_jsonl(
                "gsm8k-jsonl",
                cache_root=Path(__file__).resolve().parents[1] / "datasets" / "auto_cache",
                config="main",
                split="train",
            )
        )
        tasks = get_gsm8k_tasks_from_jsonl(dataset_jsonl, split="train", limit=args.dataset_limit)
        logger.info("Loaded GSM8K tasks: count=%s", len(tasks))
    elif dataset_key == "arc-jsonl":
        from nanorllm.agents.mcq_agent import MCQAgent
        from nanorllm.data.arc_jsonl import get_arc_tasks_from_jsonl
        from nanorllm.datasets_auto import ensure_local_jsonl
        from nanorllm.envs.mcq_env import MCQEnv
        from nanorllm.rewards.classification_reward import classification_reward

        agent = MCQAgent(system_prompt=MCQ_SYSTEM_PROMPT)
        env = MCQEnv(reward_fn=classification_reward, max_turn=args.max_turn)
        dataset_jsonl = args.dataset_path or str(
            ensure_local_jsonl(
                "arc-jsonl",
                cache_root=Path(__file__).resolve().parents[1] / "datasets" / "auto_cache",
                config="ARC-Challenge",
                split="train",
            )
        )
        tasks = get_arc_tasks_from_jsonl(dataset_jsonl, split="train", limit=args.dataset_limit)
        logger.info("Loaded ARC tasks: count=%s", len(tasks))
    elif dataset_key == "mmlu-jsonl":
        from nanorllm.agents.mcq_agent import MCQAgent
        from nanorllm.data.mmlu_jsonl import get_mmlu_tasks_from_jsonl
        from nanorllm.datasets_auto import ensure_local_jsonl
        from nanorllm.envs.mcq_env import MCQEnv
        from nanorllm.rewards.classification_reward import classification_reward

        agent = MCQAgent(system_prompt=MCQ_SYSTEM_PROMPT)
        env = MCQEnv(reward_fn=classification_reward, max_turn=args.max_turn)
        dataset_jsonl = args.dataset_path or str(
            ensure_local_jsonl(
                "mmlu-jsonl",
                cache_root=Path(__file__).resolve().parents[1] / "datasets" / "auto_cache",
                config="all",
                split="validation",
            )
        )
        tasks = get_mmlu_tasks_from_jsonl(dataset_jsonl, split="val", limit=args.dataset_limit)
        logger.info("Loaded MMLU tasks: count=%s", len(tasks))
    elif dataset_key == "humaneval-jsonl":
        from nanorllm.data.humaneval_jsonl import get_humaneval_tasks_from_jsonl
        from nanorllm.datasets_auto import ensure_local_jsonl
        from nanorllm.envs.code_eval_env import CodeEvalEnv
        from nanorllm.rewards.code_eval_reward import code_eval_reward

        agent = MathAgent(system_prompt=CODE_SYSTEM_PROMPT)
        env = CodeEvalEnv(reward_fn=code_eval_reward, max_turn=args.max_turn)
        dataset_jsonl = args.dataset_path or str(
            ensure_local_jsonl(
                "humaneval-jsonl",
                cache_root=Path(__file__).resolve().parents[1] / "datasets" / "auto_cache",
                split="test",
            )
        )
        tasks = get_humaneval_tasks_from_jsonl(dataset_jsonl, split="test", limit=args.dataset_limit)
        logger.info("Loaded HumanEval tasks: count=%s", len(tasks))
    else:
        raise ValueError(f"Unknown dataset preset: {args.dataset}")

    # --- Load base model ---
    load_start = time.perf_counter()
    resolved_model_name = MODEL_PRESETS.get(str(args.model_name).lower(), args.model_name)
    logger.info("Resolved model_name: %s", resolved_model_name)
    policy = HFCausalPolicy(
        model_name=resolved_model_name,
        device=args.device,
        prefer_offline=args.prefer_offline,
        offline_cache_dir=args.offline_cache_dir,
    )
    logger.info("Base policy loaded in %.2fs", time.perf_counter() - load_start)

    tokenizer = policy.tokenizer
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    def rollout_fn(task):
        return engine.run_episode(agent, env, policy, task, args)

    # --- Phase 1: Learning ---
    logger.info("Starting learning phase: %s epoch(s), %s tasks", args.learning_epochs, len(tasks))
    for epoch in range(args.learning_epochs):
        learn_result = run_train_epoch(tasks, rollout_fn, policy, tokenizer, optimizer, args, show_progress=args.show_progress)
        logger.info("Learning epoch %s/%s: %s", epoch + 1, args.learning_epochs, learn_result["metrics"])
    logger.info("Learning phase complete")

    # Release learning optimizer state to free GPU memory before loading reference model
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()

    # --- Phase 2: Freeze trained model as reference ---
    ref_policy = ReferencePolicy(
        policy,
        prefer_offline=args.prefer_offline,
        offline_cache_dir=args.offline_cache_dir,
        device="cpu",
    )
    logger.info("Reference policy created from trained model")

    # --- Phase 3: Split tasks ---
    forget_tasks, retain_tasks = split_tasks(tasks, args)
    logger.info(
        "Split: forget=%s retain=%s (mode=%s)",
        len(forget_tasks),
        len(retain_tasks),
        args.split_mode,
    )

    # --- Phase 4: Eval before unlearning (trained model) ---
    if args.eval_before:
        _run_eval(forget_tasks, "forget-before", engine, agent, env, policy, args)
        _run_eval(retain_tasks, "retain-before", engine, agent, env, policy, args)

    # --- Phase 5: Unlearning ---
    unlearn_optimizer = torch.optim.AdamW(policy.parameters(), lr=args.unlearn_lr)
    train_start = time.perf_counter()
    result = run_unlearn_epoch(
        forget_tasks,
        retain_tasks,
        rollout_fn,
        policy,
        ref_policy,
        tokenizer,
        unlearn_optimizer,
        args,
        show_progress=args.show_progress,
    )
    logger.info("Unlearning completed in %.2fs", time.perf_counter() - train_start)
    logger.info("Unlearning metrics: %s", result["metrics"])

    # --- Phase 6: Eval after unlearning (unlearned model) ---
    if args.eval_after:
        _run_eval(forget_tasks, "forget-after", engine, agent, env, policy, args)
        _run_eval(retain_tasks, "retain-after", engine, agent, env, policy, args)

    # --- Export ---
    all_rollouts = result.get("forget_rollouts", []) + result.get("retain_rollouts", [])
    if all_rollouts:
        viewer_data = rollout_to_viewer_json(all_rollouts)
        out_path = Path("docs/exported_trajectories.json")
        out_path.write_text(json.dumps(viewer_data, ensure_ascii=False, indent=2))
        logger.info("Exported viewer data to %s", out_path.resolve())
