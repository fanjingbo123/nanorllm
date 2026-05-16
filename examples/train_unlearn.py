import argparse
import gc
import json
import logging
import sys
import time
from dataclasses import dataclass, fields
from datetime import datetime
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
from nanorllm.utils.util import rollout_to_viewer_json

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
    model_name: str = "smollm2-135m-instruct"
    device: str = "cuda:0"

    # --- PPO / GRPO ---
    clip_eps: float = 0.2
    temperature: float = 0.9
    max_new_tokens: int = 256
    max_steps: int = 5
    num_samples_per_task: int = 4
    max_length: int = 1024
    max_turn: int = 5

    # --- Training ---
    lr: float = 5e-6
    train_batch_size: int = 4
    loss_agg_mode: str = "seq-mean-token-mean"
    mode: str = "step"

    # --- Dataset ---
    dataset: str = "arc-jsonl"
    dataset_path: str | None = None
    dataset_limit: int | None = 100

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


def _setup_file_logging(args: UnlearnArgs) -> None:
    """Mirror console output to logs/<model>/<dataset>/<timestamp>.log."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / args.model_name / args.dataset
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{ts}.log"
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(fh)
    logger.info("Logging to %s", log_path)


def _print_args_table(args: UnlearnArgs) -> None:
    """Print hyperparameters as a 4-column table via logger."""
    items = [(f.name, str(getattr(args, f.name))) for f in fields(args)]

    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title="UnlearnArgs", show_header=True, header_style="bold")
        table.add_column("Param", style="dim")
        table.add_column("Value")
        table.add_column("Param", style="dim")
        table.add_column("Value")
        for i in range(0, len(items), 2):
            left, right = items[i], items[i + 1] if i + 1 < len(items) else ("", "")
            table.add_row(left[0], left[1], right[0], right[1])

        console = Console(width=120)
        with console.capture() as capture:
            console.print(table)
        logger.info("\n%s", capture.get().rstrip())
    except ImportError:
        logger.info("UnlearnArgs: %s", args)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Agentic RL Unlearning")
    type_map = {int: int, float: float, str: str, bool: lambda x: x.lower() not in ("0", "false", "no")}
    for f in fields(UnlearnArgs):
        flag = "--" + f.name.replace("_", "-")
        if f.type is bool:
            p.add_argument(flag, action="store_true", default=f.default, help=f"(default: {f.default})")
            p.add_argument("--no-" + f.name.replace("_", "-"), dest=f.name, action="store_false", help=f"disable {f.name}")
        elif f.type == list[str] | None or str(f.type).startswith("list"):
            p.add_argument(flag, type=str, default=f.default, help=f"comma-separated (default: {f.default})")
        elif f.type == int | None:
            p.add_argument(flag, type=int, default=f.default, help=f"(default: {f.default})")
        else:
            t = type_map.get(f.type, str)
            p.add_argument(flag, type=t, default=f.default, help=f"(default: {f.default})")
    return p


if __name__ == "__main__":
    parser = _build_parser()
    cli = parser.parse_args()
    kwargs = {f.name: getattr(cli, f.name) for f in fields(UnlearnArgs)}
    # Parse comma-separated lists
    for f in fields(UnlearnArgs):
        if f.type == list[str] | None or str(f.type).startswith("list"):
            val = kwargs[f.name]
            if isinstance(val, str) and val:
                kwargs[f.name] = [v.strip() for v in val.split(",")]
            elif isinstance(val, str) and not val:
                kwargs[f.name] = None
    args = UnlearnArgs(**kwargs)
    _setup_file_logging(args)
    logger.info("Initializing unlearn run")
    _print_args_table(args)

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
    unlearn_optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
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
