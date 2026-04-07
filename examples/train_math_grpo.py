import sys
import logging
import time
from pathlib import Path
import torch
from dataclasses import dataclass

import json
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanorllm.rollout.engine import RolloutEngine
from nanorllm.agents.math_agent import MathAgent
from nanorllm.envs.math_env import MathEnv

from nanorllm.trainer.trainer import run_train_epoch
from nanorllm.policy.hf_causal import HFCausalPolicy
from nanorllm.rewards.math_reward import math_reward
from nanorllm.utils.util import rollout_to_viewer_json
from nanorllm.eval.metrics import compute_basic_eval_metrics

from nanorllm.data.simple_math import get_simple_math_tasks


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# Preset model shortcuts (case-insensitive keys)
MODEL_PRESETS = {
    "llama3-8b-instruct":"Meta-Llama-3-8B-Instruct",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-0.5b-instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "tinyllama-1.1b-chat": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "smollm2-135m-instruct": "HuggingFaceTB/SmolLM2-135M-Instruct",
}


# Preset dataset shortcuts (case-insensitive keys)
DATASET_PRESETS = {
    "simple-math": "simple_math",
    "gsm8k-jsonl": "gsm8k_jsonl",
    "arc-jsonl": "arc_jsonl",
    "mmlu-jsonl": "mmlu_jsonl",
    "humaneval-jsonl": "humaneval_jsonl",
}


MATH_SYSTEM_PROMPT = """You are a careful math problem solver. Think step by step when useful, and end with a clear final answer.\n\nFollow these rules strictly:\n1) Solve the question and return exactly one final answer wrapped in \\boxed{...}.\n2) In \\boxed{...}, output only the final value/expression (no words, units, punctuation, or extra spaces).\n3) Never output multiple boxed answers.\n"""

MCQ_SYSTEM_PROMPT = """You are a careful multiple-choice question solver.\nFollow these rules strictly:\n1) Read the question and options.\n2) Reply with ONLY one capital letter of your choice (e.g., A).\n3) Do NOT output explanations, words, or punctuation.\n"""

CODE_SYSTEM_PROMPT = """You are a helpful coding assistant.\nWrite valid Python code that defines the required function.\nOnly output the code (no backticks, no explanations).\n"""


@dataclass
class TrainArgs:
    model_name: str = "smollm2-135m-instruct"
    device: str = "cuda:0"

    clip_eps: float=0.2

    temperature: float = 0.5
    max_new_tokens: int=64
    max_steps: int = 5
    num_samples_per_task: int = 4
    max_length: int = 1024
    max_turn: int = 5

    lr: float = 1e-5
    train_batch_size: int = 3
    loss_agg_mode: str = 'seq-mean-token-mean'
    mode:str = 'step'

    # Dataset selection
    # One of DATASET_PRESETS keys: "simple-math" or "gsm8k-jsonl"
    dataset: str = "simple-math"
    dataset_path: str | None = None
    # Optional limit for quick runs
    # dataset_limit: int | None = 100
    dataset_limit: int | None = None

    # Prefer-offline model loading: snapshot to cache and load with local_files_only
    prefer_offline: bool = True
    offline_cache_dir: str = "models/"

    # Evaluation options
    eval_before: bool = True
    eval_after: bool = True
    eval_temperature: float = 0.3
    eval_num_samples_per_task: int = 1
    # eval_limit: int | None = None
    eval_limit: int | None = 20

def rollout_fn(task):
    return engine.run_episode(agent, env, policy, task, args)


if __name__ == "__main__":
    logger.info("Initializing training run")
    args = TrainArgs()
    logger.info("TrainArgs: %s", args)
    
    engine = RolloutEngine()
    # Resolve dataset from preset-style selection and build corresponding tasks/env/agent
    dataset_key = str(args.dataset).lower()
    if dataset_key == "simple-math":
        agent = MathAgent(system_prompt=MATH_SYSTEM_PROMPT)
        env = MathEnv(reward_fn=math_reward, max_turn=args.max_turn)
        # tasks = get_simple_math_tasks()[:2]
        tasks = get_simple_math_tasks()
        logger.info("Selected %s tasks: %s", len(tasks), [task["task_id"] for task in tasks])
    elif dataset_key == "gsm8k-jsonl":
        from nanorllm.data.gsm8k_jsonl import get_gsm8k_tasks_from_jsonl
        from nanorllm.datasets_auto import ensure_local_jsonl
        agent = MathAgent(system_prompt=MATH_SYSTEM_PROMPT)
        env = MathEnv(reward_fn=math_reward, max_turn=args.max_turn)
        # Auto-download to datasets/ if dataset_path not provided
        dataset_jsonl = args.dataset_path or str(
            ensure_local_jsonl(
                "gsm8k-jsonl",
                cache_root=Path(__file__).resolve().parents[1] / "datasets" / "auto_cache",
                config="main",
                split="train",
            )
        )
        tasks = get_gsm8k_tasks_from_jsonl(dataset_jsonl, split="train", limit=args.dataset_limit)
        logger.info("Loaded GSM8K tasks from %s (count=%s)", dataset_jsonl, len(tasks))
    elif dataset_key == "arc-jsonl":
        from nanorllm.agents.mcq_agent import MCQAgent
        from nanorllm.envs.mcq_env import MCQEnv
        from nanorllm.rewards.classification_reward import classification_reward
        from nanorllm.data.arc_jsonl import get_arc_tasks_from_jsonl
        from nanorllm.datasets_auto import ensure_local_jsonl
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
        logger.info("Loaded ARC tasks from %s (count=%s)", dataset_jsonl, len(tasks))
    elif dataset_key == "mmlu-jsonl":
        from nanorllm.agents.mcq_agent import MCQAgent
        from nanorllm.envs.mcq_env import MCQEnv
        from nanorllm.rewards.classification_reward import classification_reward
        from nanorllm.data.mmlu_jsonl import get_mmlu_tasks_from_jsonl
        from nanorllm.datasets_auto import ensure_local_jsonl
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
        logger.info("Loaded MMLU tasks from %s (count=%s)", dataset_jsonl, len(tasks))
    elif dataset_key == "humaneval-jsonl":
        from nanorllm.data.humaneval_jsonl import get_humaneval_tasks_from_jsonl
        from nanorllm.envs.code_eval_env import CodeEvalEnv
        from nanorllm.rewards.code_eval_reward import code_eval_reward
        from nanorllm.datasets_auto import ensure_local_jsonl
        # MathAgent works since it formats observation['question'] as user text
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
        logger.info("Loaded HumanEval-like tasks from %s (count=%s)", dataset_jsonl, len(tasks))
    else:
        raise ValueError(f"Unknown dataset preset: {args.dataset}")
    load_start = time.perf_counter()
    resolved_model_name = MODEL_PRESETS.get(str(args.model_name).lower(), args.model_name)
    logger.info("Resolved model_name: %s", resolved_model_name)
    policy = HFCausalPolicy(
        model_name=resolved_model_name,
        device=args.device,
        prefer_offline=args.prefer_offline,
        offline_cache_dir=args.offline_cache_dir,
    )
    logger.info("Policy loaded in %.2fs", time.perf_counter() - load_start)
    tokenizer = policy.tokenizer
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    # tasks have been prepared above with the selected dataset preset

    # Optional pre-train evaluation
    if args.eval_before:
        from types import SimpleNamespace
        eval_args = SimpleNamespace(
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
            temperature=args.eval_temperature,
            num_samples_per_task=args.eval_num_samples_per_task,
            max_turn=args.max_turn,
            max_length=args.max_length,
        )
        eval_tasks = tasks[: args.eval_limit] if args.eval_limit else tasks
        def rollout_fn_eval(task):
            return engine.run_episode(agent, env, policy, task, eval_args)
        from nanorllm.rollout.collector import execute_tasks as _exec
        logger.info("Running pre-train eval on %s tasks", len(eval_tasks))
        eval_rollouts = _exec(eval_tasks, eval_args.num_samples_per_task, rollout_fn_eval)
        eval_metrics = compute_basic_eval_metrics(eval_rollouts)
        logger.info("Pre-train eval: %s", eval_metrics)

    train_start = time.perf_counter()
    result = run_train_epoch(
        tasks,
        rollout_fn,
        policy,
        tokenizer,
        optimizer,
        args
    )
    logger.info("Training run completed in %.2fs", time.perf_counter() - train_start)
    logger.info("Training metrics: %s", result["metrics"])
    # Log final reward statistics across collected rollouts
    rollouts = result.get("rollouts", [])
    if rollouts:
        successes = sum(1 for r in rollouts if getattr(r.trajectory, "final_reward", 0.0) >= 1.0)
        avg_final_reward = sum(float(getattr(r.trajectory, "final_reward", 0.0)) for r in rollouts) / len(rollouts)
        logger.info("Final rewards: avg=%.4f, success=%d/%d", avg_final_reward, successes, len(rollouts))

    # Optional post-train evaluation
    if args.eval_after:
        from types import SimpleNamespace
        eval_args = SimpleNamespace(
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
            temperature=args.eval_temperature,
            num_samples_per_task=args.eval_num_samples_per_task,
            max_turn=args.max_turn,
            max_length=args.max_length,
        )
        eval_tasks = tasks[: args.eval_limit] if args.eval_limit else tasks
        def rollout_fn_eval(task):
            return engine.run_episode(agent, env, policy, task, eval_args)
        from nanorllm.rollout.collector import execute_tasks as _exec
        logger.info("Running post-train eval on %s tasks", len(eval_tasks))
        eval_rollouts = _exec(eval_tasks, eval_args.num_samples_per_task, rollout_fn_eval)
        eval_metrics = compute_basic_eval_metrics(eval_rollouts)
        logger.info("Post-train eval: %s", eval_metrics)

    viewer_data = rollout_to_viewer_json(result["rollouts"]) 
    out_path = Path("docs/exported_trajectories.json")
    out_path.write_text(json.dumps(viewer_data, ensure_ascii=False, indent=2))
    logger.info("Exported viewer data to %s", out_path.resolve())
