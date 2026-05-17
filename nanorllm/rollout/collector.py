import logging
import time
from typing import Any

from nanorllm.core.trajectory import Rollout

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

logger = logging.getLogger(__name__)


def stats_rollout(rollout: Rollout) -> dict[str, Any]:
    num_steps = len(rollout.trajectory.steps)
    prompt_length = sum(int(step_view.prompt_ids.numel()) for step_view in rollout.step_views)
    response_length = sum(int(step_view.response_ids.numel()) for step_view in rollout.step_views)
    return {
        "num_steps": num_steps,
        "prompt_length": prompt_length,
        "response_length": response_length,
    }


def execute_tasks(tasks, num_samples_per_task, rollout_fn, *, show_progress: bool = False) -> list[Rollout]:
    rollouts = []
    total_rollouts = len(tasks) * num_samples_per_task

    task_iter = tasks
    if show_progress and _HAS_TQDM:
        task_iter = tqdm(tasks, desc="rollout", unit="task")

    rollout_idx = 0
    for task in task_iter:
        task_id = task.get("task_id", "task")
        if show_progress and _HAS_TQDM and hasattr(task_iter, "set_postfix_str"):
            task_iter.set_postfix_str(task_id)

        for sample_idx in range(1, num_samples_per_task + 1):
            rollout_idx += 1
            run_id = f"{task_id}_sample{sample_idx}"
            start_time = time.perf_counter()
            rollout_result = rollout_fn(task)
            rollout_time = time.perf_counter() - start_time

            rollout_result.run_id = run_id
            rollout_result.stats = stats_rollout(rollout_result)
            rollout_result.timing["rollout_time"] = rollout_time
            rollouts.append(rollout_result)
    return rollouts


def execute_tasks_batch(tasks, num_samples_per_task, policy, agents, envs,
                        args, *, show_progress=False) -> list[Rollout]:
    """Batch rollout: process tasks in parallel chunks via run_episodes_batch.

    rollout_batch_size controls the chunk size; 0 means full batch.
    """
    from nanorllm.rollout.engine import run_episodes_batch

    flat_tasks = [task for task in tasks for _ in range(num_samples_per_task)]
    total = len(flat_tasks)
    if total == 0:
        return []

    if agents is None or envs is None:
        raise ValueError("agents and envs are required for batch rollout")
    if len(agents) < total or len(envs) < total:
        raise ValueError(
            f"Need {total} agents/envs, got {len(agents)} agents and {len(envs)} envs"
        )

    raw_bs = getattr(args, "rollout_batch_size", 16)
    if raw_bs is None:
        raw_bs = 16
    if raw_bs == 0:
        bs = total
    elif raw_bs > 0:
        bs = raw_bs
    else:
        raise ValueError("rollout_batch_size must be >= 0")

    pbar = None
    if show_progress and _HAS_TQDM:
        pbar = tqdm(total=total, desc="rollout", unit="episode")

    all_rollouts = []
    for start in range(0, total, bs):
        end = min(start + bs, total)
        chunk_tasks = flat_tasks[start:end]
        chunk_agents = agents[start:end]
        chunk_envs = envs[start:end]
        if pbar is not None:
            pbar.set_postfix_str(chunk_tasks[0].get("task_id", ""))
        rollouts = run_episodes_batch(
            policy, chunk_agents, chunk_envs, chunk_tasks, args, pbar=pbar,
        )
        for i, rollout in enumerate(rollouts):
            idx = start + i
            rollout.run_id = f"{flat_tasks[idx].get('task_id', 'task')}_sample{(idx % num_samples_per_task) + 1}"
            rollout.stats = stats_rollout(rollout)
        all_rollouts.extend(rollouts)

    if pbar is not None:
        pbar.close()
    return all_rollouts
