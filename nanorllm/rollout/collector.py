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
