from __future__ import annotations

from typing import Any

from nanorllm.core.trajectory import Rollout


def compute_basic_eval_metrics(rollouts: list[Rollout]) -> dict[str, Any]:
    """Compute simple evaluation metrics from rollouts.

    - pass_rate: fraction of rollouts with final_reward >= 1.0 (exact pass)
    - avg_final_reward: mean of final_reward across rollouts
    - avg_num_steps: mean number of steps per rollout
    - num_rollouts: total rollouts evaluated

    Notes:
      - For HumanEval-like tasks where reward can be fractional (pass ratio),
        pass is defined as final_reward >= 1.0; avg_final_reward reflects the
        average fraction passed.
    """
    if not rollouts:
        return {
            "num_rollouts": 0,
            "pass_rate": 0.0,
            "avg_final_reward": 0.0,
            "avg_num_steps": 0.0,
        }

    num = len(rollouts)
    passes = 0
    reward_sum = 0.0
    steps_sum = 0

    for r in rollouts:
        reward = float(getattr(r.trajectory, "final_reward", 0.0))
        reward_sum += reward
        steps_sum += int(len(r.trajectory.steps or []))
        if reward >= 1.0:
            passes += 1

    return {
        "num_rollouts": num,
        "pass_rate": passes / num,
        "avg_final_reward": reward_sum / num,
        "avg_num_steps": steps_sum / num,
    }

