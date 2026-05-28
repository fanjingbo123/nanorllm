from __future__ import annotations

from nanorllm.core.types import RewardOutput
from nanorllm.envs.code_eval_env import _run_ds1000, _run_test_cases, _run_test_code


def code_eval_reward(task: dict, action, timeout: float = 5.0) -> RewardOutput:
    """Evaluate Python code, with DS-1000 code_context taking priority."""
    code = str(action.value or "")
    code_context = task.get("code_context")

    # DS-1000
    if isinstance(code_context, str) and code_context.strip():
        ok, last_error = _run_ds1000(code, code_context, timeout=timeout)
        return RewardOutput(
            reward=1.0 if ok else 0.0,
            is_correct=ok,
            metadata={"last_error": last_error},
        )

    entry_point = str(task.get("entry_point", "")).strip()
    test_cases = task.get("test_cases")
    test_code = task.get("test")

    try:
        if isinstance(test_cases, list) and test_cases:
            passed, total, last_error = _run_test_cases(code, entry_point, test_cases, timeout=timeout)
            is_correct = passed == total and total > 0
            reward = 1.0 if is_correct else float(passed) / float(total or 1)
            return RewardOutput(
                reward=reward,
                is_correct=is_correct,
                metadata={"passed": passed, "total": total, "last_error": last_error},
            )

        if isinstance(test_code, str) and test_code.strip():
            ok, last_error = _run_test_code(code, test_code, timeout=max(timeout, 10.0))
            return RewardOutput(
                reward=1.0 if ok else 0.0,
                is_correct=ok,
                metadata={"last_error": last_error},
            )

        return RewardOutput(
            reward=0.0,
            is_correct=False,
            metadata={"reason": "no_test_cases_or_test_code"},
        )
    except Exception as e:
        return RewardOutput(reward=0.0, is_correct=False, metadata={"error": str(e)})
