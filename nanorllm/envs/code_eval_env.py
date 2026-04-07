import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from nanorllm.envs.base import BaseEnv
from nanorllm.core.types import RewardOutput


def _run_test_cases(code: str, entry_point: str, test_cases: list, timeout: float = 5.0) -> tuple[int, int, str]:
    """Run simple black-box tests by calling entry_point with positional args.

    Each test case: {"input": [args], "output": expected}
    Returns: (num_passed, total, last_error)
    """
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        solution_path = td_path / "solution.py"
        runner_path = td_path / "runner.py"

        solution_path.write_text(code, encoding="utf-8")

        # Build a small runner to import the solution and run the tests.
        lines = [
            "import json, sys",
            "import importlib.util, runpy",
            "spec = importlib.util.spec_from_file_location('solution', sys.argv[1])",
            "mod = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(mod)",
            f"fn = getattr(mod, '{entry_point}', None)",
            "passed = 0",
            "total = 0",
            "last_error = ''",
            "for case in json.loads(sys.argv[2]):",
            "    total += 1",
            "    args = case.get('input', [])",
            "    expected = case.get('output')",
            "    try:",
            "        got = fn(*args)",
            "        if got == expected:",
            "            passed += 1",
            "        else:",
            "            last_error = f'expected {expected}, got {got}'",
            "    except Exception as e:",
            "        last_error = str(e)",
            "print(json.dumps({'passed': passed, 'total': total, 'last_error': last_error}))",
        ]
        runner_path.write_text("\n".join(lines), encoding="utf-8")

        proc = subprocess.run(
            [sys.executable, str(runner_path), str(solution_path), json.dumps(test_cases)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            return 0, len(test_cases), proc.stderr.strip() or proc.stdout.strip()
        try:
            data = json.loads(proc.stdout.strip())
        except Exception:
            return 0, len(test_cases), proc.stdout.strip()
        return int(data.get('passed', 0)), int(data.get('total', len(test_cases))), str(data.get('last_error', ''))


def _run_test_code(code: str, test_code: str, timeout: float = 10.0) -> tuple[bool, str]:
    """Run HumanEval-style test program against the solution.

    The `test_code` usually imports from solution.py (e.g., `from solution import entry_point`) and
    performs assertions. We treat "exit code 0" as pass, otherwise failed with captured stderr/stdout.
    Returns (passed, last_error).
    """
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        solution_path = td_path / "solution.py"
        test_path = td_path / "test_program.py"
        runner_path = td_path / "runner.py"

        solution_path.write_text(code, encoding="utf-8")
        test_path.write_text(test_code, encoding="utf-8")

        runner_lines = [
            "import sys,runpy",
            # Ensure current temp dir (with solution.py) is on sys.path for imports
            "sys.path.insert(0, sys.argv[1])",
            # Execute test program; any AssertionError or exception should bubble up
            "runpy.run_path(sys.argv[2], run_name='__main__')",
        ]
        runner_path.write_text("\n".join(runner_lines), encoding="utf-8")

        proc = subprocess.run(
            [sys.executable, str(runner_path), str(td_path), str(test_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode == 0:
            return True, ""
        last_err = proc.stderr.strip() or proc.stdout.strip()
        return False, last_err


class CodeEvalEnv(BaseEnv):
    def __init__(self, reward_fn, max_turn: int = 3, timeout: float = 5.0):
        self.task = None
        self.turn_count = 0
        self.max_turn = max_turn
        self.reward_fn = reward_fn
        self.timeout = timeout

    def reset(self, task):
        self.task = task
        observation = {"question": task.get("question")}
        info = {"task_id": task.get("task_id")}
        self.turn_count = 0
        return observation, info

    def step(self, action):
        reward_output: RewardOutput = self.reward_fn(self.task, action, timeout=self.timeout)
        self.turn_count += 1
        if reward_output.is_correct:
            done = True
            observation = {"feedback": "success"}
        else:
            if self.turn_count < self.max_turn:
                done = False
                observation = {"feedback": reward_output.metadata.get('last_error', 'incorrect')} if reward_output.metadata else {"feedback": "incorrect"}
            else:
                done = True
                observation = {"feedback": "exceeds max turn"}

        info = {
            "is_correct": reward_output.is_correct,
            **(reward_output.metadata or {}),
        }
        return observation, reward_output.reward, done, info
