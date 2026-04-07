"""HumanEval-like JSONL loader.

Because multiple variants exist, we support a simple superset shape per line:
  - task_id: str (optional; auto-assigned if missing)
  - prompt: str (function signature + docstring)
  - entry_point: str (function name to implement)
  - test_cases: optional list of {"input": [args], "output": any}

Outputs task schema used by a simple code-eval env:
  {"task_id", "question": prompt, "entry_point", "test_cases": list|None}

If test_cases is missing, the downstream env can still run but will default to
reward=0 with metadata explaining "no_test_cases".
"""

from __future__ import annotations

import json
from typing import Iterable


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def get_humaneval_tasks_from_jsonl(path: str, split: str = "test", limit: int | None = None) -> list[dict]:
    tasks: list[dict] = []
    for idx, row in enumerate(_iter_jsonl(path)):
        prompt = str(row.get("prompt", ""))
        entry_point = str(row.get("entry_point", row.get("function_name", "")))
        test_cases = row.get("test_cases")  # optional structured tests
        test_code = row.get("test")  # original HumanEval test program
        task_id = row.get("task_id") or f"humaneval-{split}-{idx + 1:06d}"
        tasks.append({
            "task_id": task_id,
            "question": prompt,
            "entry_point": entry_point,
            "test_cases": test_cases,
            "test": test_code,
        })
        if limit is not None and len(tasks) >= limit:
            break
    return tasks
