"""MMLU JSONL loader.

Expected JSONL fields per line (common export shape):
  - question: str
  - choices: list[str]  (length typically 4)
  - answer: "A"|"B"|... or integer index (0-based or 1-based)

Outputs unified task schema:
  {"task_id", "question", "choices": list[str], "answer": "A"}
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


LETTERS = ["A", "B", "C", "D", "E", "F"]


def _normalize_answer(ans, num_choices: int) -> str:
    if isinstance(ans, str):
        ans = ans.strip()
        if ans.upper() in LETTERS[:num_choices]:
            return ans.upper()
        # Sometimes the raw letter is lowercase
        return ans.upper()
    if isinstance(ans, int):
        # Index may be 0-based or 1-based; clamp within range
        if 0 <= ans < num_choices:
            return LETTERS[ans]
        if 1 <= ans <= num_choices:
            return LETTERS[ans - 1]
    return ""


def get_mmlu_tasks_from_jsonl(path: str, split: str = "val", limit: int | None = None) -> list[dict]:
    tasks: list[dict] = []
    for idx, row in enumerate(_iter_jsonl(path)):
        question = str(row.get("question", ""))
        choices = [str(x).strip() for x in (row.get("choices") or [])]
        answer = _normalize_answer(row.get("answer", ""), len(choices))
        task_id = f"mmlu-{split}-{idx + 1:06d}"
        tasks.append({
            "task_id": task_id,
            "question": question,
            "choices": choices,
            "answer": answer,
        })
        if limit is not None and len(tasks) >= limit:
            break
    return tasks

