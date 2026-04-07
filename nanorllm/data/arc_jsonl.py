"""ARC (AI2 Reasoning Challenge) JSONL loader.

Expected JSONL fields per line (robust to common variants):
  - question or question_stem: str
  - choices: list of {"label": "A"|... , "text": str} or list[str]
  - answerKey or answer: "A"|... (capital letter)

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


def _normalize_choices(obj) -> list[str]:
    if isinstance(obj, list):
        if obj and isinstance(obj[0], dict):
            # Typical ARC format: {"label": "A", "text": "..."}
            return [str(x.get("text", "")).strip() for x in obj]
        return [str(x).strip() for x in obj]
    return []


def get_arc_tasks_from_jsonl(path: str, split: str = "train", limit: int | None = None) -> list[dict]:
    tasks: list[dict] = []
    for idx, row in enumerate(_iter_jsonl(path)):
        question = str(row.get("question", row.get("question_stem", "")))
        choices = _normalize_choices(row.get("choices", []))
        answer = str(row.get("answerKey", row.get("answer", ""))).strip().upper()
        task_id = f"arc-{split}-{idx + 1:06d}"
        tasks.append({
            "task_id": task_id,
            "question": question,
            "choices": choices,
            "answer": answer,
        })
        if limit is not None and len(tasks) >= limit:
            break
    return tasks

