"""DS-1000 JSONL loader.

Expected JSONL fields (flat or nested under "metadata"):
  - prompt: str
  - code_context: str
  - library: str
  - problem_id: int | str
  - reference_code: str (optional)
  - perturbation_type: str (optional)
  - perturbation_origin_id: int | None (optional)

Outputs unified task schema:
  {"task_id", "question": prompt, "code_context", "reference_code",
   "library", "perturbation_type", "perturbation_origin_id"}
"""

from __future__ import annotations

import json
import random
from typing import Iterable


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def get_ds1000_tasks_from_jsonl(
    path: str,
    limit: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> list[dict]:
    tasks: list[dict] = []
    for row in _iter_jsonl(path):
        meta = row.get("metadata") or {}
        library = row.get("library", meta.get("library", ""))
        problem_id = row.get("problem_id", meta.get("problem_id", ""))
        prompt = row.get("prompt", meta.get("prompt", ""))
        code_context = row.get("code_context", meta.get("code_context", ""))

        if not library:
            raise ValueError(f"Missing library for row {row}")
        if not prompt:
            raise ValueError(f"Missing prompt for row {row}")
        if not code_context:
            raise ValueError(f"Missing code_context for row {row}")

        tasks.append({
            "task_id": f"ds1000-{library}-{problem_id}",
            "question": prompt,
            "code_context": code_context,
            "reference_code": row.get("reference_code", meta.get("reference_code", "")),
            "library": library,
            "perturbation_type": row.get("perturbation_type", meta.get("perturbation_type", "")),
            "perturbation_origin_id": row.get("perturbation_origin_id", meta.get("perturbation_origin_id")),
        })

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(tasks)
    if limit is not None:
        tasks = tasks[:limit]
    return tasks
