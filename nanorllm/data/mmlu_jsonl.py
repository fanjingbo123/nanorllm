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
import random
from pathlib import Path
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
        return ans.upper()
    if isinstance(ans, int):
        if 0 <= ans < num_choices:
            return LETTERS[ans]
        if 1 <= ans <= num_choices:
            return LETTERS[ans - 1]
    return ""


def get_mmlu_tasks_from_jsonl(
    path: str, split: str = "val",
    limit: int | None = None,
    subject_hint: str | None = None,
) -> list[dict]:
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
            "subject": subject_hint or row.get("subject", ""),
        })
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions",
]


def get_mmlu_tasks_per_subject(
    cache_root: Path,
    splits: list[str] | None = None,
    subjects: list[str] | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Load MMLU per-subject, merge all subjects across splits, shuffle, truncate."""
    from nanorllm.datasets_auto import ensure_local_jsonl

    if splits is None:
        splits = ["validation"]
    if subjects is None:
        subjects = MMLU_SUBJECTS

    all_tasks: list[dict] = []
    for subject in subjects:
        for split in splits:
            jsonl_path = ensure_local_jsonl(
                "mmlu-jsonl", config=subject, split=split,
                cache_root=cache_root,
            )
            tasks = get_mmlu_tasks_from_jsonl(
                str(jsonl_path), split=split, subject_hint=subject,
            )
            # Re-ID per-subject to avoid collisions
            for i, t in enumerate(tasks):
                t["task_id"] = f"mmlu-{subject}-{split}-{i + 1:06d}"
            all_tasks.extend(tasks)

    random.shuffle(all_tasks)
    if limit is not None:
        all_tasks = all_tasks[:limit]
    return all_tasks

