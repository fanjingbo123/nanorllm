"""GSM8K JSONL loader that converts records to the nanorllm task schema.

Expected input: a JSONL file where each line has at least:
  - "question": str
  - "answer": str  (GSM8K style often contains the final answer after '#### ')

This loader extracts the final numeric/text answer using a simple rule:
  - If '#### <final>' appears, take the part after ####
  - Otherwise, fall back to the whole answer string
Then it normalizes via nanorllm.rewards.math_reward.normalize_math_answer to
match reward-side normalization behavior.
"""

from __future__ import annotations

import json
import re
from typing import Iterable

from nanorllm.rewards.math_reward import normalize_math_answer


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


_GSM8K_FINAL_RE = re.compile(r"####\s*([^\n]+)")


def _extract_gsm8k_final_answer(answer_text: str) -> str:
    """Extract the final answer from a GSM8K-style solution string.

    Examples:
      "We compute ... Therefore the answer is 45.\n#### 45" -> "45"
    """
    m = _GSM8K_FINAL_RE.search(answer_text or "")
    if m:
        candidate = m.group(1).strip()
    else:
        candidate = (answer_text or "").strip()
    return normalize_math_answer(candidate)


def get_gsm8k_tasks_from_jsonl(path: str, split: str = "train", limit: int | None = None) -> list[dict[str, str]]:
    """Load GSM8K records from a JSONL file into the task schema.

    Returns a list of {"task_id", "question", "answer"} dicts.
    """
    tasks: list[dict[str, str]] = []
    for idx, row in enumerate(_iter_jsonl(path)):
        question = str(row.get("question", ""))
        raw_answer = str(row.get("answer", ""))
        answer = _extract_gsm8k_final_answer(raw_answer)
        task_id = f"gsm8k-{split}-{idx + 1:06d}"
        tasks.append({"task_id": task_id, "question": question, "answer": answer})
        if limit is not None and len(tasks) >= limit:
            break
    return tasks

