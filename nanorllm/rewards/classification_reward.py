import re
from nanorllm.core.types import RewardOutput


CHOICE_LETTERS = ["A", "B", "C", "D", "E", "F"]


def extract_choice_letter(text: str, num_choices: int) -> str | None:
    """Extract a single capital choice letter from model output.

    Accepts plain letters (A-D/E), optional boxing (\\boxed{A}), or formats like
    "(A)", "option A", etc. Returns the last valid letter found.
    """
    if not text:
        return None
    # Common wrappers
    m = re.findall(r"\\boxed\s*\{\s*([A-F])\s*\}", text)
    if m:
        letter = m[-1].upper()
        idx = CHOICE_LETTERS.index(letter)
        if idx < num_choices:
            return letter

    # Plain capital letter tokens, choose the last within range
    letters = re.findall(r"\b([A-F])\b", text)
    for letter in reversed(letters):
        letter = letter.upper()
        if letter in CHOICE_LETTERS[:num_choices]:
            return letter

    # Formats like (A) or option A
    letters = re.findall(r"\(([^)])\)", text)
    for letter in reversed(letters):
        letter = letter.upper()
        if letter in CHOICE_LETTERS[:num_choices]:
            return letter
    return None


def classification_reward(task, action) -> RewardOutput:
    choices = task.get("choices") or []
    gold = str(task.get("answer", "")).strip().upper()
    pred = extract_choice_letter(str(action.value), len(choices))
    is_correct = (pred == gold)
    return RewardOutput(
        reward=1 if is_correct else 0,
        is_correct=is_correct,
        metadata={
            "predicted_choice": pred,
            "expected_choice": gold,
        },
    )

