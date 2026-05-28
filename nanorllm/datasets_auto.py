"""Auto-download HF datasets to local JSONL under datasets/ and reuse thereafter.

Supported presets:
  - "gsm8k-jsonl" -> hf dataset: ("gsm8k", config="main"), split default "train"
  - "arc-jsonl"   -> hf dataset: ("ai2_arc", config="ARC-Challenge"), split default "train"
  - "mmlu-jsonl"  -> hf dataset: ("cais/mmlu", no config), split default "validation" (aka dev)

Writes files to: {cache_root}/{name}/{config}/{split}.jsonl (config optional).
If the file already exists, it is reused and no network call is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _default_cache_root() -> Path:
    # Place under repo datasets/auto_cache by default
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "datasets" / "auto_cache"


def _preset_to_hf(preset: str, config: Optional[str], split: Optional[str]) -> Tuple[str, Optional[str], str]:
    p = preset.lower()
    if p == "gsm8k-jsonl":
        return "gsm8k", (config or "main"), (split or "train")
    if p == "arc-jsonl":
        return "ai2_arc", (config or "ARC-Challenge"), (split or "train")
    if p == "mmlu-jsonl":
        # MMLU requires a config (subject or "all"). Default to "all".
        # Default split: validation (aka dev).
        return "cais/mmlu", (config or "all"), (split or "validation")
    if p == "humaneval-jsonl":
        return "openai_humaneval", (config or None), (split or "test")
    if p == "ds1000-jsonl":
        return "xlangai/DS-1000", None, "test"
    raise ValueError(f"Unsupported preset for auto download: {preset}")


def _target_path(cache_root: Path, preset: str, hf_name: str, hf_config: Optional[str], split: str) -> Path:
    name_part = hf_name.replace("/", "--")
    if hf_config:
        return cache_root / name_part / hf_config / f"{split}.jsonl"
    return cache_root / name_part / f"{split}.jsonl"


def ensure_local_jsonl(
    preset: str,
    *,
    cache_root: Optional[str | Path] = None,
    config: Optional[str] = None,
    split: Optional[str] = None,
) -> Path:
    """Ensure a local JSONL exists for the given preset; return its path.

    Uses the `datasets` library to load from HF and convert to JSONL once.
    Subsequent calls reuse the JSONL from datasets/auto_cache.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "The 'datasets' package is required for auto download. Install via pip or set TrainArgs.dataset_path."
        ) from e

    hf_name, hf_config, hf_split = _preset_to_hf(preset, config, split)
    root = Path(cache_root) if cache_root is not None else _default_cache_root()
    target_jsonl = _target_path(root, preset, hf_name, hf_config, hf_split)
    if target_jsonl.exists():
        return target_jsonl

    _ensure_parent(target_jsonl)
    ds = load_dataset(hf_name, hf_config, split=hf_split)
    ds.to_json(str(target_jsonl), lines=True)
    return target_jsonl
