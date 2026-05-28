import random


def split_by_task_ids(tasks: list[dict], forget_ids: set[str]) -> tuple[list[dict], list[dict]]:
    """Split tasks by explicit task_id list: forget_ids -> forget, rest -> retain."""
    forget = [t for t in tasks if t["task_id"] in forget_ids]
    retain = [t for t in tasks if t["task_id"] not in forget_ids]
    return forget, retain


def split_by_ratio(tasks: list[dict], forget_ratio: float, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Randomly split tasks into forget/retain by ratio."""
    rng = random.Random(seed)
    indices = list(range(len(tasks)))
    rng.shuffle(indices)
    split = int(len(tasks) * forget_ratio)
    forget = [tasks[i] for i in indices[:split]]
    retain = [tasks[i] for i in indices[split:]]
    return forget, retain


def split_by_subject(tasks: list[dict], forget_subjects: set[str]) -> tuple[list[dict], list[dict]]:
    """Split by subject field: forget_subjects → forget, rest → retain."""
    forget = [t for t in tasks if t.get("subject", "") in forget_subjects]
    retain = [t for t in tasks if t.get("subject", "") not in forget_subjects]
    if not forget:
        raise ValueError(f"No tasks found for forget_subjects={forget_subjects}")
    if not retain:
        raise ValueError("Retain set is empty")
    return forget, retain


def split_by_library(tasks: list[dict], forget_libraries: set[str]) -> tuple[list[dict], list[dict]]:
    """Split by library field."""
    if not forget_libraries:
        raise ValueError("split_mode='library' requires forget_libraries")
    for t in tasks:
        if not t.get("library"):
            raise ValueError(f"Task {t.get('task_id')} has no library field")
    forget = [t for t in tasks if t["library"] in forget_libraries]
    retain = [t for t in tasks if t["library"] not in forget_libraries]
    if not forget:
        raise ValueError(f"No tasks found for forget_libraries={forget_libraries}")
    if not retain:
        raise ValueError("Retain set is empty")
    return forget, retain


def split_tasks(tasks: list[dict], args) -> tuple[list[dict], list[dict]]:
    """Unified entry: dispatch to split_by_task_ids or split_by_ratio based on args."""
    if args.split_mode == "task_ids":
        if not args.forget_task_ids:
            raise ValueError("split_mode='task_ids' requires forget_task_ids to be set")
        ids = set(args.forget_task_ids)
        return split_by_task_ids(tasks, ids)
    elif args.split_mode == "ratio":
        return split_by_ratio(tasks, args.forget_ratio, args.split_seed)
    elif args.split_mode == "subject":
        if not args.forget_subjects:
            raise ValueError("split_mode='subject' requires forget_subjects to be set")
        return split_by_subject(tasks, set(args.forget_subjects))
    elif args.split_mode == "library":
        if not args.forget_libraries:
            raise ValueError("split_mode='library' requires forget_libraries to be set")
        return split_by_library(tasks, set(args.forget_libraries))
    else:
        raise ValueError(f"Unknown split_mode: {args.split_mode}")
