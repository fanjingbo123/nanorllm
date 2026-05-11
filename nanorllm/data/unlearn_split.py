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


def split_tasks(tasks: list[dict], args) -> tuple[list[dict], list[dict]]:
    """Unified entry: dispatch to split_by_task_ids or split_by_ratio based on args."""
    if args.split_mode == "task_ids":
        if not args.forget_task_ids:
            raise ValueError("split_mode='task_ids' requires forget_task_ids to be set")
        ids = set(args.forget_task_ids)
        return split_by_task_ids(tasks, ids)
    elif args.split_mode == "ratio":
        return split_by_ratio(tasks, args.forget_ratio, args.split_seed)
    else:
        raise ValueError(f"Unknown split_mode: {args.split_mode}")
