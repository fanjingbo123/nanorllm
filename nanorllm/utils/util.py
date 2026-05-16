import argparse
import logging
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any

def render_messages(
    messages: list[dict[str, Any]],
    add_generation_prompt: bool = False,
) -> str:
    rendered_messages: list[str] = []

    for message in messages or []:
        role = str(message.get("role", "user")).strip().upper() #先大写
        content = message.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        rendered_messages.append(f"<{role}>\n{content.strip()}")

    prompt_text = "\n\n".join(part for part in rendered_messages if part).strip()
    if add_generation_prompt:
        if prompt_text:
            prompt_text = f"{prompt_text}\n\n<ASSISTANT>\n"
        else:
            prompt_text = "<ASSISTANT>\n"
    return prompt_text



def to_jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        return {k: to_jsonable(v) for k, v in value.__dict__.items()}
    return value

def rollout_to_viewer_json(rollouts):
    trajectories = []

    for sample_index_global, rollout in enumerate(rollouts):
        trajectory = rollout.trajectory
        task = rollout.task or {}
        step_views = rollout.step_views or []

        steps = []
        for i, step in enumerate(trajectory.steps):
            training = None
            if i < len(step_views):
                s = step_views[i]
                training = {
                    "advantage": to_jsonable(rollout.advantage),
                    "prompt_ids": to_jsonable(s.prompt_ids),
                    "response_ids": to_jsonable(s.response_ids),
                    "response_logprobs": to_jsonable(s.response_logprobs),
                }

            steps.append({
                "index": i,
                "observation": to_jsonable(step.observation),
                "prompt_messages": to_jsonable(step.prompt_messages),
                "model_response": to_jsonable(step.model_response),
                "action": to_jsonable(step.action),
                "reward": to_jsonable(step.reward),
                "done": to_jsonable(step.done),
                "info": to_jsonable(step.info),
                "training": training,
            })

        trajectories.append({
            "task_id": trajectory.task_id,
            "sample_index": sample_index_global,
            "final_reward": trajectory.final_reward,
            "terminated": trajectory.terminated,
            "termination_reason": trajectory.termination_reason,
            "task": {
                "question": task.get("question"),
                "answer": task.get("answer"),
            },
            "steps": steps,
        })

    return {
        "meta": {
            "env_name": "MathEnv",
            "created_at": "2026-03-19T00:00:00+08:00",
            "source": "train_math_grpo",
            "num_trajectories": len(trajectories),
        },
        "trajectories": trajectories,
    }


def build_parser(dataclass_cls, description: str = "") -> argparse.ArgumentParser:
    """Build an argparse.ArgumentParser from a dataclass."""
    p = argparse.ArgumentParser(description=description)
    type_map = {int: int, float: float, str: str, bool: lambda x: x.lower() not in ("0", "false", "no")}
    for f in fields(dataclass_cls):
        flag = "--" + f.name.replace("_", "-")
        if f.type is bool:
            p.add_argument(flag, action="store_true", default=f.default, help=f"(default: {f.default})")
            p.add_argument("--no-" + f.name.replace("_", "-"), dest=f.name, action="store_false", help=f"disable {f.name}")
        elif f.type == list[str] | None or str(f.type).startswith("list"):
            p.add_argument(flag, type=str, default=f.default, help=f"comma-separated (default: {f.default})")
        elif f.type == int | None:
            p.add_argument(flag, type=int, default=f.default, help=f"(default: {f.default})")
        else:
            t = type_map.get(f.type, str)
            p.add_argument(flag, type=t, default=f.default, help=f"(default: {f.default})")
    return p


def parse_dataclass(parser: argparse.ArgumentParser, dataclass_cls):
    """Parse CLI args and return a dataclass instance, handling comma-separated lists."""
    cli = parser.parse_args()
    kwargs = {f.name: getattr(cli, f.name) for f in fields(dataclass_cls)}
    for f in fields(dataclass_cls):
        if f.type == list[str] | None or str(f.type).startswith("list"):
            val = kwargs[f.name]
            if isinstance(val, str) and val:
                kwargs[f.name] = [v.strip() for v in val.split(",")]
            elif isinstance(val, str) and not val:
                kwargs[f.name] = None
    return dataclass_cls(**kwargs)


def setup_file_logging(model_name: str, dataset: str, logger: logging.Logger) -> None:
    """Mirror console output to logs/<model>/<dataset>/<timestamp>.log."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / model_name / dataset
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{ts}.log"
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(fh)
    logger.info("Logging to %s", log_path)


def print_args_table(args, logger: logging.Logger, title: str = "Args") -> None:
    """Print a 4-column hyperparameter table via logger."""
    items = [(f.name, str(getattr(args, f.name))) for f in fields(args)]
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title=title, show_header=True, header_style="bold")
        table.add_column("Param", style="dim")
        table.add_column("Value")
        table.add_column("Param", style="dim")
        table.add_column("Value")
        for i in range(0, len(items), 2):
            left = items[i]
            right = items[i + 1] if i + 1 < len(items) else ("", "")
            table.add_row(left[0], left[1], right[0], right[1])

        console = Console(width=120, force_terminal=False)
        with console.capture() as capture:
            console.print(table)
        logger.info("\n%s", capture.get().rstrip())
    except ImportError:
        logger.info("%s: %s", title, args)
