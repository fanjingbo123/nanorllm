from collections import defaultdict
import math

from nanorllm.core.trajectory import Rollout


RELATIVE_SIGNAL_EPS = 1e-8


def has_relative_signal(
    rollout_group: list[Rollout],
    eps: float = RELATIVE_SIGNAL_EPS,
) -> bool:
    if not rollout_group:
        return False

    rewards = [rollout.trajectory.final_reward for rollout in rollout_group]
    return max(rewards) - min(rewards) > eps



def compute_advantage(
    grouped_rollouts: dict[str | None, list[Rollout]],
) -> list[Rollout]:
    rollouts = []
    for task_id, rollout_group in grouped_rollouts.items():
        # GRPO only learns from relative differences within the same task group.
        # If every rollout got (almost) the same reward, normalization would
        # collapse to ~0 and this group would not provide a useful preference signal.
        if not has_relative_signal(rollout_group):
            continue

        group_scores = [rollout.trajectory.final_reward for rollout in rollout_group]
        avg_reward = sum(group_scores) / len(group_scores)
        variance = sum((x-avg_reward)**2 for x in group_scores) / len(group_scores)
        std_reward = math.sqrt(variance)

        for rollout in rollout_group:
            rollout.advantage = (rollout.trajectory.final_reward - avg_reward) / (std_reward + RELATIVE_SIGNAL_EPS)
            rollouts.append(rollout)

    return rollouts



def group_by_task_id(rollouts: list[Rollout]) -> dict[str | None, list[Rollout]]:
    grouped_rollouts: dict[str | None, list[Rollout]] = defaultdict(list)
    for rollout in rollouts:
        grouped_rollouts[rollout.trajectory.task_id].append(rollout)
    return grouped_rollouts


def build_unlearn_samples_from_rollouts(
    forget_rollouts: list[Rollout],
    retain_rollouts: list[Rollout],
    policy,
    args,
):
    """Build TrainSamples for unlearning: invert forget rewards, retain standard GRPO.

    - forget_rollouts: final_reward inverted so GRPO pushes toward lower original return.
    - retain_rollouts: standard GRPO with original rewards maintains performance.
    - Each TrainSample is tagged with metadata['is_retain'] for KL loss filtering.
    """
    from nanorllm.trainer.collate import transform_episode_samples, transform_step_samples

    for r in forget_rollouts:
        r.trajectory.final_reward = -r.trajectory.final_reward
        r.metadata["task_type"] = "forget"
    for r in retain_rollouts:
        r.metadata["task_type"] = "retain"

    all_rollouts = forget_rollouts + retain_rollouts
    grouped = group_by_task_id(all_rollouts)
    training_rollouts = compute_advantage(grouped)

    samples = []
    if args.mode == "step":
        for rollout in training_rollouts:
            is_retain = rollout.metadata.get("task_type") == "retain"
            steps = transform_step_samples(rollout)
            for s in steps:
                s.metadata["is_retain"] = is_retain
            samples.extend(steps)
    elif args.mode == "prefix-compatible-episode-as-sequence":
        for rollout in training_rollouts:
            is_retain = rollout.metadata.get("task_type") == "retain"
            sample = transform_episode_samples(rollout, policy.tokenize_messages)
            sample.metadata["is_retain"] = is_retain
            samples.append(sample)
    else:
        raise ValueError(f"Unsupported training view mode: {args.mode}")
    return samples
