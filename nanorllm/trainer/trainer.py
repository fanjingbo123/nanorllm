import logging
import time

from nanorllm.algos.grpo import  compute_advantage, group_by_task_id
from nanorllm.core.trajectory import Rollout
from nanorllm.trainer.collate import collate_train_batch, transform_episode_samples, transform_step_samples
from nanorllm.trainer.loss import compute_policy_loss
from nanorllm.rollout.collector import execute_tasks
import torch

logger = logging.getLogger(__name__)



def build_samples_from_rollouts(
    rollouts: list[Rollout],
    policy,
    args):
  
    grouped_rollouts = group_by_task_id(rollouts)
    training_rollouts = compute_advantage(
        grouped_rollouts
    )
    samples = []
    if args.mode == 'step':
        for rollout in training_rollouts:
            samples.extend(transform_step_samples(rollout)) 
    elif args.mode == 'prefix-compatible-episode-as-sequence':
        for rollout in training_rollouts:
            samples.append(transform_episode_samples(rollout, policy.tokenize_messages)) 
    else:
        raise ValueError(f"Unsupported training view mode: {args.mode}")
    return samples




def aggregate_train_metrics(minibatch_metrics):
    if not minibatch_metrics:
        return {"loss": 0.0, "avg_advantage": 0.0}

    weighted_loss_sum = 0.0
    weighted_advantage_sum = 0.0
    total_samples = 0

    for minibatch_metric in minibatch_metrics:
        num_samples = minibatch_metric["num_samples"]
        weighted_loss_sum += float(minibatch_metric["loss"].item()) * num_samples
        weighted_advantage_sum += float(minibatch_metric["advantage"].item()) * num_samples
        total_samples += num_samples

    return {
        "loss": weighted_loss_sum / total_samples,
        "avg_advantage": weighted_advantage_sum / total_samples,
    }


def iter_minibatches(samples, train_batch_size):
    for i in range(0, len(samples), train_batch_size):
        yield samples[i: i+train_batch_size]
    

def run_train_epoch(
    tasks,
    rollout_fn,
    policy,
    tokenizer,
    optimizer,
    args,
    *,
    show_progress: bool = False,
):

    logger.info(
        "Starting train epoch: tasks=%s samples_per_task=%s max_steps=%s max_new_tokens=%s",
        len(tasks),
        args.num_samples_per_task,
        args.max_steps,
        args.max_new_tokens,
    )
    rollouts = execute_tasks(tasks, args.num_samples_per_task, rollout_fn, show_progress=show_progress)
    logger.info("Collected %s rollouts", len(rollouts))
    samples = build_samples_from_rollouts(rollouts, policy, args)
    logger.info("Built %s train samples for mode=%s", len(samples), args.mode)
    train_start = time.perf_counter()

    minibatch_metrics = []
    batch = None

    if not samples:
        metrics = aggregate_train_metrics(minibatch_metrics)
        logger.info(
            "Finished train epoch in %.2fs with metrics=%s",
            time.perf_counter() - train_start,
            metrics,
        )
        trajectories = [rollout.trajectory for rollout in rollouts]
        return {
            "rollouts": rollouts,
            "trajectories": trajectories,
            "samples": samples,
            "batch": batch,
            "metrics": metrics,
        }

    policy.model.train()
    for batch_samples in iter_minibatches(samples, args.train_batch_size):
        if batch_samples:
            batch = collate_train_batch(batch_samples, tokenizer, args, device=policy.device)
            optimizer.zero_grad()

            logger.info("Running train step on batch with shape=%s", tuple(batch["input_ids"].shape))
            logits = policy.forward(batch['input_ids'], batch['attention_mask'])
            loss = compute_policy_loss(logits, batch, args)

            loss.backward()
            optimizer.step()
            metrics = {
                        "loss": loss.detach(),
                        "advantage": batch['advantages'].detach().mean(),
                        "num_samples": int(batch['advantages'].shape[0]),
                    }
            minibatch_metrics.append(metrics)
    metrics = aggregate_train_metrics(minibatch_metrics)

    logger.info(
        "Finished train epoch in %.2fs with metrics=%s",
        time.perf_counter() - train_start,
        metrics,
    )
    trajectories = [rollout.trajectory for rollout in rollouts]
    return {
        "rollouts": rollouts,
        "trajectories": trajectories,
        "samples": samples,
        "batch": batch,
        "metrics": metrics,
    }


def run_unlearn_epoch(
    forget_tasks,
    retain_tasks,
    rollout_fn,
    policy,
    ref_policy,
    tokenizer,
    optimizer,
    args,
    *,
    show_progress: bool = False,
):
    """One epoch of agentic RL unlearning.

    Collects rollouts from both forget and retain task distributions independently,
    inverts forget-side rewards for GRPO-based return minimization, and trains with
    a combined PPO + KL loss where KL regularisation applies only to retain samples.
    """
    import torch

    from nanorllm.algos.grpo import build_unlearn_samples_from_rollouts
    from nanorllm.rollout.collector import execute_tasks
    from nanorllm.trainer.collate import collate_train_batch
    from nanorllm.trainer.loss import compute_unlearn_policy_loss

    logger.info(
        "Starting unlearn epoch: forget_tasks=%s retain_tasks=%s samples_per_task=%s",
        len(forget_tasks),
        len(retain_tasks),
        args.num_samples_per_task,
    )

    # Collect rollouts from both distributions
    forget_rollouts = execute_tasks(forget_tasks, args.num_samples_per_task, rollout_fn, show_progress=show_progress)
    logger.info("Collected %s forget rollouts", len(forget_rollouts))
    retain_rollouts = execute_tasks(retain_tasks, args.num_samples_per_task, rollout_fn, show_progress=show_progress)
    logger.info("Collected %s retain rollouts", len(retain_rollouts))

    samples = build_unlearn_samples_from_rollouts(
        forget_rollouts, retain_rollouts, policy, args
    )
    logger.info("Built %s unlearn train samples for mode=%s", len(samples), args.mode)

    # Precompute ref token logprobs for retain samples on reference device (CPU).
    # This avoids full-vocab logits transfer and per-batch CPU forward during training.
    for s in samples:
        if s.metadata.get("is_retain", False):
            ids = s.input_ids
            if ids.shape[0] > args.max_length:
                ids = ids[-args.max_length:]
            ref_probs = ref_policy.get_token_logprobs(
                ids.unsqueeze(0),
                torch.ones_like(ids).unsqueeze(0),
                ids.unsqueeze(0),
                args,
            )
            s.metadata["ref_logprobs"] = ref_probs.squeeze(0).cpu()

    train_start = time.perf_counter()
    minibatch_metrics = []

    if not samples:
        metrics = aggregate_train_metrics(minibatch_metrics)
        logger.info(
            "Finished unlearn epoch in %.2fs with metrics=%s",
            time.perf_counter() - train_start,
            metrics,
        )
        return {
            "forget_rollouts": forget_rollouts,
            "retain_rollouts": retain_rollouts,
            "samples": samples,
            "metrics": metrics,
        }

    policy.model.train()
    for batch_samples in iter_minibatches(samples, args.train_batch_size):
        if not batch_samples:
            continue
        batch = collate_train_batch(batch_samples, tokenizer, args, device=policy.device)
        is_retain = torch.tensor(
            [bool(s.metadata.get("is_retain", False)) for s in batch_samples],
            device=policy.device,
        )
        batch["is_retain"] = is_retain

        # Extract precomputed ref_logprobs and pad to batch length
        ref_list = []
        batch_len = batch["old_logprobs"].shape[1]
        for s in batch_samples:
            rp = s.metadata.get("ref_logprobs")
            if rp is None:
                rp = torch.zeros(batch_len)
            else:
                pad_len = batch_len - rp.shape[0]
                if pad_len > 0:
                    rp = torch.nn.functional.pad(rp, (0, pad_len), value=0.0)
            ref_list.append(rp)
        batch["ref_logprobs"] = torch.stack(ref_list, dim=0).to(policy.device)

        optimizer.zero_grad()
        logits = policy.forward(batch["input_ids"], batch["attention_mask"])
        loss = compute_unlearn_policy_loss(logits, batch, args)
        loss.backward()
        optimizer.step()

        metric = {
            "loss": loss.detach(),
            "advantage": batch["advantages"].detach().mean(),
            "num_samples": int(batch["advantages"].shape[0]),
        }
        minibatch_metrics.append(metric)

    metrics = aggregate_train_metrics(minibatch_metrics)
    logger.info(
        "Finished unlearn epoch in %.2fs with metrics=%s",
        time.perf_counter() - train_start,
        metrics,
    )
    return {
        "forget_rollouts": forget_rollouts,
        "retain_rollouts": retain_rollouts,
        "samples": samples,
        "metrics": metrics,
    }
