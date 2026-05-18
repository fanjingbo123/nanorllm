import logging
import time

from nanorllm.algos.grpo import  compute_advantage, group_by_task_id
from nanorllm.core.trajectory import Rollout
from nanorllm.trainer.collate import collate_train_batch, transform_episode_samples, transform_step_samples
from nanorllm.trainer.loss import compute_policy_loss
from nanorllm.rollout.collector import _HAS_TQDM, execute_tasks
from nanorllm.utils.util import log_cuda
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
    agents=None,
    envs=None,
    show_progress: bool = False,
):
    if (agents is None) != (envs is None):
        raise ValueError("agents and envs must be provided together")

    logger.info(
        "Starting train epoch: tasks=%s samples_per_task=%s max_steps=%s max_new_tokens=%s",
        len(tasks),
        args.num_samples_per_task,
        args.max_steps,
        args.max_new_tokens,
    )
    if agents is not None:
        from nanorllm.rollout.collector import execute_tasks_batch
        rollouts = execute_tasks_batch(tasks, args.num_samples_per_task, policy, agents, envs, args, show_progress=show_progress)
    else:
        rollouts = execute_tasks(tasks, args.num_samples_per_task, rollout_fn, show_progress=show_progress)
    logger.info("Collected %s rollouts", len(rollouts))
    samples = build_samples_from_rollouts(rollouts, policy, args)
    logger.info("Built %s train samples for mode=%s", len(samples), args.mode)
    log_cuda("train-samples-built")
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
    log_cuda("train-loop-start")
    batch_iter = iter_minibatches(samples, args.train_batch_size)
    if show_progress:
        try:
            from tqdm import tqdm
            n_batches = (len(samples) + args.train_batch_size - 1) // args.train_batch_size
            batch_iter = tqdm(batch_iter, total=n_batches, desc="train", unit="batch")
        except ImportError:
            pass
    batch_idx = 0
    for batch_samples in batch_iter:
        if batch_samples:
            batch = collate_train_batch(batch_samples, tokenizer, args, device=policy.device)
            if batch_idx == 0:
                log_cuda("train-batch-0-fwd-before")
            optimizer.zero_grad()

            logits = policy.forward(batch['input_ids'], batch['attention_mask'])
            loss = compute_policy_loss(logits, batch, args)

            loss.backward()
            if batch_idx == 0:
                log_cuda("train-batch-0-bwd-after")
            gn = getattr(args, "max_grad_norm", 0)
            if gn > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), gn)
            optimizer.step()
            if batch_idx == 0:
                log_cuda("train-batch-0-step-after")
            elif batch_idx % 10 == 0:
                log_cuda(f"train-step-{batch_idx}")
            metrics = {
                        "loss": loss.item(),
                        "advantage": batch['advantages'].mean().item(),
                        "num_samples": int(batch['advantages'].shape[0]),
                    }
            minibatch_metrics.append(metrics)
            batch_idx += 1
    log_cuda("train-loop-end")
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
    forget_agents=None,
    forget_envs=None,
    retain_agents=None,
    retain_envs=None,
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

    if (forget_agents is None) != (forget_envs is None):
        raise ValueError("forget_agents and forget_envs must be provided together")
    if (retain_agents is None) != (retain_envs is None):
        raise ValueError("retain_agents and retain_envs must be provided together")

    logger.info(
        "Starting unlearn epoch: forget_tasks=%s retain_tasks=%s samples_per_task=%s",
        len(forget_tasks),
        len(retain_tasks),
        args.num_samples_per_task,
    )

    # Collect rollouts from both distributions
    if forget_agents is not None:
        from nanorllm.rollout.collector import execute_tasks_batch
        forget_rollouts = execute_tasks_batch(forget_tasks, args.num_samples_per_task, policy, forget_agents, forget_envs, args, show_progress=show_progress)
    else:
        forget_rollouts = execute_tasks(forget_tasks, args.num_samples_per_task, rollout_fn, show_progress=show_progress)
    logger.info("Collected %s forget rollouts", len(forget_rollouts))

    if retain_agents is not None:
        from nanorllm.rollout.collector import execute_tasks_batch
        retain_rollouts = execute_tasks_batch(retain_tasks, args.num_samples_per_task, policy, retain_agents, retain_envs, args, show_progress=show_progress)
    else:
        retain_rollouts = execute_tasks(retain_tasks, args.num_samples_per_task, rollout_fn, show_progress=show_progress)
    logger.info("Collected %s retain rollouts", len(retain_rollouts))

    samples = build_unlearn_samples_from_rollouts(
        forget_rollouts, retain_rollouts, policy, args
    )
    logger.info("Built %s unlearn train samples for mode=%s", len(samples), args.mode)

    # Precompute ref token logprobs for retain samples
    retain_samples = [s for s in samples if s.metadata.get("is_retain", False)]
    pbar = None
    if show_progress and _HAS_TQDM:
        from tqdm import tqdm
        pbar = tqdm(total=len(retain_samples), desc="ref-logprobs", unit="sample")

    log_cuda("unlearn-ref-pre")
    for s in retain_samples:
        ids = s.input_ids
        if ids.shape[0] > args.max_length:
            ids = ids[-args.max_length:]
        ref_probs = ref_policy.get_token_logprobs(
            ids.unsqueeze(0),
            torch.ones_like(ids).unsqueeze(0),
            ids.unsqueeze(0),
            args,
        )
        s.metadata["ref_logprobs"] = ref_probs.squeeze(0).detach().cpu()
        if pbar is not None:
            pbar.update(1)
    log_cuda("unlearn-ref-post")

    if pbar is not None:
        pbar.close()

    # ref_logprobs are now saved as CPU tensors in sample metadata;
    # move ref model off GPU to free ~12GB before training
    ref_policy.model.to("cpu")
    torch.cuda.empty_cache()
    log_cuda("unlearn-ref-release-after")

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
    log_cuda("train-loop-start")
    batch_iter = iter_minibatches(samples, args.train_batch_size)
    if show_progress:
        try:
            from tqdm import tqdm
            n_batches = (len(samples) + args.train_batch_size - 1) // args.train_batch_size
            batch_iter = tqdm(batch_iter, total=n_batches, desc="unlearn", unit="batch")
        except ImportError:
            pass
    batch_idx = 0
    for batch_samples in batch_iter:
        if not batch_samples:
            continue
        batch = collate_train_batch(batch_samples, tokenizer, args, device=policy.device)
        if batch_idx == 0:
            log_cuda("train-batch-0-fwd-before")
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
        if batch_idx == 0:
            log_cuda("train-batch-0-bwd-after")
        gn = getattr(args, "max_grad_norm", 0)
        if gn > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), gn)
        optimizer.step()
        if batch_idx == 0:
            log_cuda("train-batch-0-step-after")
        elif batch_idx % 10 == 0:
            log_cuda(f"train-step-{batch_idx}")

        metric = {
            "loss": loss.item(),
            "advantage": batch["advantages"].mean().item(),
            "num_samples": int(batch["advantages"].shape[0]),
        }
        minibatch_metrics.append(metric)
        batch_idx += 1

    log_cuda("train-loop-end")
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
