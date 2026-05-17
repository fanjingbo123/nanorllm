from nanorllm.core.trajectory import StepRolloutView, Rollout


TERMINATION_REASON_MAX_STEPS = "max_steps"


class RolloutEngine:
    def run_episode(self, agent, env, llm, task, args):
        agent.reset()
        observation, info = env.reset(task)
        reward = 0.0
        done = False
        agent.update_from_env(observation, reward, done, info)

        step_model_outputs = []
        for i in range(args.max_steps):
            model_output = llm.generate(agent.messages, args)
            step_model_outputs.append(
                StepRolloutView(
                    prompt_ids=model_output['prompt_ids'],
                    response_ids=model_output['response_ids'],
                    response_logprobs=model_output['response_logprobs'],
                )
            )

            action = agent.update_from_model(model_output['text'])
            observation, reward, done, info = env.step(action)
            if i == args.max_steps - 1 and not done:
                info = info or {}
                info['termination_reason'] = TERMINATION_REASON_MAX_STEPS

            agent.update_from_env(observation, reward, done, info)
            if done:
                break

        if not agent.trajectory.terminated:
            agent.trajectory.termination_reason = TERMINATION_REASON_MAX_STEPS
            agent.trajectory.terminated = True

        return Rollout(
            trajectory=agent.trajectory,
            step_views=step_model_outputs,
            task=task,
            metadata={'env_name': 'MathEnv'},
        )


def run_episodes_batch(policy, agents, envs, tasks, args, *, pbar=None) -> list[Rollout]:
    """Run multiple episodes in parallel, calling generate_batch at each step.

    Each episode uses an independent agent/env instance.  The termination
    semantics (done / max_steps / info['termination_reason']) match the serial
    RolloutEngine.run_episode.
    """
    for i, task in enumerate(tasks):
        agents[i].reset()
        obs, info = envs[i].reset(task)
        agents[i].update_from_env(obs, 0.0, False, info)

    active = list(range(len(tasks)))
    step_views = {i: [] for i in range(len(tasks))}

    for step in range(args.max_steps):
        if not active:
            break
        messages = [agents[i].messages for i in active]
        outputs = policy.generate_batch(messages, args)

        new_done = []
        for j, i in enumerate(active):
            output = outputs[j]
            step_views[i].append(StepRolloutView(
                prompt_ids=output['prompt_ids'],
                response_ids=output['response_ids'],
                response_logprobs=output['response_logprobs'],
            ))
            action = agents[i].update_from_model(output['text'])
            obs, reward, done, info = envs[i].step(action)
            if step == args.max_steps - 1 and not done:
                info = info or {}
                info['termination_reason'] = TERMINATION_REASON_MAX_STEPS
            agents[i].update_from_env(obs, reward, done, info)
            if done or agents[i].trajectory.terminated:
                new_done.append(i)
        for i in new_done:
            active.remove(i)
        if pbar is not None:
            pbar.update(len(new_done))

    for i in active:
        if not agents[i].trajectory.terminated:
            agents[i].trajectory.terminated = True
            agents[i].trajectory.termination_reason = TERMINATION_REASON_MAX_STEPS
    if pbar is not None and active:
        pbar.update(len(active))

    return [Rollout(
        trajectory=agents[i].trajectory,
        step_views=step_views[i],
        task=tasks[i],
        metadata={'env_name': type(envs[i]).__name__},
    ) for i in range(len(tasks))]
