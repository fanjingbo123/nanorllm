from nanorllm.envs.base import BaseEnv
from nanorllm.core.types import RewardOutput


class MCQEnv(BaseEnv):
    def __init__(self, reward_fn, max_turn: int = 1):
        self.task = None
        self.turn_count = 0
        self.max_turn = max_turn
        self.reward_fn = reward_fn

    def reset(self, task):
        self.task = task
        observation = {"question": task.get("question"), "choices": task.get("choices")}
        info = {"task_id": task.get("task_id")}
        self.turn_count = 0
        return observation, info

    def step(self, action):
        reward_output: RewardOutput = self.reward_fn(self.task, action)
        self.turn_count += 1

        if reward_output.is_correct:
            done = True
            observation = {"feedback": "success"}
        else:
            if self.turn_count < self.max_turn:
                done = False
                observation = {
                    "feedback": "Incorrect. Reply with ONLY the capital letter of your choice (e.g., A)."
                }
            else:
                done = True
                observation = {"feedback": "exceeds max turn"}

        info = {
            "is_correct": reward_output.is_correct,
            **reward_output.metadata,
        }
        return observation, reward_output.reward, done, info

