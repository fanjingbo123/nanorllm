import torch

from nanorllm.policy.base import BasePolicy
from nanorllm.policy.hf_causal import HFCausalPolicy


class ReferencePolicy(BasePolicy):
    """Frozen copy of an HFCausalPolicy, used as π_θ_0 for KL regularization."""

    def __init__(self, policy: HFCausalPolicy):
        super().__init__(model_name=policy.model_name, device=policy.device)
        self._policy = policy
        self._policy.model.eval()
        for p in self._policy.model.parameters():
            p.requires_grad = False

    @property
    def model(self) -> torch.nn.Module:
        return self._policy.model

    @property
    def tokenizer(self):
        return self._policy.tokenizer

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        with torch.no_grad():
            return self._policy.forward(input_ids, attention_mask)
