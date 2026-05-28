import torch

from nanorllm.policy.base import BasePolicy
from nanorllm.policy.hf_causal import HFCausalPolicy


class ReferencePolicy(BasePolicy):
    """Frozen copy of a trained HFCausalPolicy, used as π_θ_0 for KL regularization.

    Creates an independent model instance, copies the trained policy's
    state_dict, and moves the model to a (possibly different) device.
    """

    def __init__(self, policy: HFCausalPolicy, *, prefer_offline: bool = False, offline_cache_dir: str | None = None, device: str | None = None):
        self._device = device or policy.device
        super().__init__(model_name=policy.model_name, device=self._device)
        # Load directly on the target device to avoid GPU memory spike
        self._policy = HFCausalPolicy(
            model_name=policy.model_name,
            device=self._device,
            prefer_offline=prefer_offline,
            offline_cache_dir=offline_cache_dir,
        )
        # load_state_dict handles device transfer automatically
        self._policy.model.load_state_dict(policy.model.state_dict())
        self._policy.model.eval()
        for p in self._policy.model.parameters():
            p.requires_grad = False

    @property
    def model(self) -> torch.nn.Module:
        return self._policy.model

    @property
    def tokenizer(self):
        return self._policy.tokenizer

    @property
    def device(self) -> str:
        return self._device

    def to(self, device: str):
        self._device = device
        self._policy.model.to(device)
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        orig_device = input_ids.device
        with torch.no_grad():
            input_ids = input_ids.to(self._device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self._device)
            logits = self._policy.forward(input_ids, attention_mask)
            return logits.to(orig_device)

    def get_token_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        labels: torch.Tensor,
        args,
    ) -> torch.Tensor:
        """Forward + log_softmax + gather on reference device, return [B,T]."""
        orig_device = input_ids.device
        with torch.no_grad():
            input_ids = input_ids.to(self._device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self._device)
            labels = labels.to(self._device)
            logits = self._policy.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
            from nanorllm.trainer.loss import compute_token_logprobs

            token_probs = compute_token_logprobs(logits, labels, args)
            return token_probs.to(orig_device)
