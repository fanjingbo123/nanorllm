import json
import logging
import platform
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load as load_safetensors
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from nanorllm.policy.base import BasePolicy
from nanorllm.utils.util import render_messages

logger = logging.getLogger(__name__)


def load_tokenizer(model_name: str, *, local_files_only: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _resolve_model_directory(model_name: str) -> Path:
    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return model_path if model_path.is_dir() else model_path.parent

    snapshot_path = snapshot_download(
        repo_id=model_name,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.safetensors.index.json",
        ],
    )
    return Path(snapshot_path)


def _resolve_safetensor_files(model_dir: Path) -> list[Path]:
    index_files = sorted(model_dir.glob("*.safetensors.index.json"))
    if index_files:
        index_data = json.loads(index_files[0].read_text())
        shard_names = []
        for shard_name in index_data["weight_map"].values():
            if shard_name not in shard_names:
                shard_names.append(shard_name)
        return [model_dir / shard_name for shard_name in shard_names]

    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors weights found under {model_dir}")
    return safetensor_files


def _load_safetensor_state_dict_eagerly(model_name: str) -> dict[str, torch.Tensor]:
    model_dir = _resolve_model_directory(model_name)
    state_dict = {}
    for shard_path in _resolve_safetensor_files(model_dir):
        with shard_path.open("rb") as handle:
            shard_state = load_safetensors(handle.read())
        state_dict.update(shard_state)
    return state_dict


def _build_model_from_config(model_name: str) -> torch.nn.Module:
    config = AutoConfig.from_pretrained(model_name)
    model_kwargs = {}
    if getattr(config, "torch_dtype", None) is not None:
        model_kwargs["dtype"] = config.torch_dtype
    return AutoModelForCausalLM.from_config(config, **model_kwargs)


def _load_model_eagerly(model_name: str) -> torch.nn.Module:
    model = _build_model_from_config(model_name)
    incompatible_keys = model.load_state_dict(
        _load_safetensor_state_dict_eagerly(model_name),
        strict=False,
    )
    if incompatible_keys.unexpected_keys:
        raise ValueError(
            f"Unexpected keys when loading {model_name}: {incompatible_keys.unexpected_keys}"
        )

    allowed_missing_keys = {"lm_head.weight"}
    unexpected_missing_keys = [
        key
        for key in incompatible_keys.missing_keys
        if key not in allowed_missing_keys
    ]
    if unexpected_missing_keys:
        raise ValueError(
            f"Missing keys when loading {model_name}: {unexpected_missing_keys}"
        )

    if hasattr(model, "tie_weights"):
        model.tie_weights()
    return model


def load_model(model_name: str, device: str, *, local_files_only: bool = False):
    if device == "cpu" and platform.system() == "Darwin":
        # Avoid safetensors mmap-backed weights on macOS CPU by
        # materializing the checkpoint into RAM before load_state_dict.
        try:
            logger.info("Loading model with eager safetensors materialization on macOS CPU: %s", model_name)
            return _load_model_eagerly(model_name)
        except FileNotFoundError:
            logger.info("No safetensors checkpoint found for %s, falling back to from_pretrained()", model_name)
            pass

    logger.info(
        "Loading model with from_pretrained(): %s (local_files_only=%s)",
        model_name,
        local_files_only,
    )
    return AutoModelForCausalLM.from_pretrained(model_name, local_files_only=local_files_only)


def _snapshot_to_local_dir(model_name: str, cache_root: str | None = None) -> Path:
    """Resolve a local snapshot path for the given repo.

    Behavior:
    - If `cache_root/models--{org}--{repo}/snapshots/<rev>` exists, reuse it.
    - Otherwise call `snapshot_download` to materialize a snapshot under the
      per-repo directory and return the created snapshot path.
    - If `cache_root` is None, fall back to the default HF cache location.
    """
    if cache_root:
        cache_root = Path(cache_root).expanduser()
        cache_root.mkdir(parents=True, exist_ok=True)
        try:
            org, repo = model_name.split("/", 1)
        except ValueError:
            org, repo = ("", model_name)
        local_repo_dir = cache_root / (f"models--{org}--{repo}" if org else f"models--{repo}")
        snapshots_dir = local_repo_dir / "snapshots"
        # Prefer an existing snapshot if available
        if snapshots_dir.exists():
            ref_main = local_repo_dir / "refs" / "main"
            if ref_main.exists():
                try:
                    rev = ref_main.read_text(encoding="utf-8").strip()
                    candidate = snapshots_dir / rev
                    if candidate.exists():
                        logger.info("Using existing local snapshot: %s", candidate)
                        return candidate
                except Exception:
                    pass
            candidates = [p for p in snapshots_dir.iterdir() if p.is_dir()]
            if candidates:
                candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                logger.info("Using latest local snapshot: %s", candidates[0])
                return candidates[0]

        # Fetch a fresh snapshot into the per-repo directory
        snapshot_path = snapshot_download(
            repo_id=model_name,
            local_dir=str(local_repo_dir),
            repo_type=None,
            allow_patterns=None,
        )
        logger.info("Downloaded snapshot to: %s", snapshot_path)
        return Path(snapshot_path)
    else:
        snapshot_path = snapshot_download(
            repo_id=model_name,
            repo_type=None,
            allow_patterns=None,
        )
        logger.info("Downloaded snapshot to default cache: %s", snapshot_path)
        return Path(snapshot_path)


class HFCausalPolicy(BasePolicy):
    def __init__(self, model_name: str, device: str, *, prefer_offline: bool = False, offline_cache_dir: str | None = None):
        super().__init__(model_name=model_name, device=device)
        if prefer_offline:
            logger.info("Prefer-offline loading enabled; snapshotting %s", model_name)
            local_dir = _snapshot_to_local_dir(model_name, offline_cache_dir)
            logger.info("Loading tokenizer/model from local snapshot: %s", local_dir)
            self._tokenizer = load_tokenizer(str(local_dir), local_files_only=True)
            self._model = load_model(str(local_dir), device, local_files_only=True)
        else:
            self._tokenizer = load_tokenizer(model_name)
            self._model = load_model(model_name, device)
        # Ensure the model tensors live on the requested device so that
        # training and generation work when args.device != "cpu".
        # Note: from_pretrained() loads on CPU by default unless a device_map is used.
        try:
            self._model.to(self.device)
        except Exception as e:
            logger.error("Failed to move model to device %s: %s", self.device, e)
            raise
        
        # Double-check and fix-up later if something unexpectedly re-created
        # parameters/buffers on a different device.
        self._assert_or_relocate_model_device()

    def _assert_or_relocate_model_device(self):
        try:
            param_device = next(self._model.parameters()).device
        except StopIteration:
            # No parameters (edge case) – assume already fine.
            return
        target = torch.device(self.device)
        if param_device != target:
            logger.warning(
                "Model parameters on %s but expected %s; relocating now.",
                param_device,
                target,
            )
            self._model.to(target)

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    def tokenize_messages(
        self,
        prompt_or_messages: str | list[dict[str, Any]],
        add_generation_prompt: bool = True,
    ) -> torch.Tensor:
        if isinstance(prompt_or_messages, str):
            tokenized = self._tokenizer(prompt_or_messages, add_special_tokens=False, return_tensors="pt")
            return tokenized["input_ids"]

        if hasattr(self._tokenizer, "apply_chat_template"):
            tokenized = self._tokenizer.apply_chat_template(
                prompt_or_messages, tokenize=True, add_generation_prompt=add_generation_prompt, return_tensors="pt"
            )
            if isinstance(tokenized, torch.Tensor):
                return tokenized
            return tokenized["input_ids"]

        prompt_text = render_messages(prompt_or_messages, add_generation_prompt=add_generation_prompt)
        tokenized = self._tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")
        return tokenized["input_ids"]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        # Ensure the model and inputs are colocated
        self._assert_or_relocate_model_device()
        return self._model(input_ids=input_ids, attention_mask=attention_mask).logits


    def _sample_token(self, logits: torch.Tensor, temperature: float):
        scaled_logits = logits / temperature # 不能忘
        probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
        token_id = torch.multinomial(probs, num_samples=1) # 从概率分布中采样一个token，暂时不考虑top-k或top-p等采样策略
        log_probs = torch.nn.functional.log_softmax(scaled_logits, dim=-1)
        token_log_prob = torch.gather(log_probs, index=token_id, dim=-1)
        return token_id, token_log_prob
    

    def generate(self, prompt_or_messages: str | list[dict[str, Any]], args):
        # Ensure model is on the intended device before any forward
        self._assert_or_relocate_model_device()
        response_ids = []
        response_logprobs = []
        prompt_ids = self.tokenize_messages(prompt_or_messages, add_generation_prompt=True)
        input_ids = prompt_ids.to(self.device)
        attention_mask = torch.ones_like(input_ids, device=self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values # kv cache

            for _ in range(args.max_new_tokens):
                token_id, token_log_prob = self._sample_token(logits, args.temperature)
                response_logprobs.append(token_log_prob)
                response_ids.append(token_id)

                if token_id.item() == self._tokenizer.eos_token_id:
                    break

                attention_mask = torch.concat([attention_mask, torch.ones_like(token_id)], dim=-1)
                outputs = self._model(input_ids=token_id, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True)
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

        if response_ids:
            response_logprobs = torch.stack(response_logprobs, dim=-1).view(-1)
            response_ids = torch.stack(response_ids, dim=-1).view(-1)
        else:
            response_logprobs = torch.empty(0, device=self.device)
            response_ids = torch.empty(0, dtype=torch.long, device=self.device)

        response_ids = response_ids.detach().cpu()
        response_logprobs = response_logprobs.detach().cpu()
        prompt_ids = prompt_ids.view(-1).detach().cpu()
        text = self._tokenizer.decode(response_ids, skip_special_tokens=True)
        return {
            "text": text,
            "response_ids": response_ids,
            "response_logprobs": response_logprobs,
            "prompt_ids": prompt_ids,
        }



if __name__ == '__main__':
    policy = HFCausalPolicy(model_name='openai-community/gpt2', device='cpu')
    results = policy.generate('''<SYSTEM>
You are a careful math problem solver. Think step by step when useful, and end with a clear final answer.

Follow these rules strictly:
1) Solve the question and return exactly one final answer wrapped in \boxed{...}.
2) In \boxed{...}, output only the final value/expression (no words, units, punctuation, or extra spaces).
3) Never output multiple boxed answers.

<USER>
17 + 28 = ?''', 10, 0.5)
