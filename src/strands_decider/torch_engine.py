"""The PyTorch engine: cuda, mps and cpu.

Its shared-prefix path encodes the state once and forks the KV cache across the question
batch; the forking below handles transformers 5's per-layer cache objects and the
recurrent states of hybrid torsos.
"""

from __future__ import annotations

import copy
from typing import Any

import torch

from .infer import EngineConfig, SystemOneEngine, UnforkableCache, _option_token_index
from .modeling import (
    StrandsDeciderConfig,
    StrandsDeciderModel,
    apply_temperature,
    masked_log_softmax,
    pool_last_token,
)
from .prompting import RenderedQuestion

# Per-row state a cache layer may hold (transformers 5): attention layers' `keys` and
# `values`; a linear-attention (Gated DeltaNet) layer's `conv_states` and
# `recurrent_states`, each a dict of tensors keyed by state index. First dim is the batch.
_ROW_STATES = ("keys", "values", "conv_states", "recurrent_states")


def _fork_layered_cache(cache: Any, n: int) -> Any:
    """`n` copies of a batch-1 cache that keeps one object per layer (transformers 5).

    Covers hybrid torsos: Qwen3.5 mixes attention layers with Gated DeltaNet layers
    whose recurrent and convolution states must be repeated too, or the suffix forward
    fails on a shape mismatch. The fork gets new layer objects, new dicts and new
    tensors: a DeltaNet layer updates its states and flags (`has_previous_state`) in
    place, so a fork that shared them would corrupt the prefix it came from. The
    approach is decider-2b's (`decider/shared_prefix.py`, Apache-2.0).

    Raises UnforkableCache for a layer holding tensors under any other name: we cannot tell
    whether such a tensor has a batch dimension, so the caller falls back to batched
    encoding rather than guess.
    """
    fork = copy.copy(cache)
    fork.layers = []
    for layer in cache.layers:
        nl = copy.copy(layer)
        for name, v in list(vars(nl).items()):
            is_state = name in _ROW_STATES
            if isinstance(v, torch.Tensor):
                if not is_state:
                    raise UnforkableCache(f"cache layer {type(layer).__name__} holds tensor {name!r}")
                if v.numel():
                    setattr(nl, name, v.expand(n, *v.shape[1:]).contiguous())
            elif isinstance(v, dict):
                if any(isinstance(t, torch.Tensor) for t in v.values()) and not is_state:
                    raise UnforkableCache(f"cache layer {type(layer).__name__} holds tensors in {name!r}")
                setattr(nl, name, {k: t.expand(n, *t.shape[1:]).contiguous()
                                   if isinstance(t, torch.Tensor) and t.numel() else t
                                   for k, t in v.items()})
        fork.layers.append(nl)
    return fork


def _expand_cache(cache: Any, n: int) -> Any:
    """Repeat a batch-1 KV cache across `n` rows.

    transformers 5 keeps a per-layer object per layer (`cache.layers`), for plain and
    hybrid torsos alike; that is the only layout this package admits. `.contiguous()` is
    not optional: the suffix forward concatenates into these tensors, and an expanded
    (stride-0) view cannot be written to.
    """
    if isinstance(getattr(cache, "layers", None), list):
        return _fork_layered_cache(cache, n)
    raise UnforkableCache(f"unsupported KV cache type: {type(cache)!r}")


class TorchEngine(SystemOneEngine):
    """The PyTorch engine: cuda, mps and cpu."""

    def __init__(self, model: StrandsDeciderModel, config: EngineConfig | None = None):
        self.cfg = config or EngineConfig()
        if str(self.cfg.device).startswith("mps"):
            # No fla/Triton on macOS; replace the reference chunk rule's slow MPS solver.
            from .mps_kernels import install

            install()
        self.model = model.to(self.cfg.device).eval()
        if str(self.cfg.device) == "cpu":
            self._upcast_torso_for_cpu()
        self.tok = model.tokenizer
        self.device = self.cfg.device

    @property
    def model_config(self) -> StrandsDeciderConfig:
        return self.model.config

    def _upcast_torso_for_cpu(self) -> None:
        """Run a half-precision torso in fp32 on CPU.

        CPU kernels for bf16 are slower than fp32, not faster: on an M3 Pro a
        256-token v19 question takes 7.7 s in bf16 and 3.5 s in fp32, with the same
        answer. The cost is memory, about 7 GiB of torso weights instead of 3.5. The
        readout is already fp32. Done under inference_mode because `load_engine`
        builds the model there, and inference tensors cannot be modified outside it.
        """
        torso = self.model.torso
        first = next(torso.parameters(), None)  # a stub torso in the tests has none
        if first is not None and first.dtype in (torch.bfloat16, torch.float16):
            with torch.inference_mode():
                torso.to(torch.float32)

    def _temperatures(self, kinds: list[str]) -> torch.Tensor:
        """Per-row temperature, falling back to the global scalar where unfitted."""
        cfg = self.model.config
        by_kind = getattr(cfg, "temperature_by_kind", None) or {}
        return torch.tensor(
            [float(by_kind.get(k, cfg.temperature)) for k in kinds],
            device=self.device,
            dtype=torch.float32,
        )

    def _option_idx(self, rendered: list[RenderedQuestion], base: int) -> torch.Tensor:
        """Option token positions for a pointer head, offset by `base`.

        `base` is 0 when the caller forwards only the question (the shared-prefix path,
        where `hidden` is the suffix) and the state's token count when it forwards the
        whole prompt. Uses the offsets `_fit` kept, so front truncation is accounted for.
        """
        rows = [
            _option_token_index(offs, rq.option_spans, 0)
            for rq, offs in zip(rendered, self._last_offsets, strict=True)
        ]
        width = max(len(r) for r in rows)
        return torch.tensor(
            [[base + i for i in r] + [-1] * (width - len(r)) for r in rows],
            dtype=torch.long,
            device=self.device,
        )

    def _pad(self, seqs: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Right-pad to a rectangle; pool_last_token finds the last real token by mask."""
        pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        width = max(len(x) for x in seqs)
        ids = torch.tensor(
            [x + [pad_id] * (width - len(x)) for x in seqs], device=self.device
        )
        mask = torch.tensor(
            [[1] * len(x) + [0] * (width - len(x)) for x in seqs], device=self.device
        )
        return ids, mask

    # ---- low level -------------------------------------------------------

    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def _slot_probs_batched(
        self, state_text: str, question_texts: list[str],
        n_slots: list[int], kinds: list[str],
        rendered: list[RenderedQuestion] | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Fallback: encode each full prompt independently.

        Previously this concatenated state and question into one string and let the
        tokeniser truncate, which cuts from the right -- removing the options and the
        `<answer>` marker the head pools at. Now it shares `_fit` with the cached path,
        so both truncate the same thing in the same direction.
        """
        s, q = self._fit(state_text, question_texts)
        ids, mask = self._pad([s + qi for qi in q])
        # This path forwards the whole prompt, so option positions sit after the state.
        assert rendered is not None or self.model.config.head_type != "pointer"
        opt_idx = (self._option_idx(rendered, len(s))  # type: ignore[arg-type]
                   if self.model.config.head_type == "pointer" else None)
        out = self.model(
            input_ids=ids,
            attention_mask=mask,
            n_slots=torch.tensor(n_slots, device=self.device),
            temperature=self._temperatures(kinds),
            opt_idx=opt_idx,
        )
        return out["log_probs"].exp(), int(mask.sum().item())

    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def _slot_probs_shared_prefix(
        self,
        state_text: str,
        question_texts: list[str],
        n_slots: list[int],
        kinds: list[str],
        rendered: list[RenderedQuestion] | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Encode the state once, then all question suffixes against that cache."""
        m = len(question_texts)
        # The suffixes continue an already-tokenised sequence, so _fit adds no BOS to
        # them -- one mid-sequence would be a token training never saw.
        s, q = self._fit(state_text, question_texts)
        prefix_ids = torch.tensor([s], device=self.device)
        prefix_len = prefix_ids.size(1)

        prefix_out = self.model.torso(
            input_ids=prefix_ids,
            attention_mask=torch.ones_like(prefix_ids),
            use_cache=True,
            return_dict=True,
        )
        cache = _expand_cache(prefix_out.past_key_values, m)

        suffix_ids, suffix_mask = self._pad(q)
        full_mask = torch.cat(
            [
                torch.ones(m, prefix_len, dtype=suffix_mask.dtype, device=self.device),
                suffix_mask,
            ],
            dim=1,
        )

        hidden = self.model.encode(
            input_ids=suffix_ids,
            attention_mask=full_mask,
            past_key_values=cache,
        )
        # pool_last_token slices the mask to the forwarded tail, so it lands on the
        # last real *suffix* token -- which is `<answer>`, the position that has
        # attended to state and question alike.
        pooled = pool_last_token(hidden, full_mask).to(torch.float32)
        if self.model.config.head_type == "pointer":
            # `hidden` is the suffix only, so option positions are suffix-relative and
            # need no prefix offset -- the cached state never enters the gather.
            from .modeling import gather_options

            assert rendered is not None
            options = gather_options(hidden, self._option_idx(rendered, 0))
            raw = self.model.head(pooled, options.to(torch.float32))
        else:
            raw = self.model.head(pooled)
        logits = apply_temperature(raw, self._temperatures(kinds))
        log_probs = masked_log_softmax(logits, torch.tensor(n_slots, device=self.device))
        n_tokens = prefix_len + int(suffix_mask.sum().item())
        return log_probs.exp(), n_tokens
