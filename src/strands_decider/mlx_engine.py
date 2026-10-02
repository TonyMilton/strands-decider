"""Inference on Apple silicon with MLX.

The checkpoint's adapter is merged into its base at load, in fp32: base + B @ A * scale is
not representable in bf16, so a narrower dtype would round the adapter's update.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen3_5
from mlx_lm.models.cache import ArraysCache, KVCache
from transformers import AutoTokenizer

from .infer import EngineConfig, SystemOneEngine, _option_token_index
from .modeling import MASK_VALUE, StrandsDeciderConfig, checkpoint_dir, config_path, load_head_state
from .prompting import RenderedQuestion
from .schema import SystemOneRequest, SystemOneResponse


def available() -> bool:
    return bool(mx.metal.is_available())


def load_language_model(config: StrandsDeciderConfig, checkpoint: Path) -> Any:
    """The text model with the adapter merged, as StrandsDeciderModel.load builds it."""
    base = Path(checkpoint_dir(config.base_model))
    base_cfg = json.loads((base / "config.json").read_text())
    if base_cfg["model_type"] not in {"qwen3_5", "qwen3_5_text"}:
        raise ValueError(f"MLX supports Qwen3.5 torsos only, not {base_cfg['model_type']!r}")
    dtype = getattr(mx, config.torch_dtype)
    # Rounded to the training dtype first, as StrandsDeciderModel loads the base: it ships
    # some tensors (A_log, the linear-attention norms) in fp32, and training saw them
    # rounded. sanitize drops the vision tower, which lazy arrays never compute.
    weights = {k: v.astype(dtype).astype(mx.float32)
               for f in sorted(base.glob("*.safetensors")) for k, v in _tensors(f).items()}
    if config.use_lora:
        multimodal = any(k.startswith("model.language_model.") for k in weights)
        _merge_lora(weights, checkpoint / "lora", "model.language_model." if multimodal else "model.")
    model = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(base_cfg))
    model.load_weights(list(model.sanitize(weights).items()))
    model.eval()  # in training mode, Gated DeltaNet runs without its Metal kernel
    return model.language_model


def _merge_lora(weights: dict[str, mx.array], lora: Path, prefix: str) -> None:
    cfg = json.loads((lora / "adapter_config.json").read_text())
    if cfg.get("use_dora") or cfg.get("use_rslora") or cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise ValueError(f"{lora}: MLX merges plain LoRA only")
    scale = cfg["lora_alpha"] / cfg["r"]
    adapter = _tensors(lora / "adapter_model.safetensors")
    for k, a in adapter.items():
        if k.endswith(".lora_B.weight"):
            continue
        if not k.endswith(".lora_A.weight"):
            raise ValueError(f"{lora}: unexpected adapter tensor {k}")
        b = adapter[k.replace(".lora_A.", ".lora_B.")]
        name = prefix + k.removeprefix("base_model.model.").removesuffix(".lora_A.weight") + ".weight"
        # In fp32 and in peft's order (B @ A, then the scale), as merge_and_unload does.
        weights[name] = weights[name] + (b.astype(mx.float32) @ a.astype(mx.float32)) * scale


def _tensors(path: Path) -> dict[str, mx.array]:
    return cast(dict[str, mx.array], mx.load(str(path)))  # a .safetensors file is a dict


class PointerHead(nn.Module):
    def __init__(self, hidden_size: int, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)  # eps 1e-5, as torch's default
        self.q = nn.Linear(hidden_size, dim)
        self.k = nn.Linear(hidden_size, dim)
        self.scale = dim ** -0.5

    def __call__(self, decide: mx.array, options: mx.array) -> mx.array:
        d = self.q(self.norm(decide))[:, :, None]
        o = self.k(self.norm(options))
        logits: mx.array = (o @ d).squeeze(-1) * self.scale
        return logits


class MlxEngine(SystemOneEngine):
    def __init__(self, checkpoint: Path, config: EngineConfig):
        self.cfg = config
        model_config = StrandsDeciderConfig.from_json(config_path(str(checkpoint)))
        if model_config.head_type != "pointer":
            raise ValueError(f"MLX supports the pointer head only, not {model_config.head_type!r}")
        self._model_config = model_config
        self.tok = AutoTokenizer.from_pretrained(str(checkpoint))
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        lm = load_language_model(model_config, checkpoint)
        # The decoder with its final norm and no LM head, as transformers' last_hidden_state.
        self.torso = lm.model
        self._make_cache = lm.make_cache
        self.head = PointerHead(self.torso.embed_tokens.weight.shape[1], model_config.pointer_dim)
        self.head.load_weights([(k, mx.array(v.float().numpy())) for k, v in
                                load_head_state(str(checkpoint)).items()])
        mx.eval(self.torso.parameters(), self.head.parameters())
        # MLX keeps freed buffers for reuse, by default up to its memory limit: on v19 that
        # held 10 GiB beyond the weights between requests. A 1 GiB cap measured no slower.
        mx.set_cache_limit(1 << 30)
        self._lock = threading.Lock()

    @property
    def model_config(self) -> StrandsDeciderConfig:
        return self._model_config

    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        # The server answers on a thread pool, and `_fit` keeps a request's offsets on the
        # engine: run concurrently, one request is read with another's.
        with self._lock:
            return super().evaluate(request)

    def _temperatures(self, kinds: list[str]) -> mx.array:
        cfg = self._model_config
        by_kind = cfg.temperature_by_kind or {}
        return mx.array([float(by_kind.get(k, cfg.temperature)) for k in kinds], dtype=mx.float32)

    def _slot_probs_batched(
        self, state_text: str, question_texts: list[str],
        n_slots: list[int], kinds: list[str],
        rendered: list[RenderedQuestion] | None = None,
    ) -> tuple[mx.array, int]:
        s, q = self._fit(state_text, question_texts)
        hidden, lengths = self._forward([s + qi for qi in q])
        assert rendered is not None
        return self._readout(hidden, lengths, rendered, len(s), n_slots, kinds), sum(lengths)

    def _slot_probs_shared_prefix(
        self, state_text: str, question_texts: list[str],
        n_slots: list[int], kinds: list[str],
        rendered: list[RenderedQuestion] | None = None,
    ) -> tuple[mx.array, int]:
        s, q = self._fit(state_text, question_texts)
        cache = self._make_cache()
        self.torso(mx.array([s]), cache=cache)
        hidden, lengths = self._forward(q, cache=[_fork(c, len(q)) for c in cache])
        assert rendered is not None
        return self._readout(hidden, lengths, rendered, 0, n_slots, kinds), len(s) + sum(lengths)

    def _forward(self, seqs: list[list[int]], cache: list[Any] | None = None) -> tuple[mx.array, list[int]]:
        lengths = [len(x) for x in seqs]
        width = max(lengths)
        # Right padding needs no mask: attention is causal and the recurrent layers run
        # forward, so pads only follow the positions read below.
        ids = mx.array([x + [self.tok.pad_token_id] * (width - len(x)) for x in seqs])
        return self.torso(ids, cache=cache).astype(mx.float32), lengths

    def _readout(
        self, hidden: mx.array, lengths: list[int], rendered: list[RenderedQuestion],
        base: int, n_slots: list[int], kinds: list[str],
    ) -> mx.array:
        rows = [[base + i for i in _option_token_index(offs, rq.option_spans, 0)]
                for rq, offs in zip(rendered, self._last_offsets, strict=True)]
        k = max(len(r) for r in rows)
        # Padded option slots read position 0 and are masked out below, as in torch.
        opt_idx = mx.array([r + [0] * (k - len(r)) for r in rows])
        batch = mx.arange(len(rows))
        options = hidden[batch[:, None], opt_idx]
        pooled = hidden[batch, mx.array(lengths) - 1]

        logits = self.head(pooled, options) / mx.maximum(self._temperatures(kinds), 1e-6)[:, None]
        valid = mx.arange(k)[None, :] < mx.array(n_slots)[:, None]
        probs = mx.where(valid, mx.softmax(mx.where(valid, logits, MASK_VALUE), axis=-1), 0.0)
        mx.eval(probs)
        return probs


def _fork(cache: Any, n: int) -> Any:
    if isinstance(cache, KVCache):
        keys, values, offset = cache.state
        assert keys is not None and values is not None
        fork = KVCache()
        fork.state = (mx.repeat(keys[..., :offset, :], n, axis=0),
                      mx.repeat(values[..., :offset, :], n, axis=0), offset)
        return fork
    if isinstance(cache, ArraysCache):
        assert all(c is not None for c in cache.cache)  # the prefix forward fills every state
        arrays = ArraysCache(len(cache.cache))
        arrays.state = ([mx.repeat(cast(mx.array, c), n, axis=0) for c in cache.cache], None, None)
        return arrays
    raise TypeError(f"unsupported MLX cache type: {type(cache)!r}")


def load_mlx_engine(checkpoint: str, *, use_prefix_cache: bool = True) -> MlxEngine:
    return MlxEngine(Path(checkpoint_dir(checkpoint)),
                     EngineConfig(device="mlx", use_prefix_cache=use_prefix_cache))
