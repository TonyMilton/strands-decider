"""Hybrid (recurrent + attention) torsos use the shared-prefix cache like any other.

Until the cache fork repeated a Gated DeltaNet layer's recurrent and convolution states
as well as keys and values, the engine had to turn the prefix cache off for them (the
suffix forward failed on a shape mismatch). tests/test_prefix_cache.py pins the fork
against full-prompt encoding on a real hybrid cache.
"""
from types import SimpleNamespace

import torch.nn as nn

from strands_decider.infer import EngineConfig
from strands_decider.modeling import StrandsDeciderModel
from strands_decider.torch_engine import TorchEngine


class _Stub(nn.Module):
    def __init__(self, layer_types):
        super().__init__()
        self.torso = nn.Module()
        self.torso.config = SimpleNamespace(layer_types=layer_types)
        self.tokenizer = None
        self.config = SimpleNamespace(temperature=1.0)


def test_is_hybrid_reads_layer_types():
    assert StrandsDeciderModel.is_hybrid(_Stub(["linear_attention", "full_attention"]).torso)
    assert not StrandsDeciderModel.is_hybrid(_Stub(["full_attention"] * 4).torso)
    assert not StrandsDeciderModel.is_hybrid(_Stub(None).torso)  # Qwen3 configs have no layer_types


def test_engine_keeps_prefix_cache_for_hybrid_torsos():
    hybrid = TorchEngine(_Stub(["linear_attention", "full_attention"]),
                             EngineConfig(device="cpu", use_prefix_cache=True))
    assert hybrid.cfg.use_prefix_cache is True
    plain = TorchEngine(_Stub(["full_attention"]), EngineConfig(device="cpu", use_prefix_cache=True))
    assert plain.cfg.use_prefix_cache is True
