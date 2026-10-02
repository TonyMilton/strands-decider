"""The shared-prefix path must agree with encoding each full prompt: for plain attention
torsos, and for hybrid ones (Qwen3.5: Gated DeltaNet + attention), whose recurrent and
convolution states `_expand_cache` has to repeat along with the keys and values.

Tiny random-weight Qwen3.5 text models: on CUDA when there is one (with
flash-linear-attention installed that is the production kernel path, and its Triton
kernels require CUDA tensors), otherwise on CPU with transformers' reference kernels.
"""
from __future__ import annotations

import copy

import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3_5TextConfig"):
    pytest.skip("this transformers has no Qwen3.5", allow_module_level=True)

from strands_decider.infer import UnforkableCache  # noqa: E402
from strands_decider.torch_engine import _expand_cache  # noqa: E402

HYBRID = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
PLAIN = ["full_attention"] * 2
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _torso(layer_types):
    torch.manual_seed(0)
    cfg = transformers.Qwen3_5TextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=len(layer_types),
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
        linear_value_head_dim=16, layer_types=layer_types)
    return transformers.Qwen3_5ForCausalLM(cfg).model.eval().to(DEV)


def _prefix_cache(torso, prefix):
    ids = torch.tensor([prefix], device=DEV)
    return torso(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True,
                 return_dict=True).past_key_values


@pytest.mark.parametrize("layer_types", [HYBRID, PLAIN], ids=["hybrid", "plain"])
@torch.inference_mode()
def test_shared_prefix_matches_full_prompts(layer_types):
    torso = _torso(layer_types)
    g = torch.Generator().manual_seed(1)
    prefix = torch.randint(1, 128, (11,), generator=g).tolist()
    suffixes = [torch.randint(1, 128, (n,), generator=g).tolist() for n in (3, 5, 2)]

    cache = _expand_cache(_prefix_cache(torso, prefix), len(suffixes))
    width = max(map(len, suffixes))
    ids = torch.tensor([s + [0] * (width - len(s)) for s in suffixes], device=DEV)  # right-padded
    mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in suffixes], device=DEV)
    full_mask = torch.cat([torch.ones(len(suffixes), len(prefix), dtype=mask.dtype, device=DEV), mask], 1)
    shared = torso(input_ids=ids, attention_mask=full_mask, past_key_values=cache,
                   return_dict=True).last_hidden_state

    for i, s in enumerate(suffixes):
        whole = torch.tensor([prefix + s], device=DEV)
        ref = torso(input_ids=whole, attention_mask=torch.ones_like(whole),
                    return_dict=True).last_hidden_state[0, len(prefix):]
        # every real suffix position, including the last one (the pooling position)
        torch.testing.assert_close(shared[i, : len(s)], ref, atol=1e-4, rtol=1e-4)


@torch.inference_mode()
def test_fork_leaves_the_prefix_cache_untouched():
    torso = _torso(HYBRID)
    prefix = list(range(1, 12))
    cache = _prefix_cache(torso, prefix)
    before = copy.deepcopy([vars(layer) for layer in cache.layers])
    fork = _expand_cache(cache, 3)
    ids = torch.tensor([[5, 6, 7]] * 3, device=DEV)
    torso(input_ids=ids, attention_mask=torch.ones(3, len(prefix) + 3, dtype=torch.long, device=DEV),
          past_key_values=fork, return_dict=True)
    for old, layer in zip(before, cache.layers, strict=True):
        for name, v in old.items():
            now = vars(layer)[name]
            if isinstance(v, torch.Tensor):
                assert torch.equal(v, now), name
            elif isinstance(v, dict):
                assert v.keys() == now.keys(), name
                for k in v:
                    assert (torch.equal(v[k], now[k]) if isinstance(v[k], torch.Tensor)
                            else v[k] == now[k]), (name, k)


def test_unknown_cache_state_is_refused():
    torso = _torso(HYBRID)
    cache = _prefix_cache(torso, list(range(1, 8)))
    cache.layers[0].mystery = torch.zeros(1, 4, device=DEV)  # batch dim or not? we cannot tell
    with pytest.raises(UnforkableCache):
        _expand_cache(cache, 2)
