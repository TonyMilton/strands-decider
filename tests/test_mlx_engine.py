"""The MLX backend answers as the PyTorch engine does.

A tiny random Qwen3.5 base stands in for the real one, saved in fp32 so that loading it
in the training dtype (bf16) changes its values: a conversion that skipped that step
would not match the PyTorch engine, which loads the base in bf16 too.
"""
from __future__ import annotations

import pytest
import torch

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("MLX needs Apple silicon", allow_module_level=True)
transformers = pytest.importorskip("transformers")

from strands_decider import cli, mlx_engine  # noqa: E402
from strands_decider.infer import load_engine  # noqa: E402
from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel  # noqa: E402
from strands_decider.prompting import render_question, render_state  # noqa: E402
from strands_decider.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion  # noqa: E402

QUESTIONS = {
    "refund": NoulQuestion(instructions="Does the customer want a refund?"),
    "team": ChoiceQuestion(instructions="Which team?",
                           criteria={"billing": "money", "shipping": "parcels", "other": ""}),
    "urgency": ScoreQuestion(instructions="How urgent is it?",
                             criteria=["not", "a little", "quite", "very"]),
}
STATE = "Order 4411 was charged twice and the parcel is late. Please fix this today!"


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {"<pad>": 0, "<eos>": 1, **{c: i + 2 for i, c in enumerate(alphabet)}}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>")


@pytest.fixture
def ckpt(tmp_path):
    torch.manual_seed(0)
    base = tmp_path / "base"
    cfg = transformers.Qwen3_5TextConfig(
        vocab_size=260, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
        linear_value_head_dim=16, full_attention_interval=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"])
    transformers.Qwen3_5ForCausalLM(cfg).save_pretrained(base)
    _tokenizer().save_pretrained(base)

    config = StrandsDeciderConfig(base_model=str(base), head_type="pointer", pointer_dim=16,
                                  lora_r=2, torch_dtype="bfloat16", max_length=512)
    model = StrandsDeciderModel.from_pretrained_base(config)
    with torch.no_grad():  # lora_B starts at zero: make the adapter change the torso
        for n, p in model.torso.named_parameters():
            if "lora_B" in n:
                p.normal_(std=0.05)
    path = tmp_path / "ckpt"
    model.save_pretrained(str(path))
    return path


def _probs(engine, questions):
    rendered = [render_question(q) for q in questions.values()]
    args = (render_state(STATE), [r.text for r in rendered], [r.n_slots for r in rendered],
            [r.kind for r in rendered])
    shared, _ = engine._slot_probs_shared_prefix(*args, rendered=rendered)
    batched, _ = engine._slot_probs_batched(*args, rendered=rendered)
    return [torch.tensor(p.tolist()) for p in (shared, batched)]


def test_mlx_matches_the_pytorch_engine(ckpt):
    ref = load_engine(str(ckpt), device="cpu")  # fp32 torso, adapter unmerged
    got = load_engine(str(ckpt), device="mlx")
    assert not got.torso.training  # else Gated DeltaNet skips its Metal kernel
    for want, have in zip(_probs(ref, QUESTIONS), _probs(got, QUESTIONS), strict=True):
        torch.testing.assert_close(have, want, atol=1e-5, rtol=0)
    a, b = ref.ask(STATE, QUESTIONS), got.ask(STATE, QUESTIONS)
    assert a.usage == b.usage
    assert {k: v.type for k, v in a.answers.items()} == {k: v.type for k, v in b.answers.items()}


@pytest.mark.parametrize(("cuda", "mlx", "mps", "want"), [
    (True, True, True, "cuda"),
    (False, True, True, "mlx"),
    (False, False, True, "mps"),
    (False, False, False, "cpu"),
])
def test_auto_device_order(monkeypatch, cuda, mlx, mps, want):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    monkeypatch.setattr(mlx_engine, "available", lambda: mlx)
    assert cli._auto_device() == want


def test_concurrent_requests_get_their_own_answers(ckpt):
    from concurrent.futures import ThreadPoolExecutor

    engine = load_engine(str(ckpt), device="mlx")
    # Different lengths and option counts, so a request read with another's offsets shows.
    cases = [(STATE * (1 + i % 5), dict(list(QUESTIONS.items())[: 1 + i % 3])) for i in range(12)]
    want = [engine.ask(s, q) for s, q in cases]
    with ThreadPoolExecutor(6) as pool:  # as the server's handlers run
        got = list(pool.map(lambda c: engine.ask(*c), cases * 3))
    assert got == want * 3
