"""Serving one request: N questions against one state.

The headline property of a System One model is that asking more questions about the
same state barely costs more time. That is not automatic -- it falls out of doing
the work in the right order:

    1. Encode the state ONCE, keeping its KV cache.          <- the expensive part
    2. Broadcast that cache across the question batch.
    3. Forward only the (short) question suffixes.           <- the cheap part

For a 2000-token state and five 40-token questions, the naive approach encodes
~10,200 tokens; this encodes ~2,200. The questions genuinely evaluate in parallel,
independently -- no question can see another's text, which is what keeps the answers
decomposable in the way the API promises.

Set `use_prefix_cache=False` to fall back to plain batched encoding. The two paths
agree to within float tolerance (tests/test_prefix_cache.py pins that, for plain and hybrid torsos), so the
fallback is a safety valve rather than a different model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from .modeling import StrandsDeciderConfig, StrandsDeciderModel
from .prompting import (
    RenderedQuestion,
    read_choice,
    read_noul,
    read_score,
    render_question,
    render_state,
    score_legend,
)
from .schema import (
    Answer,
    ChoiceAnswer,
    Content,
    NoulAnswer,
    Question,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
    derive_confidence,
    derive_score_confidence,
)


@dataclass
class EngineConfig:
    device: str = "cuda"
    use_prefix_cache: bool = True
    max_batch: int = 32
    model_name: str = "strands-decider-0.1.0"
    # Largest share of the context window the question may claim before the state
    # starts being squeezed. Questions are normally short, so this rarely binds.
    max_question_fraction: float = 0.75


class UnforkableCache(TypeError):
    """A KV cache layout `_expand_cache` cannot safely repeat across rows."""


class SystemOneEngine:
    """Answers a System One request on any device: renders and truncates the prompt,
    batches the questions, and decodes the slot probabilities into answers.

    A subclass runs the model. `_slot_probs_batched` and `_slot_probs_shared_prefix`
    return each question's slot probabilities against one state, and the input token
    count. `TorchEngine` is the PyTorch one.
    """

    cfg: EngineConfig
    tok: Any

    @property
    def model_config(self) -> StrandsDeciderConfig:
        raise NotImplementedError

    def _fit(self, state_text: str, question_texts: list[str]) -> tuple[list[int], list[list[int]]]:
        """Tokenise state and questions, giving the QUESTION first claim on the window.

        The question and its options are what make a task answerable; the state is the
        part that can be sampled. Letting the state take the window first meant a long
        state consumed all of it and left the question a floor of 8 tokens -- far too
        few to hold the option list, so the model was choosing among options it could
        not see. Measured on JevBench, all 19 `long_policy` states hit the cap exactly,
        and accuracy there (0.316) sat on top of the 0.301 chance rate for those option
        counts.

        A question longer than its reserve is truncated from the FRONT, keeping the
        tail. The options and the trailing `<answer>` marker are structurally required
        -- `<answer>` is the pooling position, so losing it makes the head read an
        arbitrary token -- whereas losing some instruction text only costs meaning.
        """
        max_len = self.model_config.max_length
        enc = self.tok(question_texts, add_special_tokens=False,
                       return_offsets_mapping=True)
        q = enc["input_ids"]
        offs = enc["offset_mapping"]
        longest = max(len(x) for x in q)
        # Cap the reserve so a pathological question cannot starve the state entirely.
        reserve = min(longest, max(1, int(max_len * self.cfg.max_question_fraction)))
        # Front truncation shifts every token index, so the offsets move with the ids
        # and a pointer readout keeps pointing at the right option.
        cut = [max(0, len(x) - reserve) for x in q]
        self._last_offsets = [o[c:] for o, c in zip(offs, cut, strict=True)]
        q = [x[c:] for x, c in zip(q, cut, strict=True)]
        state_budget = max(1, max_len - reserve)
        s = self.tok(
            state_text,
            add_special_tokens=True,
            truncation=True,
            max_length=state_budget,
        )["input_ids"]
        return s, q

    def _slot_probs_batched(
        self, state_text: str, question_texts: list[str],
        n_slots: list[int], kinds: list[str],
        rendered: list[RenderedQuestion] | None = None,
    ) -> tuple[Any, int]:
        """Encode each full prompt independently."""
        raise NotImplementedError

    def _slot_probs_shared_prefix(
        self, state_text: str, question_texts: list[str],
        n_slots: list[int], kinds: list[str],
        rendered: list[RenderedQuestion] | None = None,
    ) -> tuple[Any, int]:
        """Encode the state once, then all question suffixes against that cache."""
        raise NotImplementedError

    # ---- public ----------------------------------------------------------

    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        names = list(request.questions.keys())
        questions: list[Question] = [request.questions[n] for n in names]

        rendered: list[RenderedQuestion] = []
        for q in questions:
            rq = render_question(q)
            # A pointer head scores each option from its own hidden state, so there is
            # no slot count to exceed; only the fixed-width readout has a ceiling.
            if (
                self.model_config.head_type != "pointer"
                and rq.n_slots > self.model_config.num_slots
            ):
                raise ValueError(
                    f"question has {rq.n_slots} options but this model has "
                    f"{self.model_config.num_slots} slots; split the question or "
                    f"retrain with a larger num_slots"
                )
            rendered.append(rq)

        state_text = render_state(request.state)
        n_slots = [rq.n_slots for rq in rendered]

        answers: dict[str, Answer] = {}
        total_tokens = 0

        for start in range(0, len(names), self.cfg.max_batch):
            chunk = slice(start, start + self.cfg.max_batch)
            chunk_rendered = rendered[chunk]
            chunk_slots = n_slots[chunk]
            chunk_kinds = [rq.kind for rq in chunk_rendered]

            probs = None
            # One question gains nothing from a shared prefix and pays a second forward:
            # measured on JevBench (one question per task), p50 0.111 s batched, 0.204 s
            # through the prefix path, with the same answers.
            if self.cfg.use_prefix_cache and len(chunk_rendered) > 1:
                try:
                    probs, ntok = self._slot_probs_shared_prefix(
                        state_text, [rq.text for rq in chunk_rendered], chunk_slots,
                        chunk_kinds, rendered=chunk_rendered,
                    )
                    total_tokens += ntok
                except UnforkableCache as e:
                    print(f"[strands-decider] shared-prefix cache disabled ({e}); using batched encoding")
                    self.cfg = replace(self.cfg, use_prefix_cache=False)
            if probs is None:
                probs, ntok = self._slot_probs_batched(
                    state_text, [rq.text for rq in chunk_rendered], chunk_slots,
                    chunk_kinds, rendered=chunk_rendered,
                )
                total_tokens += ntok

            for i, name in enumerate(names[chunk]):
                rq = chunk_rendered[i]
                row = probs[i, : rq.n_slots].tolist()
                answers[name] = _to_answer(
                    rq, row, ordinal_smoothing=self.model_config.ordinal_smoothing
                )

        return SystemOneResponse(
            model=self.cfg.model_name,
            answers=answers,
            # One slot decision per question: the output side really is this cheap.
            usage=Usage(input_tokens=total_tokens, output_tokens=len(names)),
        )

    def ask(self, state: Content, questions: dict[str, Question]) -> SystemOneResponse:
        return self.evaluate(SystemOneRequest(state=state, questions=questions))


def _option_token_index(
    offsets: Sequence[Sequence[int]],
    spans: Sequence[Sequence[int]],
    base: int,
) -> list[int]:
    """Last token index of each option's line, for a pointer readout.

    The last token of the span is the one that has just read the whole option under
    causal attention. Raises when an option has no surviving token: truncation has
    removed it, and scoring it from a neighbour's representation would be silently
    wrong.
    """
    out: list[int] = []
    for s, e in spans:
        a, b = base + s, base + e
        last = -1
        for j, (lo, hi) in enumerate(offsets):
            if hi <= lo:
                continue
            if lo >= a and hi <= b:
                last = j
        if last < 0:
            raise ValueError(
                f"option span ({a},{b}) has no tokens left; the prompt was "
                "truncated through its option list"
            )
        out.append(last)
    return out


def _to_answer(
    rq: RenderedQuestion, probs: list[float], *, ordinal_smoothing: float = 0.0
) -> Answer:
    if rq.kind == "noul":
        return NoulAnswer(noul=round(read_noul(probs, rq), 4))

    if rq.kind == "choice":
        by_name = read_choice(probs, rq)
        best = max(by_name, key=lambda k: by_name[k])
        return ChoiceAnswer(
            choice=best,
            probabilities={k: round(v, 4) for k, v in by_name.items()},
            confidence=round(derive_confidence(list(by_name.values())), 4),
        )

    score, ordered = read_score(probs, rq)
    return ScoreAnswer(
        score=round(score, 4),
        # Report the rubric in canonical ascending order, not the order rendered.
        legend=score_legend(rq),
        probabilities={k: round(v, 4) for k, v in ordered.items()},
        # Ordinal confidence, not max-probability: see derive_score_confidence.
        confidence=round(
            derive_score_confidence(
                list(ordered.values()), ordinal_smoothing=ordinal_smoothing
            ),
            4,
        ),
    )


@torch.inference_mode()  # type: ignore[untyped-decorator]
def load_engine(
    checkpoint: str,
    *,
    device: str = "cuda",
    use_prefix_cache: bool = True,
    attn_implementation: str | None = None,
) -> SystemOneEngine:
    if device == "mlx":
        from .mlx_engine import load_mlx_engine

        return load_mlx_engine(checkpoint, use_prefix_cache=use_prefix_cache)
    from .torch_engine import TorchEngine  # here, not at the top: it subclasses SystemOneEngine

    model = StrandsDeciderModel.load(checkpoint, attn_implementation=attn_implementation)
    return TorchEngine(
        model, EngineConfig(device=device, use_prefix_cache=use_prefix_cache)
    )
