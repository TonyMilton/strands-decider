# Inference

This document describes how to run a checkpoint: install, get a checkpoint, ask it questions
from the command line, and serve it over HTTP. Strands decider answers typed questions about a
state, the text to classify; it does not generate text. A question is a `noul` (yes or no,
returned as P(true)), a `choice` (one of N options) or a `score` (a level on an ordered
scale). Install the package, point the commands at the published Hub id or at a checkpoint you
trained, then `strands-decider ask` or `strands-decider serve`. Pass `--device cuda`, `mlx`, `mps` or `cpu`
explicitly, or omit it for the first available of `cuda`, `mlx` (with the
[`[mlx]` extra](#mlx-on-apple-silicon) installed), `mps` and `cpu`. A
checkpoint from `training/recipe.sh all` is already calibrated; calibrate any other
checkpoint with `strands-decider calibrate` before you serve it.

- [Environments for serving](#environments-for-serving): Linux with CUDA, a Mac, or CPU.
- [Model artifact](#model-artifact): what a checkpoint holds, and the Hugging Face export.
- [Ask](#ask): one command, with recorded v19 output and how to read its confidence.
- [Serve](#serve): `POST /v1/systemone` and `/health`, on `127.0.0.1` with no authentication.
- [Asking many questions is nearly free](#asking-many-questions-is-nearly-free): the shared-prefix cache.
- [Serving on a Mac](#serving-on-a-mac): MPS, and the one kernel that had to be replaced.
- [MLX on Apple silicon](#mlx-on-apple-silicon): faster than MPS, and in fp32, closer to
  the fp32 reference.
- [`../examples/strands/`](../examples/strands/README.md): an agent built with the Strands
  Agents SDK that uses the server, with its client in `_client.py`.
- [`../evaluation/results.md`](../evaluation/results.md): measured latency and accuracy on an
  RTX 3090 and on a Mac; [`../evaluation/jevbench.md`](../evaluation/jevbench.md): JevBench,
  the external benchmark.
- [`../docs/architecture.md`](../docs/architecture.md#the-routing-convention): the confidence
  formulas and the routing convention.
- [`../training/README.md`](../training/README.md): how to train a checkpoint.

Paths are relative to the repository root unless they are links. A module path such as
`mps_kernels.py` is relative to `src/strands_decider/`.

## Environments for serving

On Linux, or under WSL2 on Windows, with an NVIDIA GPU, serving uses the training
environment in [Setup](../training/README.md#setup).

A server started inside WSL (`strands-decider serve ... --port 8099`) is reachable from Windows
at `127.0.0.1:8099` for as long as its WSL session is alive.

**Serving on macOS (Apple silicon), inference only.** No `flash-linear-attention`:
Triton has no macOS build. Nothing else changes; pass `--device mps`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 transformers==5.17.0 peft==0.21.0
pip install -e ".[dev]"
strands-decider serve checkpoints/hobson-2b-recipe --device mps --port 8099
```

Transformers' `causal_conv1d_fn` falls back to its reference PyTorch path. On CPU,
`chunk_gated_delta_rule` also falls back. On MPS, `mps_kernels.py` replaces that second
function, so only the first fallback matters there, and it costs 32 ms a forward.

**CPU only.** The same commands as on macOS, with `--device cpu`. On Linux without a GPU,
`pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu` skips the CUDA
wheels. Both reference fallback paths above apply, and inference is much slower than on a
GPU.

## Model artifact

The v19 reference weights are published as
[`StrandsAgents/strands-decider-2B-hobson-v19`](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19).
Every command that takes a checkpoint path also takes that id, and the loader downloads the
repository to the Hub cache. You can equally train your own with the recipe
([training/README.md](../training/README.md#usage)); `recipe.sh` writes it to
`checkpoints/hobson-2b-recipe`.

A checkpoint directory holds these files. `StrandsDeciderModel.save_pretrained` in `modeling.py`
writes them, `strands-decider calibrate` adds the temperatures, and `StrandsDeciderModel.load` reads them.

| File | Contents |
| --- | --- |
| `strands_decider_config.json` | The model configuration: the base model name, the readout type and size, the window (`max_length`) and the fitted temperatures. A checkpoint saved before the rename, such as the published v19, holds `hobson_config.json` instead; `load` reads either |
| `lora/` | The LoRA adapter (`adapter_config.json`, `adapter_model.safetensors`) |
| `slot_head.pt` or `head.safetensors` | The readout weights. `save_pretrained` writes `slot_head.pt`. An export from `strands_decider.hf_export` holds `head.safetensors` instead. |
| `tokenizer.json`, `tokenizer_config.json` | The tokenizer |
| `chat_template.jinja` | The chat template, saved with the tokenizer. Inference does not read it. |

Training also writes `train_config.json` and `history.json`. Inference does not read them.

The base weights are not in the checkpoint. At first use, `load` downloads
`Qwen/Qwen3.5-2B-Base` (about 4.5 GB) from Hugging Face. The loader passes no revision, so
it gets the current `main` of that repository. A later change to that repository changes
what a fresh install loads.

`python -m strands_decider.hf_export export CKPT OUT --run-id RUN` writes a checkpoint as a Hugging
Face-format folder. Without `--run-id`, the exporter takes the run id from a
`results/<run-id>/` part of the CKPT path, and it stops if the path has none. The folder
holds the readout as `head.safetensors`, so it has no pickle files. It also holds the PEFT
adapter, the tokenizer and configuration files, and a model card (`README.md`).
`provenance.json`, `LICENSE.md` with the licence status and `MANIFEST.sha256` complete it.
`python -m strands_decider.hf_export verify DIR` checks a folder. `StrandsDeciderModel.load` reads the
folder as it reads a checkpoint, and the base weights still download separately from
Hugging Face. Where a command takes a checkpoint path (`strands-decider serve`, `strands-decider ask`,
`strands-decider info`), a Hub model repo id such as `StrandsAgents/strands-decider-2B-hobson-v19` also works: the loader
downloads the repo to the Hub cache. `strands-decider calibrate` writes into the checkpoint, so it
needs a local directory. `provenance.json` records the base-model revision, but the loader does not
read it. OUT can be a local directory or an `s3://` URI. The exporter does not upload to
the Hugging Face Hub.

## Ask

There are three question types, and all three are one mechanism read back three ways.

- A **noul** question is a yes/no question. It has two options and returns the
  probability of "true".
- A **choice** question has any number of options. It returns the top option, a
  probability for each option, and a confidence: the top probability, rescaled so that an
  even split gives 0 and a certain answer gives 1, whatever the number of options.
- A **score** question has ordered levels. It returns an expected value on that scale,
  the distribution, and a confidence.

On the command line, a choice or a score is written as `question=option,option,...`. The
confidence is computed from the distribution, not predicted.

The output below was recorded from a v19 checkpoint, which the v19 reference recipe
trains. The command below names `checkpoints/hobson-2b-recipe`, the path
`training/recipe.sh` (or `bash training/recipe.sh`) writes; pass
`StrandsAgents/strands-decider-2B-hobson-v19` instead to run the published weights. On a Mac,
add `--device mps`. A retrained checkpoint gives somewhat different numbers with the same
answers: a later v19 retrain gives 0.809, `technical` at 0.783 and a score confidence
of 0.472 on CPU.

```bash
strands-decider ask checkpoints/hobson-2b-recipe \
  --state "Help! My payouts have been failing for 3 days." \
  --noul "Does this convey urgency?" \
  --choice "Which team should handle this?=billing,technical,sales" \
  --score "How frustrated is the writer?=calm,frustrated,very angry"
```

```
noul_0 noul = 0.801
choice_0 -> technical (confidence 0.622)
    technical                0.748
    billing                  0.234
    sales                    0.018
score_0 score = 1.24 (confidence 0.578)
    0: calm                                     0.093
    1: frustrated                               0.578
    2: very angry                               0.330
```

That is real output from v19, not an illustration — and it is
worth reading the confidence, not just the answer. Whether failing payouts are
`billing` or `technical` is arguable, and the model leans but says so: 0.622 is in the
"confirm" band of the [routing convention](../docs/architecture.md#the-routing-convention), well short
of the 0.9 at which that convention acts without a check.

## Serve

```bash
strands-decider serve checkpoints/hobson-2b-recipe --port 8000
```

`POST /v1/systemone` takes a state and typed questions. JevBench's `typesafe` adapter
targets this endpoint and runs against it unmodified: 231 of 231 tasks attempted and
schema validity 1.000 ([External benchmark](../evaluation/jevbench.md#external-benchmark-jevbench-v1-public-set)).
Compatibility with the Jev API itself is not verified. `GET /health` returns the model
name, the checkpoint path, the base model, the device and the calibration temperature.

The server binds to `127.0.0.1` by default and has no authentication. Its behaviour under
concurrent requests is not verified. Use it for local experiments only
([Security](../README.md#security)).

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Help! My payouts have been failing for 3 days.",
  "model": "hobson-latest",
  "questions": {
    "is_urgent":  {"type": "noul",   "instructions": "Does this convey urgency?"},
    "department": {"type": "choice", "instructions": "Route it.",
                   "criteria": {"billing": "money", "technical": "bugs", "sales": "pricing"}},
    "frustration":{"type": "score",  "instructions": "How frustrated?",
                   "criteria": ["Calm", "Frustrated", "Very angry"]}
  }
}'
```

A hybrid torso such as v14's shares the state across questions like any other; a single
question is encoded in one pass ([Asking many questions is nearly free](#asking-many-questions-is-nearly-free)).

## Asking many questions is nearly free

*The latency table below was measured on v7, a Qwen3 torso. The cache now also forks the
recurrent state of the hybrid Qwen3.5 torso
([Hybrid torsos share the prefix too](../docs/architecture.md#design-decisions-worth-knowing)).
The v14 figures at the end of this section use it.*

The state is encoded **once** and its KV cache is broadcast across the question batch;
only the short question suffixes are forwarded per question.

```
naive:  N x (state + question)     tokens
strands-decider: state + N x question       tokens
```

Measured against the served 1.7B checkpoint, one ~640-token state:

| questions | latency | input tokens |
| --- | --- | --- |
| 1 | 219.4 ms | 666 |
| 6 | 217.4 ms | 966 |

Five extra questions cost roughly nothing. The naive path would have encoded ~4,000
tokens instead of 966. The questions genuinely evaluate independently — no question
can see another's text — which is what keeps the answers decomposable.

`--no-prefix-cache` falls back to plain batched encoding. The two paths are
*numerically identical in fp32* (pinned by `test_prefix_cache_is_exact_in_fp32`, and for
hybrid torsos by `tests/test_prefix_cache.py`); in bf16 they differ by ~2e-3 from
accumulation order alone, far below any threshold worth routing on. On v14 on the RTX 3090, eight
questions over a ~2,000-token state take 369 ms through the shared prefix against
1,964 ms batched, and sixteen take 445 ms against 4,086 ms. A request with one question
skips the prefix path: it would pay a second forward for nothing.

## Serving on a Mac

Measured with v19 on an M3 Pro (36 GB) running macOS 26.6: torch 2.7.1, transformers
5.17.0, bf16 on MPS.
**The answers match the RTX 3090 run, but the latency does not.** Under 300 tokens, a v19
question takes a median 153 ms warm and 310 ms on the first request of its length. Across
JevBench the warm median is 234 ms and the 95th percentile is 2,628 ms. These figures are
for v19 on this Mac only. They are not a latency claim for other hardware or other
checkpoints.

The accuracy and latency measurements are in
[Serving on a Mac: accuracy and latency](../evaluation/results.md#serving-on-a-mac-accuracy-and-latency).

**One kernel had to be replaced.** Without `fla`, the reference chunk rule for the 18
Gated DeltaNet layers was 806 of 1,739 ms of a 1,024-token forward. Nearly all of it
was `torch.linalg.solve_triangular`, about 20 ms a call on MPS. `mps_kernels.py` inverts
the same unit-lower-triangular system by recursive block inversion, which is six levels
of batched matmuls, and applies the inverse to both right-hand sides. The engine
installs it for MPS only, and the forward is 1.7x faster (1,024 tokens: 1,750 to
1,029 ms). The obvious inverse, the finite Neumann product (I-N)(I+N²)(I+N⁴)..., passed
random-input tests and was wrong by up to 5e8 on real activations. Correlated,
l2-normalised keys make the powers of N huge before they cancel.
`tests/test_mps_kernels.py` builds its inputs that way and fails 14 cases against that
version.

## MLX on Apple silicon

`--device mlx` runs the torso with [MLX](https://github.com/ml-explore/mlx) and mlx-lm's
Qwen3.5, whose Gated DeltaNet layers have a fused Metal kernel, instead of PyTorch.
Rendering, truncation, the shared-prefix cache and the answer decoding are the PyTorch
engine's own; only the forward pass differs. Qwen3.5 checkpoints with a pointer head only,
which includes v19.

```bash
pip install -e ".[mlx]"
strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19   # picks mlx
```

**The adapter is merged into the base at load, in fp32.** The merged weight
`base + B·A·scale` is not representable in bf16, so a bf16 merge would round the adapter's
update. The base is rounded to the checkpoint's training dtype first, as
`StrandsDeciderModel.load` loads it: the Qwen3.5 base ships `A_log` and the linear-attention
norm weights in fp32, and training saw them rounded to bf16. It reads the same base files
from the Hub cache as PyTorch does, and the merge takes about a second. The fp32 weights
take 7 GiB of memory.

**Parity.** Measured with v19 on an M3 Max (36 GB): mlx 0.32.3, mlx-lm 0.32.0, against the
PyTorch engine on CPU, which runs the torso in fp32 with the adapter unmerged. Over 157
answers (noul, choice up to 100 options, score up to 10 levels, dict and list states,
non-Latin and zero-width text, truncated states and questions, and up to 40 questions in
one request), the largest probability difference was 4.6e-6 and no answer changed; no
option's probability crossed any multiple of 0.05. Hidden states agree to a relative
2.6e-5 or better at every layer, up to 4,096 tokens. bf16 on MPS differs from the same
reference by up to 1.6e-2, and crossed such a threshold 10 times. The warm median per
request was 112 ms on MLX and 225 ms on MPS.

mlx-lm before 0.32 puts the epsilon of Gated DeltaNet's q/k normalisation where
transformers does not, which the `[mlx]` extra rules out. It is capped below 0.33 because
the engine reads mlx-lm's model layout and cache state directly.

`tests/test_mlx_engine.py` holds a tiny random Qwen3.5 checkpoint to 1e-5 of the PyTorch
engine; it fails if the base is merged without the bf16 rounding.
