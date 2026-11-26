# Loop SLM Research

A local-first research project investigating whether recurrent computation and shared transformer weights can improve the reasoning efficiency of small language models.

## Primary research question

How much capability can a small pretrained language model gain from recurrent/shared-weight computation, where do additional iterations saturate, and do any gains survive comparisons at controlled training and inference compute?

## Initial research direction

- Base checkpoint: HuggingFaceTB/SmolLM2-135M, pinned to revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2 (Apache-2.0).
- Initial recurrent unit: the complete 30-layer decoder stack, repeatedly applied with shared parameters.
- Fixed depths: 1, 2, 3, 4, 6, and 8 passes.
- Embeddings, final normalization, and language-model head execute once.
- Adaptive halting and teacher distillation are deferred until fixed-depth comparisons are reliable.

See [the research plan](docs/RESEARCH-PLAN.md), [architecture decisions](docs/ARCHITECTURE-DECISIONS.md), [experiment matrix](configs/experiments.yaml), and [dataset registry](docs/DATASETS.md).

## Research guardrails

- Loop iterations add computation, not independent parameters or missing knowledge.
- Do not equate recurrent depth with parameter count.
- Keep model/tokenizer revisions, datasets, splits, and evaluation conditions controlled.
- Report parameter count, training tokens, optimizer steps, inference compute/latency, peak memory, and task metrics.
- Separate architectural gains from gains due to additional data, training compute, or distillation.
- Preserve negative results. Do not claim frontier-model equivalence from narrow benchmark wins.
- Verify dataset provenance and licenses before training.

## Hardware and cloud compute

Initial local development target: RTX 3050 Laptop GPU with 4 GB VRAM and 16 GB system RAM. Re-measure actual available resources before training.

Kaggle Notebooks are the primary free cloud GPU target; Colab Free is a secondary interactive option. See [cloud compute strategy](docs/CLOUD-COMPUTE.md). All cloud runs must be checkpointable and resumable. A preconfigured [Kaggle GPU smoke-test notebook](notebooks/kaggle_gpu_smoke_test.ipynb) is provided; select GPU once and use Run All.

## Development

Python 3.10+ is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
ruff check .
```

The package contains a generic shared recurrent block primitive and a Transformers 4.57.1 Llama adapter. The adapter is a research forward-pass implementation only: KV cache and generation are deliberately disabled.

## Current repository status

Implemented:
- Research plan, hypotheses, experimental controls, and decision gates.
- Initial full-decoder-stack recurrent architecture decision.
- Initial experiment configuration.
- Cloud compute assessment, Kaggle smoke-test script, and preconfigured Kaggle notebook.
- Dataset governance registry.
- Generic shared recurrent block primitive and unit tests.
- Atomic checkpoint save/restore with optimizer and RNG state.
- Llama recurrent adapter with native-logit parity test for loop count 1, recurrent gradient test, and padding-mask parity test.
- CPU GitHub Actions test workflow.

Not yet implemented or validated:
- Dataset has not yet been selected or license-approved.
- Full training/evaluation pipeline and resumable checkpoints.
- Autoregressive generation/KV-cache support.
- Actual GPU smoke-test result.
- Full resumable training/evaluation loop.
- Curated, license-reviewed dataset.
- Evaluation harness and benchmark results.
- Cloud execution.

Do not treat this repository as a complete training framework until the baseline/recurrent training pipeline, resumable checkpointing, and cloud smoke tests exist.

## Engineering workflow

- Read [AGENTS.md](AGENTS.md) before coding.
- Follow the [Vibe Coding Playbook](VIBE-CODING-PLAYBOOK.md).
- Record experiment configurations and results in versioned files.
- Keep raw datasets, model weights, caches, and credentials out of Git.
- Update [project state](docs/PROJECT-STATE.md) after substantial work.
