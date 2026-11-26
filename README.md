# Loop SLM Research

A local-first research project investigating whether recurrent computation and shared transformer weights can improve the reasoning efficiency of small language models.

## Primary research question

How much capability can a small pretrained language model gain from recurrent/shared-weight computation, where do additional iterations saturate, and do any gains survive comparisons at controlled training and inference compute?

## Initial research direction

- Base checkpoint candidate: HuggingFaceTB/SmolLM2-135M.
- Initial recurrent unit: the complete 30-layer decoder stack, repeatedly applied with shared parameters.
- Fixed depths: 1, 2, 3, 4, 6, and 8 passes.
- Embeddings, final normalization, and language-model head execute once.
- Adaptive halting and teacher distillation are deferred until fixed-depth comparisons are reliable.

See [the research plan](docs/RESEARCH-PLAN.md), [architecture decisions](docs/ARCHITECTURE-DECISIONS.md), and [experiment matrix](configs/experiments.yaml).

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

Kaggle Notebooks are the primary free cloud GPU target; Colab Free is a secondary interactive option. See [cloud compute strategy](docs/CLOUD-COMPUTE.md). All cloud runs must be checkpointable and resumable.

## Development

Python 3.10+ is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
ruff check .
```

The package currently contains a generic shared recurrent block primitive. It is not yet a Transformers/Llama model adapter.

## Current repository status

Implemented:
- Research plan, hypotheses, experimental controls, and decision gates.
- Initial full-decoder-stack recurrent architecture decision.
- Initial experiment configuration.
- Cloud compute assessment.
- Generic shared recurrent block primitive and unit tests.
- CPU GitHub Actions test workflow.

Not yet implemented or validated:
- Pinned model/tokenizer revision.
- Transformers/Llama adapter.
- Baseline and recurrent training pipeline.
- Curated, license-reviewed dataset.
- Evaluation harness and benchmark results.
- Cloud execution.

Do not treat this repository as a working training framework until the model adapter, training pipeline, and tests exist.

## Engineering workflow

- Read [AGENTS.md](AGENTS.md) before coding.
- Follow the [Vibe Coding Playbook](VIBE-CODING-PLAYBOOK.md).
- Record experiment configurations and results in versioned files.
- Keep raw datasets, model weights, caches, and credentials out of Git.
- Update [project state](docs/PROJECT-STATE.md) after substantial work.
