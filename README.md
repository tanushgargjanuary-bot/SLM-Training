# Loop SLM Research

A local-first research project investigating whether recurrent computation and shared transformer weights can improve the reasoning efficiency of small language models.

## Primary research question

How much capability can a small pretrained language model gain from recurrent/shared-weight computation, where do additional iterations saturate, and do any gains survive comparisons at controlled training and inference compute?

## Initial objectives

1. Establish a reproducible conventional-model baseline.
2. Convert a compatible pretrained small decoder-only model into a recurrent/shared-weight variant.
3. Verify that the converted model trains correctly and preserves checkpoint/generation behaviour.
4. Compare fixed recurrent depths and measure diminishing returns.
5. Explore gated updates and adaptive computation only after fixed-depth experiments are reliable.
6. Investigate reasoning-focused fine-tuning and teacher distillation separately from architectural gains.

## Research guardrails

- Loop iterations add computation, not independent parameters or missing knowledge.
- Do not equate recurrent depth with parameter count.
- Keep model/tokenizer revisions, datasets, splits, and evaluation conditions controlled.
- Report parameter count, training tokens, optimizer steps, inference compute/latency, peak memory, and task metrics.
- Separate architectural gains from gains due to additional data, training compute, or distillation.
- Preserve negative results. Do not claim frontier-model equivalence from narrow benchmark wins.
- Verify dataset provenance and licenses before training.

## Initial experimental matrix

| Variant | Description | Status |
|---|---|---|
| Baseline | Original pretrained model | Planned |
| Loop-2 | Shared block executed twice | Planned |
| Loop-3 | Shared block executed three times | Planned |
| Loop-4 | Shared block executed four times | Planned |
| Loop-6 | Shared block executed six times | Planned |
| Loop-8 | Shared block executed eight times | Planned |
| Adaptive | Learned halting / variable recurrent depth | Later |

## Hardware target

Initial local development target: RTX 3050 Laptop GPU with 4 GB VRAM and 16 GB system RAM. Re-measure actual available resources before training. Start with short-sequence smoke tests and small checkpoints; do not assume full fine-tuning will fit.

## Current repository status

This repository is a research scaffold. It does not yet contain a validated recurrent model implementation, training pipeline, curated dataset, or benchmark results.

## Engineering workflow

- Read [AGENTS.md](AGENTS.md) before coding.
- Follow the [Vibe Coding Playbook](VIBE-CODING-PLAYBOOK.md).
- Record experiment configurations and results in versioned files.
- Keep raw datasets, model weights, caches, and credentials out of Git.
- Update the research plan and continuity records when implementation begins.

## Planned documentation

- Architecture conversion and layer-mapping design.
- Dataset registry with provenance and licenses.
- Reproducible training configurations.
- Evaluation protocol and benchmark results.
- Hardware measurements and run logs.

Do not treat this repository as a working training framework until the relevant implementation and tests exist.
