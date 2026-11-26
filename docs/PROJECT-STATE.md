# Project State

**Last updated:** 2026-11-26  
**Branch:** main  
**Status:** Architecture and foundational primitive implemented; no model training executed.

## Objective

Measure whether recurrent reuse of shared transformer computation improves a small pretrained language model, and identify diminishing returns under explicit parameter, training-compute, and inference-compute budgets.

## Current decisions

- Initial checkpoint: HuggingFaceTB/SmolLM2-135M, pinned to revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2 (Apache-2.0).
- Initial recurrent unit: full decoder stack, repeated 1/2/3/4/6/8 times.
- Embeddings, final normalization, and LM head execute once.
- Same decoder-layer module instances are reused.
- KV cache remains disabled until a recurrent cache design is implemented and tested.
- First phase is a pretrained-initialization pilot, not a from-scratch reproduction.
- Kaggle is the primary free cloud GPU target; Colab is a backup for interactive tests.

## Implemented

- Research hypotheses, staged experimental protocol, and decision gates in docs/RESEARCH-PLAN.md.
- Initial architecture decision and acceptance tests in docs/ARCHITECTURE-DECISIONS.md.
- Initial matrix in configs/experiments.yaml.
- Free cloud compute assessment and checkpoint requirements in docs/CLOUD-COMPUTE.md.
- Generic SharedRecurrentBlock primitive in src/loop_slm/recurrent.py.
- RecurrentLlamaForCausalLM adapter in src/loop_slm/llama.py, pinned to Transformers 4.57.1 internals.
- GPU smoke-test script in scripts/cloud_smoke_test.py; it performs one optimizer step for baseline and Loop-2 and writes a hardware report.
- Atomic checkpoint save/restore in src/loop_slm/checkpointing.py, including model, optimizer, optional scheduler/scaler, sampler metadata, and Python/NumPy/Torch RNG state.
- Unit tests for loop-count validation, weight reuse, gradients, native loop-1 logit parity, padding-mask parity, and checkpoint/RNG restoration.
- GitHub Actions CPU test workflow.

## Not implemented / not verified

- Dataset has not been selected or license-approved; FineWeb-Edu is only a documented candidate.
- No training/evaluation loop exists.
- No recurrent autoregressive KV cache or generate() integration exists.
- No evaluation harness exists.
- Cloud smoke-test script has not been run on Kaggle or any GPU.
- No training or benchmark results exist.

## Validation

- Unit tests are configured to run in GitHub Actions.
- Record the workflow URL and exact result after the first run.
- No training result should be reported until an actual run completes and its logs are inspected.

## Next actions

1. Verify the CI run for the new Llama adapter and fix any failures.
2. Run scripts/cloud_smoke_test.py on Kaggle; inspect baseline and Loop-2 memory/step-time results.
3. Select a small, license-reviewed training corpus and held-out validation corpus.
4. Implement the baseline/recurrent training loop with append-only metrics and tested checkpoint resume.
5. Run a short controlled pilot on Kaggle and inspect loss, memory, and throughput.

## Recovery and safety

- Do not commit model weights, raw datasets, cloud credentials, or runtime caches.
- Use frequent atomic checkpoints and append-only metrics logs.
- Never discard existing work or overwrite checkpoints without explicit approval.
- Stop before paid compute, paid storage, or billing changes; obtain explicit budget approval.
