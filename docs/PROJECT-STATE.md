# Project State

**Last updated:** 2026-11-26  
**Branch:** main  
**Status:** Architecture and foundational primitive implemented; no model training executed.

## Objective

Measure whether recurrent reuse of shared transformer computation improves a small pretrained language model, and identify diminishing returns under explicit parameter, training-compute, and inference-compute budgets.

## Current decisions

- Initial checkpoint candidate: HuggingFaceTB/SmolLM2-135M.
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
- Unit tests for loop-count validation, weight reuse, gradients, native loop-1 logit parity, and padding-mask parity.
- GitHub Actions CPU test workflow.

## Not implemented / not verified

- Hugging Face model revision and tokenizer revision are not pinned.
- Dataset has not been selected or license-reviewed.
- No training/evaluation loop exists.
- No recurrent autoregressive KV cache or generate() integration exists.
- No evaluation harness exists.
- No cloud notebook has been run.
- No training or benchmark results exist.

## Validation

- Unit tests are configured to run in GitHub Actions.
- Record the workflow URL and exact result after the first run.
- No training result should be reported until an actual run completes and its logs are inspected.

## Next actions

1. Verify the new CI run and fix any failures.
2. Verify CI for the new Llama adapter and fix any failures.
3. Add deterministic checkpoint save/load tests and measure memory/step time on a tiny model.
4. Select a small, license-reviewed training corpus and held-out validation corpus.
5. Implement a resumable training loop with atomic checkpoints and append-only metrics.
6. Run the baseline smoke test on Kaggle before launching any recurrent run.

## Recovery and safety

- Do not commit model weights, raw datasets, cloud credentials, or runtime caches.
- Use frequent atomic checkpoints and append-only metrics logs.
- Never discard existing work or overwrite checkpoints without explicit approval.
- Stop before paid compute, paid storage, or billing changes; obtain explicit budget approval.
