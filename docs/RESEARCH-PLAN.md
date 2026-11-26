# Research Plan

**Status:** Design baseline; no training results yet  
**Last updated:** 2026-11-26

## 1. Research objective

Investigate whether repeatedly applying shared transformer computation improves a small pretrained language model's quality per parameter, and identify where extra recurrent passes stop paying for their added compute.

The engineering objective is to produce a reproducible, useful small language model. The research objective is narrower: measure recurrent-depth trade-offs without conflating them with extra parameters, extra data, or extra training.

## 2. Falsifiable hypotheses

- **H1 — Useful recurrence:** A recurrently repeated transformer can improve selected reasoning/generalization metrics over the same pretrained checkpoint after controlled adaptation.
- **H2 — Diminishing returns:** Marginal quality gain decreases as loop count rises.
- **H3 — Compute trade-off:** Some recurrent depths may improve quality at fixed parameter count but lose on quality per inference FLOP or latency.
- **H4 — Adaptation matters:** Recurrent variants need training/adaptation; merely repeating a pretrained model at inference may degrade output.
- **H5 — Task dependence:** Gains, if any, will vary by task. No single benchmark win establishes general intelligence or frontier-model equivalence.

Reject or revise each hypothesis when measurements contradict them.

## 3. Initial model and why

Use HuggingFaceTB/SmolLM2-135M as the initial base checkpoint. Pin model and tokenizer to revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2. The official model card declares Apache-2.0 and reports 134.5M parameters and 2T pretraining tokens.

Its published configuration is a decoder-only Llama architecture with 30 layers, hidden size 576, 9 attention heads, 3 key/value heads, intermediate size 1536, and vocabulary size 49,152. Pin an immutable model revision before any run. Record the resolved commit hash, tokenizer revision, Transformers version, and license metadata in each experiment.

Why this model:
- Small enough for cheap architecture and pipeline experiments.
- A conventional pretrained baseline is available.
- The 30-layer stack provides a natural first full-transformer recurrent block.
- It is a pilot, not a claim that 135M is the final target size.

## 4. Core experimental design

### Primary architecture

Treat the complete sequence of transformer decoder layers as one shared recurrent block. Reapply the same 30 layer modules for a selected number of passes. Keep token embeddings and the language-model head shared as usual. Apply the model's final normalization and output head once after the final recurrent pass.

Initial fixed-depth variants:
- Baseline: one normal pass.
- Loop-2, Loop-3, Loop-4, Loop-6, Loop-8: the same decoder stack is executed 2, 3, 4, 6, or 8 times.

The model's parameter count should remain constant across these variants because the same layer objects are reused. Compute and activation memory will increase with loop count. This is the central trade-off, not a free capability increase.

### Important implementation constraints

- Reuse the exact same decoder-layer parameter objects; do not clone weights.
- Do not loop embeddings, final normalization, or the LM head in the first implementation.
- Recompute position-dependent rotary embeddings consistently for the same token positions on each pass.
- Disable KV caching during initial training and full-sequence evaluation. Recurrent autoregressive KV caching needs a separate, correctly indexed cache per layer and loop; do not silently reuse an ordinary cache.
- Ensure attention masks, padding masks, position IDs, and cache positions remain correct.
- Add tests for parameter identity, gradient flow through every pass, output shape, and loop-count validation.
- Compare one-pass baseline and recurrent variants from the same immutable starting checkpoint.

This is a pretrained-initialization pilot. It is not equivalent to training a recurrent transformer from scratch. The distinction must remain explicit in reporting.

## 5. Training and controls

Run in stages. Do not begin a long run until the pipeline passes a tiny deterministic smoke test.

### Stage A — Functional smoke test
- Use a tiny synthetic batch and a short sequence.
- Verify forward pass, finite loss, backward pass, finite gradients, optimizer step, and checkpoint save/load.
- Test baseline and at least Loop-2.
- Measure peak allocated/reserved VRAM and wall-clock step time.

### Stage B — Frozen-weight diagnostic
- Evaluate repeated passes without adaptation.
- Purpose: measure whether naïve repetition is stable or harmful.
- This is diagnostic only, not the final comparison.

### Stage C — Controlled adaptation
- Fine-tune each variant from the same base revision.
- Keep dataset order, token count, tokenizer, sequence length, optimizer, effective batch size, optimizer steps, and seed controlled.
- Start with a small pilot token budget; expand only after throughput and loss curves are understood.
- If loop variants have different compute per optimizer step, report both equal-token/equal-step and equal-compute comparisons where feasible. Do not pretend these controls are interchangeable.

### Stage D — Evaluation
Track:
- Validation next-token loss / perplexity on a held-out text corpus.
- Reasoning and generalization tasks appropriate to a base (non-instruct) model.
- Exact benchmark versions, prompts, few-shot settings, decoding parameters, and evaluation code revision.
- Parameter count, training tokens, optimizer steps, estimated or measured FLOPs, inference latency, peak VRAM, and energy if measurable.
- At least three seeds for any result presented as a robust finding, subject to compute availability. Mark single-seed results exploratory.

Potential benchmark shortlist for the pilot: WikiText-2 validation perplexity, PIQA, HellaSwag, ARC-Easy, and GSM8K. Verify licenses, dataset versions, and contamination risks before use. Keep benchmark training data separate from evaluation data.

## 6. Compute-aware interpretation

Report at least three views:
1. **Quality at fixed parameter count** — all loop variants share the same model weights.
2. **Quality at fixed training budget** — equal optimizer steps/tokens, with compute difference explicitly shown.
3. **Quality per inference cost** — quality against measured latency and a documented FLOP proxy.

Do not claim a loop count is "best" without stating the objective function and compute constraint.

## 7. Research context

The October 2026 paper [How to Loop Your Transformer: Loop Scaling, Specialization, and Parallel Inference](https://arxiv.org/abs/2610.40325) studies recurrent reuse in decoder-only transformers. Its abstract reports that full-transformer looping is strongest at a fixed parameter budget, while looping a middle block can be more compute-efficient when training FLOPs are constrained.

Our first experiment uses a pretrained 135M checkpoint and therefore is not a direct reproduction of that paper's from-scratch experiments. Use the paper to inform hypotheses, not to assume our result.

## 8. Decision gates

Do not progress to larger models or adaptive halting until:
- Baseline reproduces a stable validation loss.
- Loop-2 passes functional and checkpoint tests.
- At least one recurrent variant completes a controlled pilot.
- Evaluation and compute accounting are reproducible.
- Results show whether recurrence is promising, neutral, or harmful under the stated budget.

Only then consider adaptive halting, gated state updates, teacher distillation, or a larger 360M/600M model.

## 9. Current status

- [x] Research objective recovered.
- [x] Initial model candidate selected.
- [x] First recurrent architecture specified.
- [x] Controlled experiment matrix defined.
- [x] Pin base model/tokenizer revision and record model license.
- [ ] Implement and test recurrent model adapter.
- [ ] Implement baseline training/evaluation pipeline.
- [ ] Run smoke tests.
- [ ] Run controlled pilot
