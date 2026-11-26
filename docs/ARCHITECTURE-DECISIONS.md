# Architecture Decisions

## ADR-001: Reuse the full decoder stack as the recurrent block

**Status:** Accepted for the first pilot; subject to implementation validation  
**Date:** 2026-11-26

### Context

The project needs an architecture that isolates recurrent computation while preserving a controlled comparison against a pretrained baseline. The initial checkpoint candidate is HuggingFaceTB/SmolLM2-135M, whose configuration exposes a 30-layer decoder stack.

### Decision

The initial recurrent variant will repeatedly execute the same decoder-layer modules, in their original order, for N passes. Embeddings run once before the loop. Final normalization and the language-model head run once after the final pass.

The same module instances and parameters must be reused. A recurrent pass must not create a copied layer stack.

### Why this first

- Keeps parameter count fixed across loop counts.
- Starts from the same pretrained checkpoint as the baseline.
- Tests a clean, falsifiable question: what does extra depth of shared computation do?
- Aligns with recent work reporting strong quality at fixed parameter budgets for full-transformer looping.

### Alternatives not selected for the first pilot

1. **Tie arbitrary adjacent layers from the pretrained model.** Rejected because layers were pretrained for different representational roles; naïve weight averaging/tieing may damage the checkpoint and changes parameter count.
2. **Loop only a middle block.** Keep as a follow-up because recent work suggests it may be attractive under constrained training FLOPs, but it introduces a different architecture and requires its own controls.
3. **Train a recurrent model from scratch.** Scientifically valuable and closer to some recent literature, but too expensive for the first pipeline-validation step.
4. **Distill a large teacher immediately.** Deferred because it would confound architecture gains with teacher knowledge and data quality.

### Consequences and risks

- FLOPs and activation memory rise with loop count.
- Repeating a pretrained full stack without adaptation may destabilize representations; measure this separately.
- Correct autoregressive KV caching is nontrivial because each recurrent pass needs distinct cached states. Initial training and evaluation will disable cache. A separate cache design is required before claiming efficient generation latency.
- This is a pretrained-initialization study, not a direct reproduction of from-scratch recurrent-transformer papers.

### Acceptance tests

- Parameter identities are shared across passes.
- Parameter count is invariant across loop counts.
- Gradients propagate through every pass.
- Causal and padding masks are preserved.
- Loop count 1 matches the baseline logits within numerical tolerance.
- Checkpoint save/load preserves outputs.
- Cache is disabled until a tested recurrent cache implementation exists.
