# SLM-Training — Agent Instructions

## Required intake
1. Read `VIBE-CODING-PLAYBOOK.md` and retrieve the canonical Devkit playbook linked there.
2. Read this README and the relevant research/architecture documentation.
3. Inspect the repository, current branch, working tree, configs, tests, and recent changes before editing.
4. For substantial work, create a plan with hypotheses, scope, acceptance criteria, experiments, validation, risks, and recovery.
5. Keep research claims separate from hypotheses and unverified assumptions.

## Research invariants
- The primary research question is whether recurrent/shared-weight computation improves small-model capability or quality per unit of compute.
- A loop is additional computation, not additional independent parameters or missing knowledge.
- Compare against an unchanged baseline. Keep model revision, tokenizer, data, splits, optimizer budget, and evaluation conditions controlled.
- Report parameter count, training tokens, optimizer steps, inference FLOPs or a documented proxy, latency, peak VRAM, and quality metrics.
- Measure diminishing returns across recurrent depths. Do not assume more iterations are better.
- Preserve negative results and report uncertainty where feasible.
- Check dataset provenance, license, deduplication, contamination risk, and split integrity before training.
- Never commit model weights, raw datasets, credentials, caches, or generated artifacts unless explicitly intended and appropriately licensed.

## Hardware constraints
Initial local target: RTX 3050 Laptop GPU with 4 GB VRAM and 16 GB system RAM. Treat this as a hard design constraint until hardware is re-measured. Start with small smoke tests and short sequence lengths. Do not start a training run likely to exhaust memory without a measured estimate.

## Long-term engineering default
Prefer the best proportionate long-term solution: reproducibility, maintainability, correctness, security, portability, and recoverability over the shortest immediate patch. Proactively make high-value in-scope improvements. Do not silently expand scope or add speculative complexity. Ask before destructive, irreversible, costly, permission-changing, publication, deployment, or externally consequential actions.

## Completion
Run relevant tests and experiments, inspect the final diff, update research notes and continuity records, and report exact commands and measured results. Never claim an experiment succeeded unless it was executed and its outputs inspected.
