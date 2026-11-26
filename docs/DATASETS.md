# Dataset Registry and Governance

**Status:** No training or evaluation dataset is approved yet.

## Selection requirements

Before any dataset enters a run, record:
- Exact dataset ID, config/subset, split, immutable revision, and retrieval date.
- Dataset card, license, upstream source, and any upstream terms.
- Intended use and whether redistribution is permitted.
- Filtering, deduplication, normalization, and tokenization steps.
- Train/validation/test split logic and contamination checks.
- Known risks: personal data, harmful content, benchmark leakage, duplicated samples, and domain imbalance.

Do not assume that a public download URL means unrestricted training or redistribution rights.

## Candidate: FineWeb-Edu

- Dataset: HuggingFaceFW/fineweb-edu
- Candidate config: sample-10BT, only if a small controlled sample is needed.
- Declared license: Open Data Commons Attribution (ODC-By) v1.0.
- The dataset card also states that use is subject to Common Crawl Terms of Use.
- The card reports a very large corpus and documents residual toxic/harmful material.
- SmolLM2's model card reports FineWeb-Edu among its pretraining sources. Exact overlap with the checkpoint's training examples is unknown.

Decision: candidate only, not approved. A pilot can use a small sample for pipeline validation, but validation loss on this source must be interpreted cautiously because the base checkpoint was already trained on FineWeb-Edu-derived data.

Source: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

## Evaluation candidates

| Dataset | Proposed use | Required checks |
|---|---|---|
| WikiText-2 | Held-out language-model loss/perplexity | Pin version and split; verify license and exact evaluation protocol |
| PIQA | Physical commonsense | Pin dataset revision, prompt format, and metric |
| HellaSwag | Commonsense completion | Check benchmark license and contamination risk |
| ARC-Easy | Science question answering | Pin revision and exact evaluation harness |
| GSM8K | Grade-school math reasoning | Pin revision, answer extraction, and prompt protocol |

No benchmark result should be reported until the exact dataset version, license, split, prompt, metric, and evaluation code are recorded.

## Initial policy

Use the smallest dataset slice that can answer the immediate experiment question. Keep all variants on identical data order and token budgets. Never mix evaluation examples into training. Preserve source IDs and hashes so that filtering and deduplication can be reproduced.
