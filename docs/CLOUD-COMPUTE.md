# Cloud Compute Strategy

**Last checked:** 2026-11-26  
**Default:** Free, interruptible notebook GPUs; checkpoint-first workflow.

## Provider assessment

### 1. Kaggle Notebooks — primary free training target

- Official documentation currently describes free P100 GPU access and also lists T4 x2 configurations.
- Notebook sessions are limited to 12 hours for CPU/GPU and 9 hours for TPU.
- GPU availability can be queued during busy periods.
- Kaggle's GPU tips page states up to 30 GPU hours per week; a floating quota experiment may provide more depending on demand.
- /kaggle/working has 20 GB auto-saved storage, but do not treat the runtime filesystem as durable long-term storage.

Best use: reproducible training chunks with frequent checkpoints and logs.

Official sources:
- https://www.kaggle.com/docs/notebooks
- https://www.kaggle.com/page/GPU-tips-and-tricks
- https://www.kaggle.com/discussions/product-feedback/173129

### 2. Google Colab Free — secondary interactive option

- Free GPU access exists, but accelerator type, availability, and usage limits are dynamic and not guaranteed.
- Google does not publish a fixed free GPU quota.
- Free sessions can run for at most 12 hours, depending on availability and usage.

Best use: debugging notebooks, short smoke tests, quick experiments—not a dependable long training scheduler.

Official source:
- https://research.google.com/colaboratory/faq.html

### 3. Lightning AI — useful initial credit, not unlimited free compute

- Current pricing page advertises up to 30 free credits to start; its FAQ says 5 credits on signup and 25 additional credits after adding a card.
- Pricing page estimates those credits can provide roughly 80 hours on interruptible machines, depending on GPU and rates.
- Free tier includes one active CPU Studio; GPU compute consumes credits.
- Phone verification is required; availability may vary by country.

Best use: occasional larger-GPU experiments when free credits are available. Do not add a payment method or enable paid compute without explicit budget approval.

Official sources:
- https://lightning.ai/pricing
- https://platform.lightning.ai/studio

### 4. Hugging Face ZeroGPU — not a training platform for this project

- Free accounts receive 5 minutes of GPU quota per day.
- It is designed for Gradio Spaces and demos, not long-running model training.

Best use: later demo/inference experiments only.

Official source:
- https://huggingface.co/docs/hub/main/spaces-zerogpu

## Recommendation

Start with **Kaggle**. It offers the most predictable free GPU allowance among the options checked, although sessions are interruptible and weekly quotas may vary. Use Colab as a backup for interactive debugging. Treat Lightning AI credits as a limited bonus, not the foundation of the project.

## First Kaggle smoke test

1. Create a Kaggle Notebook and enable a GPU accelerator.
2. Clone this repository in the first cell:
   ```bash
   !git clone https://github.com/tanushgargjanuary-bot/SLM-Training.git
   %cd SLM-Training
   !python -m pip install -e .
   ```
3. Run the tiny baseline and Loop-2 forward/backward/optimizer checks:
   ```bash
   !python scripts/cloud_smoke_test.py --sequence-length 64 --loops 1 2 --output /kaggle/working/smoke-report.json
   ```
4. Download or copy the JSON report from /kaggle/working before the session ends.

The script uses synthetic token IDs, downloads the pinned base checkpoint, performs one optimizer step for each variant, and records GPU memory and step time. It does not save trained weights and is not a research benchmark. Do not launch the full experiment matrix until this smoke test is inspected.

Do not split a single training run across providers until checkpoints, environment setup, and data versions are reproducible.

## Required training-run behaviour

Every run must:
1. Save a checkpoint at a fixed step interval and at the end of each session.
2. Save optimizer, scheduler, RNG, scaler, global step, and sampler state when supported.
3. Write an append-only JSONL metrics log.
4. Record Git commit, config hash, model/tokenizer revision, package versions, GPU name, CUDA/PyTorch versions, and actual dataset revision.
5. Resume from the latest validated checkpoint, not from an arbitrary partially written file.
6. Verify checkpoint integrity after upload/download.
7. Keep credentials in environment secrets; never put tokens in notebooks or Git.

Cloud sessions are ephemeral. A training process that cannot resume safely is not ready for cloud execution.
