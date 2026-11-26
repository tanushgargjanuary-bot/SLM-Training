"""Run a tiny GPU-only forward/backward/optimizer smoke test.

This is a pipeline and resource check, not a research experiment. It uses
synthetic token IDs and does not save model weights.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM

from loop_slm.llama import RecurrentLlamaForCausalLM

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"


def run_variant(loop_count: int, sequence_length: int, device: torch.device) -> dict[str, object]:
    torch.cuda.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.train()

    runner: torch.nn.Module
    if loop_count == 1:
        runner = model
    else:
        runner = RecurrentLlamaForCausalLM(model, loop_count=loop_count)
    load_seconds = time.perf_counter() - load_started

    vocab_size = model.config.vocab_size
    input_ids = torch.randint(3, vocab_size, (1, sequence_length), device=device)
    attention_mask = torch.ones_like(input_ids)
    optimizer = torch.optim.AdamW(runner.parameters(), lr=1e-5)

    torch.cuda.synchronize(device)
    step_started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    if loop_count == 1:
        output = runner(
            input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            use_cache=False,
        )
    else:
        output = runner(input_ids, attention_mask=attention_mask, labels=input_ids)
    if output.loss is None or not torch.isfinite(output.loss):
        raise RuntimeError(f"Loop-{loop_count} produced an invalid loss")
    output.loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    step_seconds = time.perf_counter() - step_started

    result = {
        "loop_count": loop_count,
        "load_seconds": round(load_seconds, 3),
        "step_seconds": round(step_seconds, 3),
        "loss": float(output.loss.detach().cpu()),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated(device) / 1024**3, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved(device) / 1024**3, 3),
        "parameter_count": sum(parameter.numel() for parameter in runner.parameters()),
    }

    del output, optimizer, runner, model, input_ids, attention_mask
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--loops", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--output", type=Path, default=Path("smoke-report.json"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not available. Select a GPU accelerator in the notebook.")

    if args.sequence_length < 2:
        raise SystemExit("--sequence-length must be at least 2")
    if any(count < 1 for count in args.loops):
        raise SystemExit("All loop counts must be positive integers")

    device = torch.device("cuda")
    results = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device),
        "sequence_length": args.sequence_length,
        "runs": [
            run_variant(count, args.sequence_length, device)
            for count in args.loops
        ],
        "warning": "Synthetic one-step smoke test only; not benchmark evidence.",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved report: {args.output}")


if __name__ == "__main__":
    main()
