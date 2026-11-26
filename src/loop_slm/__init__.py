"""Recurrent-computation research utilities."""

from loop_slm.checkpointing import load_checkpoint, save_checkpoint
from loop_slm.llama import RecurrentLlamaForCausalLM
from loop_slm.recurrent import SharedRecurrentBlock

__all__ = [
    "RecurrentLlamaForCausalLM",
    "SharedRecurrentBlock",
    "load_checkpoint",
    "save_checkpoint",
]
