"""Minimal shared-weight recurrent computation primitive."""

from __future__ import annotations

from torch import Tensor, nn


class SharedRecurrentBlock(nn.Module):
    """Apply one module instance repeatedly to a hidden-state tensor.

    The wrapped module is registered exactly once. Every iteration reuses the
    same parameters; this class never clones or averages weights.

    The wrapped module must return a Tensor. Model-specific adapters should
    normalize framework outputs to a hidden-state Tensor before using this
    primitive.
    """

    def __init__(self, block: nn.Module, loop_count: int) -> None:
        super().__init__()
        if isinstance(loop_count, bool) or not isinstance(loop_count, int):
            raise TypeError("loop_count must be an integer")
        if loop_count < 1:
            raise ValueError("loop_count must be at least 1")

        self.block = block
        self.loop_count = loop_count

    def forward(self, hidden_states: Tensor, *args: object, **kwargs: object) -> Tensor:
        """Run the shared block loop_count times and return the final hidden state."""
        for _ in range(self.loop_count):
            hidden_states = self.block(hidden_states, *args, **kwargs)
            if not isinstance(hidden_states, Tensor):
                raise TypeError(
                    "The wrapped block must return a Tensor; add a model-specific "
                    "adapter for modules that return tuples or model-output objects."
                )
        return hidden_states
