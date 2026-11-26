import pytest
import torch
from torch import nn

from loop_slm import SharedRecurrentBlock


def test_loop_count_one_matches_wrapped_module() -> None:
    torch.manual_seed(7)
    block = nn.Linear(4, 4)
    recurrent = SharedRecurrentBlock(block, loop_count=1)
    inputs = torch.randn(2, 3, 4)

    torch.testing.assert_close(recurrent(inputs), block(inputs))


def test_repeated_passes_reuse_the_same_parameters() -> None:
    block = nn.Linear(4, 4)
    recurrent = SharedRecurrentBlock(block, loop_count=8)

    assert recurrent.block is block
    assert len(list(recurrent.parameters())) == len(list(block.parameters()))
    assert sum(p.numel() for p in recurrent.parameters()) == sum(
        p.numel() for p in block.parameters()
    )


def test_gradients_flow_through_repeated_computation() -> None:
    torch.manual_seed(11)
    block = nn.Linear(4, 4)
    recurrent = SharedRecurrentBlock(block, loop_count=3)
    inputs = torch.randn(2, 4, requires_grad=True)

    recurrent(inputs).square().mean().backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    for parameter in block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("loop_count", [0, -1])
def test_rejects_non_positive_loop_count(loop_count: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        SharedRecurrentBlock(nn.Identity(), loop_count)


@pytest.mark.parametrize("loop_count", [True, 1.5, "2"])
def test_rejects_non_integer_loop_count(loop_count: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        SharedRecurrentBlock(nn.Identity(), loop_count)  # type: ignore[arg-type]


def test_rejects_non_tensor_module_output() -> None:
    class TupleBlock(nn.Module):
        def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
            return (hidden_states,)

    recurrent = SharedRecurrentBlock(TupleBlock(), loop_count=2)
    with pytest.raises(TypeError, match="must return a Tensor"):
        recurrent(torch.randn(2, 4))
