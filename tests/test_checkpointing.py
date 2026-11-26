import random

import torch
from torch import nn

from loop_slm.checkpointing import load_checkpoint, save_checkpoint


def test_checkpoint_restores_model_optimizer_and_progress(tmp_path) -> None:
    torch.manual_seed(101)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn(3, 4)
    model(inputs).square().mean().backward()
    optimizer.step()

    expected_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    path = tmp_path / "state" / "step-00001.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        global_step=1,
        metadata={"model_revision": "test-revision"},
        sampler_state={"epoch": 0, "offset": 3},
    )

    restored_model = nn.Linear(4, 2)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    state = load_checkpoint(path, restored_model, restored_optimizer)

    assert state["global_step"] == 1
    assert state["metadata"]["model_revision"] == "test-revision"
    assert state["sampler_state"] == {"epoch": 0, "offset": 3}
    for expected, actual in zip(expected_parameters, restored_model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
    assert restored_optimizer.state_dict()["state"]


def test_checkpoint_restores_rng_state(tmp_path) -> None:
    torch.manual_seed(202)
    random.seed(202)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "rng.pt"
    save_checkpoint(path, model, optimizer, global_step=0)

    expected_torch = torch.rand(4)
    expected_python = random.random()
    torch.rand(20)
    random.random()

    load_checkpoint(path, model, optimizer)

    torch.testing.assert_close(torch.rand(4), expected_torch)
    assert random.random() == expected_python


def test_checkpoint_write_leaves_no_temporary_file(tmp_path) -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(path, model, optimizer, global_step=0)

    assert path.is_file()
    assert list(tmp_path.iterdir()) == [path]
