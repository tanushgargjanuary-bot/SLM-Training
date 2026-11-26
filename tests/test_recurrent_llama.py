import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from loop_slm.llama import RecurrentLlamaForCausalLM


def make_tiny_llama() -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        attention_dropout=0.0,
    )
    return LlamaForCausalLM(config)


def test_one_loop_matches_native_llama_logits() -> None:
    torch.manual_seed(23)
    model = make_tiny_llama().eval()
    recurrent = RecurrentLlamaForCausalLM(model, loop_count=1).eval()
    input_ids = torch.randint(3, 97, (2, 12))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        expected = model(input_ids, attention_mask=attention_mask, use_cache=False).logits
        actual = recurrent(input_ids, attention_mask=attention_mask).logits

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_recurrent_passes_preserve_parameter_count_and_output_shape() -> None:
    model = make_tiny_llama()
    recurrent = RecurrentLlamaForCausalLM(model, loop_count=3)
    input_ids = torch.randint(3, 97, (2, 12))

    output = recurrent(input_ids)

    assert output.logits.shape == (2, 12, 97)
    assert sum(p.numel() for p in recurrent.parameters()) == sum(
        p.numel() for p in model.parameters()
    )


def test_recurrent_loss_backpropagates() -> None:
    torch.manual_seed(29)
    model = make_tiny_llama()
    recurrent = RecurrentLlamaForCausalLM(model, loop_count=2)
    input_ids = torch.randint(3, 97, (2, 12))

    output = recurrent(input_ids, labels=input_ids)
    assert output.loss is not None
    assert torch.isfinite(output.loss)
    output.loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("loop_count", [0, -2])
def test_rejects_invalid_loop_count(loop_count: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        RecurrentLlamaForCausalLM(make_tiny_llama(), loop_count)


def test_rejects_invalid_inputs() -> None:
    recurrent = RecurrentLlamaForCausalLM(make_tiny_llama(), loop_count=2)
    with pytest.raises(ValueError, match="shape"):
        recurrent(torch.ones(8, dtype=torch.long))



def test_one_loop_matches_native_with_padding_mask() -> None:
    torch.manual_seed(31)
    model = make_tiny_llama().eval()
    recurrent = RecurrentLlamaForCausalLM(model, loop_count=1).eval()
    input_ids = torch.randint(3, 97, (2, 12))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, -3:] = 0

    with torch.no_grad():
        expected = model(input_ids, attention_mask=attention_mask, use_cache=False).logits
        actual = recurrent(input_ids, attention_mask=attention_mask).logits

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
