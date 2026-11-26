"""Recurrent wrapper for Hugging Face Llama causal language models.

This adapter targets Transformers 4.57.1. It intentionally disables KV caching;
recurrent autoregressive caching requires a separate cache per loop and layer.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import CausalLMOutputWithPast


class RecurrentLlamaForCausalLM(nn.Module):
    """Run the same Llama decoder stack multiple times with shared weights.

    The supplied model is retained as a single child module. Its embeddings are
    evaluated once, decoder layers are reused in-place for every pass, and the
    final norm and language-model head are evaluated once.
    """

    def __init__(self, model: nn.Module, loop_count: int = 1) -> None:
        super().__init__()
        if isinstance(loop_count, bool) or not isinstance(loop_count, int):
            raise TypeError("loop_count must be an integer")
        if loop_count < 1:
            raise ValueError("loop_count must be at least 1")

        required = ("model", "lm_head")
        if any(not hasattr(model, name) for name in required):
            raise TypeError("model must expose LlamaForCausalLM-compatible model and lm_head")
        backbone = model.model
        if not all(hasattr(backbone, name) for name in ("embed_tokens", "layers", "norm", "rotary_emb")):
            raise TypeError("model.model must expose LlamaModel-compatible decoder components")

        self.backbone = model
        self.loop_count = loop_count

    @property
    def config(self) -> Any:
        return self.backbone.config

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> CausalLMOutputWithPast:
        """Return logits and optional next-token cross-entropy loss.

        Args:
            input_ids: Token IDs with shape (batch, sequence).
            attention_mask: Optional 2D padding mask with shape (batch, sequence).
            position_ids: Optional position IDs with shape (batch, sequence).
            labels: Optional labels with shape (batch, sequence); -100 is ignored.

        KV caching is deliberately unsupported. Pass a complete sequence.
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        if input_ids.shape[1] < 1:
            raise ValueError("input_ids must contain at least one token")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must have the same shape as input_ids")

        base = self.backbone.model
        inputs_embeds = base.embed_tokens(input_ids)
        batch_size, sequence_length = input_ids.shape
        cache_position = torch.arange(sequence_length, device=input_ids.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

        causal_mask = create_causal_mask(
            config=base.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )
        position_embeddings = base.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for _ in range(self.loop_count):
            for decoder_layer in base.layers:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

        hidden_states = base.norm(hidden_states)
        logits = self.backbone.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if sequence_length < 2:
                raise ValueError("labels require a sequence length of at least 2")
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)
