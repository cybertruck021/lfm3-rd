# Copyright 2026 The HuggingFace Team, Liquid AI, and LFM3-RD Authors. All rights reserved.
#
# Licensed under the LFM Open License v1.0 (the "License");
# you may not use this file except in compliance with the License.

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig


@strict
class Lfm3RDConfig(PreTrainedConfig):
    model_type = "lfm3_rd"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 1000000.0

    vocab_size: int = 128_000
    hidden_size: int = 2048
    intermediate_size: int = 7680
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    max_position_embeddings: int = 262_144
    initializer_range: float = 0.02
    norm_eps: float = 0.00001
    use_cache: bool = True
    pad_token_id: int | None = 124_893
    bos_token_id: int | None = 124_894
    eos_token_id: int | list[int] | None = 124_900
    tie_word_embeddings: bool = True
    rope_parameters: dict | None = None
    conv_bias: bool = False
    conv_L_cache: int = 7
    layer_types: list[str] | None = None
    _attention_layer_types = ("full_attention", "sliding_attention")
    head_dim: int | None = None
    sliding_window: int | None = 256
    use_sandwich_norm: bool = True

    num_prelude_layers: int = 2
    num_motifs: int = 5
    motif_size: int = 4
    num_coda_layers: int = 2

    reasoning_mode: str = "manual"
    reasoning_effort: str | None = None
    recurrent_loops: int = 5
    min_recurrent_loops: int = 1
    max_recurrent_loops: int = 32
    act_halting_threshold: float = 0.80

    mono_loss_weight: float = 0.25
    settle_loss_weight: float = 0.10
    halt_loss_weight: float = 0.50

    def __post_init__(self, **kwargs):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.rope_parameters is None:
            self.rope_parameters = {
                "full_attention": {
                    "rope_type": "default",
                    "rope_theta": 1_000_000.0,
                    "partial_rotary_factor": 0.5,
                },
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10_000.0,
                },
            }
        if self.layer_types is None:
            self.layer_types = [
                "full_attention"
                if i % 4 == 2
                else "sliding_attention"
                if i % 4 == 0 and i > 0
                else "conv"
                for i in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = [
                lt
                if lt in ("full_attention", "sliding_attention", "conv")
                else "conv"
                for lt in self.layer_types
            ]
        self.tie_word_embeddings = kwargs.pop(
            "tie_embedding", self.tie_word_embeddings
        )
        for legacy_key in (
            "rank_attn",
            "rank_conv",
            "rank_kv",
            "conv_lora_groups",
            "act_kl_prior_lambda",
        ):
            kwargs.pop(legacy_key, None)
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        expected = (
            self.num_prelude_layers
            + self.num_motifs * self.motif_size
            + self.num_coda_layers
        )
        if self.num_hidden_layers != expected:
            raise ValueError(
                f"`num_hidden_layers` ({self.num_hidden_layers}) must equal {expected}."
            )
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("`layer_types` length mismatch.")
        if self.motif_size != 4:
            raise ValueError("LFM3-RD supports `motif_size == 4` only.")
        if self.reasoning_mode not in ("manual", "auto"):
            raise ValueError("`reasoning_mode` must be 'manual' or 'auto'.")
        if self.reasoning_effort is not None and self.reasoning_effort not in (
            "low",
            "fast",
            "medium",
            "high",
            "max",
            "auto",
        ):
            raise ValueError("Invalid `reasoning_effort`.")
        if not (1 <= self.recurrent_loops <= 32):
            raise ValueError("`recurrent_loops` must be in [1, 32].")


__all__ = ["Lfm3RDConfig"]