# Copyright 2026 The HuggingFace Team, Liquid AI, and LFM3-RD Authors. All rights reserved.
#
# Licensed under the LFM Open License v1.0 (the "License");
# you may not use this file except in compliance with the License.

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.configuration_utils import PreTrainedConfig
from transformers.generation import GenerationMixin
from transformers.integrations import (
    use_kernel_forward_from_hub,
    use_kernel_func_from_hub_with_fallback,
    use_kernelized_func,
)
from transformers.integrations.accelerate import force_accelerate_hooks
from transformers.masking_utils import (
    create_causal_mask,
    create_recurrent_attention_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from .configuration_lfm3_rd import Lfm3RDConfig
from .lfm3_conv import can_use_lfm3_conv, lfm3_causal_conv1d, lfm3_causal_conv1d_update


EFFORT_TO_LOOPS = {"low": 1, "fast": 2, "medium": 5, "high": 12, "max": 32}


@dataclass
class Lfm3RDModelOutputWithPast(BaseModelOutputWithPast):
    ponder_loss: torch.FloatTensor | None = None
    recurrent_steps: int | None = None
    step_halting_probs: list[torch.Tensor] | None = None
    step_pred_depths: list[torch.Tensor] | None = None
    base_t5_coda_hidden: torch.FloatTensor | None = None


@dataclass
class Lfm3RDCausalLMOutputWithPast(CausalLMOutputWithPast):
    ponder_loss: torch.FloatTensor | None = None
    recurrent_steps: int | None = None
    deep_gain_loss: torch.FloatTensor | None = None


@use_kernel_forward_from_hub("RMSNorm")
class Lfm3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(hidden_states, self.weight.shape, self.weight.to(hidden_states.dtype), self.variance_epsilon)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Lfm3SharedCoreBank(nn.Module):
    def __init__(self, config: Lfm3RDConfig):
        super().__init__()
        D = config.hidden_size
        head_dim = getattr(config, "head_dim", None) or (D // config.num_attention_heads)
        q_dim = config.num_attention_heads * head_dim
        kv_dim = config.num_key_value_heads * head_dim
        shapes = [
            (q_dim, D), (kv_dim, D), (kv_dim, D), (D, q_dim),
            (3 * D, D), (D, D),
            (q_dim, D), (kv_dim, D), (kv_dim, D), (D, q_dim),
            (3 * D, D), (D, D),
        ]
        self.weights = nn.ParameterList([nn.Parameter(torch.empty(s)) for s in shapes])


class Lfm3SharedLoRALinear(nn.Module):
    def __init__(self, core_bank: Lfm3SharedCoreBank, core_idx: int, in_features: int, out_features: int, rank: int):
        super().__init__()
        object.__setattr__(self, "_core_bank", core_bank)
        object.__setattr__(self, "_w_cache", None)
        self.core_idx = int(core_idx)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.B = nn.Parameter(torch.empty((out_features, rank)))
        self.A = nn.Parameter(torch.empty((rank, in_features)))

    @property
    def shared_weight(self) -> torch.Tensor:
        return self._core_bank.weights[self.core_idx]

    def train(self, mode: bool = True):
        if mode:
            object.__setattr__(self, "_w_cache", None)
        return super().train(mode)

    def _apply(self, fn, recurse: bool = True):
        object.__setattr__(self, "_w_cache", None)
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        object.__setattr__(self, "_w_cache", None)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _get_eval_weight(self, dt: torch.dtype, dev: torch.device) -> torch.Tensor:
        w = self._w_cache
        if w is None or w.dtype != dt or w.device != dev:
            with torch.no_grad():
                w = torch.addmm(self.shared_weight.float(), self.B.float(), self.A.float()).to(dtype=dt, device=dev).contiguous()
            object.__setattr__(self, "_w_cache", w)
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        if not self.training:
            return F.linear(x, self._get_eval_weight(dt, x.device))
        base = F.linear(x, self.shared_weight.to(dt))
        return base + F.linear(F.linear(x, self.A.to(dt)), self.B.to(dt))

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}, core_idx={self.core_idx}"


class Lfm3SharedGroupedLoRALinear(nn.Module):
    def __init__(self, core_bank: Lfm3SharedCoreBank, core_idx: int, in_features: int, group_out_features: int,
                 rank: int, num_groups: int = 3):
        super().__init__()
        object.__setattr__(self, "_core_bank", core_bank)
        object.__setattr__(self, "_w_cache", None)
        self.core_idx = int(core_idx)
        self.in_features = int(in_features)
        self.group_out_features = int(group_out_features)
        self.rank = int(rank)
        self.num_groups = int(num_groups)
        self.B = nn.Parameter(torch.empty((num_groups, group_out_features, rank)))
        self.A = nn.Parameter(torch.empty((num_groups, rank, in_features)))

    @property
    def shared_weight(self) -> torch.Tensor:
        return self._core_bank.weights[self.core_idx]

    def train(self, mode: bool = True):
        if mode:
            object.__setattr__(self, "_w_cache", None)
        return super().train(mode)

    def _apply(self, fn, recurse: bool = True):
        object.__setattr__(self, "_w_cache", None)
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        object.__setattr__(self, "_w_cache", None)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _get_eval_weight(self, dt: torch.dtype, dev: torch.device) -> torch.Tensor:
        w = self._w_cache
        if w is None or w.dtype != dt or w.device != dev:
            with torch.no_grad():
                lora_full = torch.bmm(self.B.float(), self.A.float()).reshape(-1, self.in_features)
                w = (self.shared_weight.float() + lora_full).to(dtype=dt, device=dev).contiguous()
            object.__setattr__(self, "_w_cache", w)
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        if not self.training:
            return F.linear(x, self._get_eval_weight(dt, x.device))
        base = F.linear(x, self.shared_weight.to(dt))
        g, r, d_in = self.A.shape
        z = F.linear(x, self.A.to(dt).view(g * r, d_in)).view(-1, g, r).transpose(0, 1)
        lora = torch.bmm(z, self.B.to(dt).transpose(1, 2)).transpose(0, 1).reshape(*x.shape[:-1], -1)
        return base + lora

    def extra_repr(self):
        return f"in={self.in_features}, group_out={self.group_out_features}, groups={self.num_groups}, rank={self.rank}"


class Lfm3RDController(nn.Module):
    def __init__(self, config: Lfm3RDConfig):
        super().__init__()
        D = config.hidden_size
        self.hidden_size = D
        self.num_motifs = config.num_motifs
        router_dim = 128

        self.act_norm = Lfm3RMSNorm(D, eps=config.norm_eps)
        self.vel_norm = Lfm3RMSNorm(D, eps=config.norm_eps)

        self.state_proj = nn.Linear(D, router_dim, bias=False)
        self.vel_proj = nn.Linear(D, router_dim, bias=False)
        self.dyn_proj = nn.Linear(4, router_dim, bias=True)
        self.halting_head = nn.Linear(router_dim, 1, bias=True)

        self.step_scales = nn.Parameter(torch.ones(32))

        inj_rank = 64
        self.inj_norm = Lfm3RMSNorm(D, eps=config.norm_eps)
        self.inj_A = nn.Parameter(torch.randn(inj_rank, D) * (1.0 / math.sqrt(D)))
        self.inj_B = nn.Parameter(torch.randn(D, inj_rank) * 0.015)
        self.inj_gates = nn.Parameter(torch.full((6,), 0.20))

        ref_rank = 64
        self.ref_norm = Lfm3RMSNorm(D, eps=config.norm_eps)
        self.ref_A = nn.Parameter(torch.randn(self.num_motifs, ref_rank, D) * (1.0 / math.sqrt(D)))
        self.ref_B = nn.Parameter(torch.randn(self.num_motifs, D, ref_rank) * 0.008)

        self.refine_gates = nn.Parameter(torch.zeros(32))
        self.refine_token_proj = nn.Linear(D, 1, bias=True)

    def predict_depth_and_prob(
        self,
        h_curr: torch.Tensor,
        h_prev: torch.Tensor,
        prelude_out: torch.Tensor,
        step_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w_dt = self.state_proj.weight.dtype
        h_c_f = h_curr.float()
        h_p_f = h_prev.float()
        pre_f = prelude_out.float()

        delta = h_c_f - h_p_f
        norm_c = h_c_f.norm(dim=-1, keepdim=True).clamp(min=1e-5)
        norm_p = h_p_f.norm(dim=-1, keepdim=True).clamp(min=1e-5)
        norm_pre = pre_f.norm(dim=-1, keepdim=True).clamp(min=1e-5)

        rel_vel = (delta.norm(dim=-1, keepdim=True) / norm_c).clamp(0.0, 5.0)
        rel_drift = ((h_c_f - pre_f).norm(dim=-1, keepdim=True) / norm_pre).clamp(0.0, 10.0)
        cos_sim = ((h_c_f * h_p_f).sum(dim=-1, keepdim=True) / (norm_c * norm_p)).clamp(-1.0, 1.0)
        cos_instab = 1.0 - cos_sim
        depth_coord = torch.full_like(rel_vel, float(step_idx) / 31.0)

        dyn_feats = torch.cat([rel_vel, rel_drift, cos_instab, depth_coord], dim=-1).to(w_dt)
        h_normed = self.act_norm(h_curr).to(w_dt)
        v_normed = self.vel_norm(delta.to(h_curr.dtype)).to(w_dt)

        hidden_router = F.silu(self.state_proj(h_normed) + self.vel_proj(v_normed) + self.dyn_proj(dyn_feats))
        raw_logit = self.halting_head(hidden_router).squeeze(-1).float()

        s_pred = 31.0 * torch.sigmoid(raw_logit)
        p_ready = torch.sigmoid(2.5 * (float(step_idx) - s_pred + 1.15))
        return p_ready, s_pred

    def inject_feedback(self, prelude_out: torch.Tensor, h_latent: torch.Tensor, cycle_idx: int) -> torch.Tensor:
        dt = prelude_out.dtype
        c_idx = max(0, min(5, int(cycle_idx) - 1))
        normed_lat = self.inj_norm(h_latent - prelude_out).to(dt)
        delta_inj = 0.08 * normed_lat + F.linear(F.linear(normed_lat, self.inj_A.to(dt)), self.inj_B.to(dt))
        gate = torch.tanh(self.inj_gates[c_idx]).to(dt)
        return prelude_out + gate * delta_inj

    def compute_deep_refinement_step(
        self,
        step_idx: int,
        m_idx: int,
        h_stream: torch.Tensor,
        delta_new: torch.Tensor,
        old_delta: torch.Tensor,
        h_latent: torch.Tensor,
        prev_ready_prob: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dt = h_latent.dtype
        w_dt = self.refine_token_proj.weight.dtype
        s_clamped = max(0, min(31, int(step_idx)))

        normed_lat = self.act_norm(h_latent).to(w_dt)
        tok_logit = self.refine_token_proj(normed_lat).float() + self.refine_gates[s_clamped].float()
        tok_gate = torch.sigmoid(tok_logit)
        if prev_ready_prob is not None:
            unready_factor = (1.0 - 0.85 * prev_ready_prob.detach().float().unsqueeze(-1)).clamp(0.08, 1.0)
            tok_gate = tok_gate * unready_factor
        alpha_tok = tok_gate.to(dt)

        normed_stream = self.ref_norm(h_stream).to(dt)
        A_m = self.ref_A[m_idx].to(dt)
        B_m = self.ref_B[m_idx].to(dt)
        adapter_corr = F.linear(F.linear(normed_stream, A_m), B_m)

        step_corr = (delta_new - old_delta) + adapter_corr
        return alpha_tok, step_corr


class Lfm3RotaryEmbedding(nn.Module):
    def __init__(self, config: Lfm3RDConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.layer_types = sorted(set(config.layer_types) & set(config.rope_parameters))
        self.rope_type = {}
        for layer_type in self.layer_types:
            rope_params = self.config.rope_parameters[layer_type]
            self.rope_type[layer_type] = rope_params["rope_type"]
            curr_inv_freq, curr_attention_scaling = self.compute_rope_parameters(config, device, layer_type=layer_type)
            setattr(self, f"{layer_type}_inv_freq", nn.Buffer(curr_inv_freq, persistent=False))
            setattr(self, f"{layer_type}_original_inv_freq", nn.Buffer(curr_inv_freq.clone(), persistent=False))
            setattr(self, f"{layer_type}_attention_scaling", curr_attention_scaling)

    @staticmethod
    def compute_default_rope_parameters(config, device=None, layer_type=None, **kwargs):
        base = config.rope_parameters[layer_type]["rope_theta"]
        partial_rotary_factor = config.rope_parameters[layer_type].get("partial_rotary_factor", 1.0)
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        rope_angles = int(partial_rotary_factor * dim) // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float) / dim))
        inv_freq = torch.cat((inv_freq, torch.zeros(dim // 2 - rope_angles, dtype=torch.float)))
        return inv_freq.to(device), 1.0

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids, layer_type):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")
        freqs = position_ids[..., None].float() * inv_freq.to(device=x.device, dtype=torch.float)
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * attention_scaling).to(dtype=x.dtype), (emb.sin() * attention_scaling).to(dtype=x.dtype)

    @classmethod
    def compute_rope_parameters(cls, config, device=None, layer_type=None):
        rope_params = config.rope_parameters[layer_type]
        if rope_params["rope_type"] == "default":
            return cls.compute_default_rope_parameters(config, device, layer_type=layer_type)
        full_config = copy.copy(config)
        full_config.rope_parameters = {**config.rope_parameters, layer_type: {**rope_params, "partial_rotary_factor": 1.0}}
        inv_freq, attention_factor = ROPE_INIT_FUNCTIONS[rope_params["rope_type"]](full_config, device, layer_type=layer_type)
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        rope_angles = int(rope_params.get("partial_rotary_factor", 1.0) * dim) // 2
        inv_freq = torch.cat((inv_freq[:rope_angles], inv_freq.new_zeros(dim // 2 - rope_angles)))
        return inv_freq, attention_factor


class Lfm3MLP(nn.Module):
    def __init__(self, config: Lfm3RDConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def forward_detached_weights(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        h1 = F.linear(x, self.w1.weight.detach().to(dt))
        h3 = F.linear(x, self.w3.weight.detach().to(dt))
        return F.linear(F.silu(h1) * h3, self.w2.weight.detach().to(dt))


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
    return attn_output, attn_weights


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim).float()
    sin = sin.unsqueeze(unsqueeze_dim).float()
    q_f = q.float()
    k_f = k.float()
    q_embed = (q_f * cos) + (rotate_half(q_f) * sin)
    k_embed = (k_f * cos) + (rotate_half(k_f) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def apply_rotary_pos_emb_key_only(k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim).float()
    sin = sin.unsqueeze(unsqueeze_dim).float()
    k_f = k.float()
    k_embed = (k_f * cos) + (rotate_half(k_f) * sin)
    return k_embed.to(k.dtype)


def _core_pos(config: Lfm3RDConfig, layer_idx: int, shared_core):
    start = config.num_prelude_layers
    end = start + config.num_motifs * config.motif_size
    if shared_core is not None and start <= layer_idx < end:
        return (layer_idx - start) % config.motif_size
    return None


class Lfm3Attention(nn.Module):
    def __init__(self, config: Lfm3RDConfig, layer_idx: int, shared_core: Lfm3SharedCoreBank | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = True
        q_dim = config.num_attention_heads * self.head_dim
        kv_dim = config.num_key_value_heads * self.head_dim

        pos = _core_pos(config, layer_idx, shared_core)
        if pos is not None:
            b = 0 if pos == 0 else 6
            self.q_proj = Lfm3SharedLoRALinear(shared_core, b + 0, config.hidden_size, q_dim, config.rank_attn)
            self.k_proj = Lfm3SharedLoRALinear(shared_core, b + 1, config.hidden_size, kv_dim, config.rank_kv)
            self.v_proj = Lfm3SharedLoRALinear(shared_core, b + 2, config.hidden_size, kv_dim, config.rank_kv)
            self.o_proj = Lfm3SharedLoRALinear(shared_core, b + 3, q_dim, config.hidden_size, config.rank_attn)
        else:
            self.q_proj = nn.Linear(config.hidden_size, q_dim, bias=False)
            self.k_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
            self.v_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
            self.o_proj = nn.Linear(q_dim, config.hidden_size, bias=False)

        self.q_norm = Lfm3RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_norm = Lfm3RMSNorm(self.head_dim, eps=config.norm_eps)
        self.gate_proj = nn.Linear(config.hidden_size, config.num_attention_heads, bias=False)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(*hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(*hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(*hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0, scaling=self.scaling, sliding_window=self.sliding_window, **kwargs,
        )
        gate = torch.sigmoid(self.gate_proj(hidden_states)).unsqueeze(-1)
        attn_output = (attn_output * gate).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights

    def update_cache_only(self, normed_hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor], past_key_values: Cache):
        input_shape = normed_hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        key_states = self.k_norm(self.k_proj(normed_hidden_states).view(*hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(normed_hidden_states).view(*hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        key_states = apply_rotary_pos_emb_key_only(key_states, cos, sin)
        past_key_values.update(key_states, value_states, self.layer_idx)


def apply_mask_to_padding_states(hidden_states, attention_mask):
    if attention_mask is not None:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states


@use_kernel_func_from_hub_with_fallback("causal_conv1d_update", "causal_conv1d")
def causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = out[:, :, -seq_len:]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


@use_kernel_func_from_hub_with_fallback("causal_conv1d_fn", "causal_conv1d")
def causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
    _, hidden_size, seq_len = hidden_states.shape
    padding = weight.shape[-1] - 1
    out = F.conv1d(hidden_states.to(weight.dtype), weight=weight.unsqueeze(1), bias=bias,
                   padding=padding, groups=hidden_size)[:, :, :seq_len]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


@use_kernelized_func([causal_conv1d_update, causal_conv1d_fn])
class Lfm3ShortConv(nn.Module):
    def __init__(self, config: Lfm3RDConfig, layer_idx: int, shared_core: Lfm3SharedCoreBank | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.conv_kernel_size = config.conv_L_cache
        self.bias = config.conv_bias

        self.conv = nn.Conv1d(
            in_channels=config.hidden_size, out_channels=config.hidden_size, kernel_size=self.conv_kernel_size,
            groups=config.hidden_size, bias=self.bias, padding=self.conv_kernel_size - 1,
        )

        pos = _core_pos(config, layer_idx, shared_core)
        if pos is not None:
            b = 4 if pos == 1 else 10
            self.in_proj = Lfm3SharedGroupedLoRALinear(
                shared_core, b + 0, config.hidden_size, config.hidden_size, config.rank_conv,
                num_groups=config.conv_lora_groups,
            )
            self.out_proj = Lfm3SharedLoRALinear(shared_core, b + 1, config.hidden_size, config.hidden_size, config.rank_conv)
        else:
            self.in_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=self.bias)
            self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=self.bias)

        raw_lt = config.layer_types[layer_idx]
        self.layer_type = raw_lt if raw_lt in config._attention_layer_types else "conv"

    @force_accelerate_hooks("conv")
    def forward(self, hidden_states, past_key_values=None, attention_mask=None, seq_idx=None):
        seq_len = hidden_states.shape[1]

        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        BCx = self.in_proj(hidden_states).transpose(-1, -2)
        B, C, x = BCx.chunk(3, dim=-2)
        hidden_states = B * x
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask[:, None, :].to(hidden_states.dtype)

        has_prev = False
        if past_key_values is not None and hasattr(past_key_values, "layers") and self.layer_idx < len(past_key_values.layers):
            lyr = past_key_values.layers[self.layer_idx]
            has_prev = getattr(lyr, "has_previous_state", {}).get(0, False)

        use_kernels = 2 <= self.conv_kernel_size <= 4
        use_wide_kernel = can_use_lfm3_conv(hidden_states, self.conv.weight)

        if has_prev and seq_len == 1 and not past_key_values.layers[self.layer_idx].record_past:
            conv_state = past_key_values.layers[self.layer_idx].conv_states[0]
            if use_wide_kernel and conv_state.shape[-1] in (6, 7):
                hidden_states = lfm3_causal_conv1d_update(hidden_states, conv_state, self.conv.weight.squeeze(1), self.conv.bias)
            elif use_kernels:
                hidden_states = causal_conv1d_update(hidden_states, conv_state, self.conv.weight.squeeze(1), self.conv.bias)
            else:
                hidden_states = torch.cat([conv_state, hidden_states], dim=-1)
                conv_state.copy_(hidden_states[:, :, -conv_state.shape[-1] :])
                hidden_states = self._conv(hidden_states)[:, :, -seq_len:]
        else:
            if past_key_values is not None and seq_idx is not None:
                last_document = (seq_idx == seq_idx[:, -1:])[:, None].to(hidden_states.dtype)
                past_key_values.update_conv_state(hidden_states * last_document, self.layer_idx,
                                                  conv_kernel_size=self.conv_kernel_size)
            elif past_key_values is not None:
                hidden_states = past_key_values.update_conv_state(hidden_states, self.layer_idx,
                                                                  conv_kernel_size=self.conv_kernel_size)

            if use_wide_kernel:
                hidden_states = lfm3_causal_conv1d(hidden_states, self.conv.weight.squeeze(1), self.conv.bias, seq_idx)
            elif seq_idx is not None:
                hidden_states = self._segmented_conv(hidden_states, seq_idx)
            elif use_kernels:
                hidden_states = causal_conv1d_fn(hidden_states, self.conv.weight.squeeze(1), self.conv.bias)
            else:
                hidden_states = self._conv(hidden_states)

            if past_key_values is not None:
                hidden_states = hidden_states[:, :, -seq_len:]

        y = C * hidden_states
        y = y.transpose(-1, -2).contiguous()
        return self.out_proj(y)

    def update_cache_only(self, normed_hidden_states: torch.Tensor, past_key_values: Cache):
        BCx = self.in_proj(normed_hidden_states).transpose(-1, -2)
        B, _, x = BCx.chunk(3, dim=-2)
        bx = B * x
        past_key_values.update_conv_state(bx, self.layer_idx, conv_kernel_size=self.conv_kernel_size)

    def _conv(self, hidden_states):
        conv_out = self.conv(hidden_states.to(self.conv.weight.dtype))[..., : hidden_states.shape[-1]]
        return conv_out.to(hidden_states.dtype)

    def _segmented_conv(self, hidden_states, seq_idx):
        kernel_size = self.conv_kernel_size
        seq_idx = F.pad(seq_idx, (hidden_states.shape[-1] - seq_idx.shape[-1] + kernel_size - 1, 0), value=-1)
        windows = F.pad(hidden_states, (kernel_size - 1, 0)).unfold(-1, kernel_size, 1)
        seq_idx_windows = seq_idx.unfold(-1, kernel_size, 1)
        same_document = (seq_idx_windows == seq_idx_windows[..., -1:]).unsqueeze(1)
        conv_out = torch.einsum("bdtk,dk->bdt", windows * same_document.to(windows.dtype), self.conv.weight.squeeze(1))
        if self.bias:
            conv_out = conv_out + self.conv.bias[None, :, None]
        return conv_out.to(hidden_states.dtype)


class Lfm3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Lfm3RDConfig, layer_idx: int, shared_core: Lfm3SharedCoreBank | None = None):
        super().__init__()
        self.layer_idx = layer_idx
        raw_lt = config.layer_types[layer_idx]
        self.is_attention_layer = raw_lt in config._attention_layer_types
        self.layer_type = raw_lt if self.is_attention_layer else "conv"
        self.use_sandwich_norm = config.use_sandwich_norm

        if self.is_attention_layer:
            self.self_attn = Lfm3Attention(config, layer_idx, shared_core=shared_core)
        else:
            self.conv = Lfm3ShortConv(config, layer_idx, shared_core=shared_core)
        self.feed_forward = Lfm3MLP(config)
        self.operator_norm = Lfm3RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = Lfm3RMSNorm(config.hidden_size, eps=config.norm_eps)
        if self.use_sandwich_norm:
            self.post_operator_norm = Lfm3RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.post_ffn_norm = Lfm3RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, position_ids=None,
                past_key_values=None, seq_idx=None, **kwargs):
        residual = hidden_states
        if self.is_attention_layer:
            hidden_states, _ = self.self_attn(
                hidden_states=self.operator_norm(hidden_states),
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )
        else:
            hidden_states = self.conv(
                hidden_states=self.operator_norm(hidden_states),
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                seq_idx=seq_idx,
            )
        if self.use_sandwich_norm:
            hidden_states = self.post_operator_norm(hidden_states)
        hidden_states = hidden_states + residual

        ffn_out = self.feed_forward(self.ffn_norm(hidden_states))
        if self.use_sandwich_norm:
            ffn_out = self.post_ffn_norm(ffn_out)
        return hidden_states + ffn_out

    def forward_ffn_only(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_key_values: Cache | None = None,
    ) -> torch.Tensor:
        if past_key_values is not None:
            op_in = self.operator_norm(hidden_states)
            if self.is_attention_layer and position_embeddings is not None:
                self.self_attn.update_cache_only(op_in, position_embeddings, past_key_values)
            elif not self.is_attention_layer:
                self.conv.update_cache_only(op_in, past_key_values)

        normed = F.rms_norm(
            hidden_states,
            self.ffn_norm.weight.shape,
            self.ffn_norm.weight.detach().to(hidden_states.dtype),
            self.ffn_norm.variance_epsilon,
        )
        ffn_out = self.feed_forward.forward_detached_weights(normed)
        if self.use_sandwich_norm:
            ffn_out = F.rms_norm(
                ffn_out,
                self.post_ffn_norm.weight.shape,
                self.post_ffn_norm.weight.detach().to(ffn_out.dtype),
                self.post_ffn_norm.variance_epsilon,
            )
        return hidden_states + ffn_out


def _run_blocks(blocks, hidden_states, masks, pos_emb, position_ids, past_key_values, seq_idx, **kwargs):
    h = hidden_states
    for blk in blocks:
        lt = blk.layer_type
        h = blk(
            h,
            attention_mask=masks.get(lt, masks.get("conv")),
            position_embeddings=pos_emb.get(lt),
            position_ids=position_ids,
            past_key_values=past_key_values,
            seq_idx=seq_idx,
            **kwargs,
        )
    return h


class Lfm3RDPhase(nn.Module):
    def __init__(self, config: Lfm3RDConfig, phase_idx: int, shared_core: Lfm3SharedCoreBank):
        super().__init__()
        self.phase_idx = phase_idx
        base = config.num_prelude_layers + phase_idx * config.motif_size
        self.blocks = nn.ModuleList(
            [Lfm3DecoderLayer(config, base + pos, shared_core=shared_core) for pos in range(config.motif_size)]
        )

    def forward(self, hidden_states, masks, pos_emb, position_ids, past_key_values, seq_idx, **kwargs):
        return _run_blocks(self.blocks, hidden_states, masks, pos_emb, position_ids, past_key_values, seq_idx, **kwargs)

    def forward_ffn_only(
        self,
        hidden_states: torch.Tensor,
        pos_emb: dict | None = None,
        past_key_values: Cache | None = None,
    ) -> torch.Tensor:
        h = hidden_states
        for blk in self.blocks:
            pe = pos_emb.get(blk.layer_type) if pos_emb is not None else None
            h = blk.forward_ffn_only(h, position_embeddings=pe, past_key_values=past_key_values)
        return h


class Lfm3RDCore(nn.Module):
    def __init__(self, config: Lfm3RDConfig):
        super().__init__()
        self.shared = Lfm3SharedCoreBank(config)
        self.phases = nn.ModuleList([Lfm3RDPhase(config, m, self.shared) for m in range(config.num_motifs)])


def create_masks_by_layer_type(config, inputs_embeds, attention_mask, past_key_values, position_ids=None) -> dict:
    config = config.get_text_config()
    mask_kwargs = {
        "config": config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }
    layer_types, mapping = config.layer_types, {}
    if "full_attention" in layer_types:
        mapping["full_attention"] = create_causal_mask(**mask_kwargs, layer_idx=layer_types.index("full_attention"))
    if "sliding_attention" in layer_types:
        mapping["sliding_attention"] = create_sliding_window_causal_mask(
            **mask_kwargs, layer_idx=layer_types.index("sliding_attention")
        )
    if "conv" in layer_types or "linear_attention" in layer_types:
        conv_mask = create_recurrent_attention_mask(**mask_kwargs)
        mapping["conv"] = conv_mask
        mapping["linear_attention"] = conv_mask
    return mapping


def _snapshot_phase_cache(past_key_values, m_idx, config, save_kv_tensors: bool = False):
    if past_key_values is None or not hasattr(past_key_values, "layers"):
        return None
    snap = {}
    base = config.num_prelude_layers + m_idx * config.motif_size
    layers = past_key_values.layers
    for pos in range(config.motif_size):
        l_idx = base + pos
        if l_idx >= len(layers):
            continue
        lyr = layers[l_idx]
        if hasattr(lyr, "is_conv_states_initialized"):
            states_snap = {}
            for s_i in range(getattr(lyr, "number_of_states", 1)):
                is_init = lyr.is_conv_states_initialized.get(s_i, False)
                c_state = lyr.conv_states.get(s_i, None)
                states_snap[s_i] = {
                    "is_conv_states_initialized": is_init,
                    "has_previous_state": lyr.has_previous_state.get(s_i, False),
                    "conv_kernel_size": lyr.conv_kernel_size.get(s_i, None),
                    "conv_state": c_state.clone() if (is_init and c_state is not None) else None,
                }
            snap[l_idx] = ("conv", states_snap, getattr(lyr, "record_past", False))
        else:
            is_init = getattr(lyr, "is_initialized", False)
            keys = getattr(lyr, "keys", None)
            k_t = lyr.keys.clone() if (save_kv_tensors and is_init and getattr(lyr, "keys", None) is not None) else None
            v_t = lyr.values.clone() if (save_kv_tensors and is_init and getattr(lyr, "values", None) is not None) else None
            snap[l_idx] = (
                "attn",
                is_init,
                keys.shape[-2] if (is_init and keys is not None and keys.numel() > 0) else 0,
                getattr(lyr, "cumulative_length", 0),
                k_t,
                v_t,
            )
    return snap


def _restore_phase_cache(past_key_values, snap):
    if past_key_values is None or not snap:
        return
    layers = getattr(past_key_values, "layers", [])
    for l_idx, info in snap.items():
        if l_idx >= len(layers):
            continue
        lyr = layers[l_idx]
        kind = info[0]
        if kind == "conv":
            states_snap, rec_past = info[1], info[2]
            for s_i, sdata in states_snap.items():
                lyr.is_conv_states_initialized[s_i] = sdata["is_conv_states_initialized"]
                lyr.has_previous_state[s_i] = sdata["has_previous_state"]
                lyr.conv_kernel_size[s_i] = sdata["conv_kernel_size"]
                if sdata["conv_state"] is not None:
                    lyr.conv_states[s_i] = sdata["conv_state"].clone()
                else:
                    lyr.conv_states[s_i] = None
            lyr.record_past = rec_past
        elif kind == "attn":
            is_init, k_len, cum_len = info[1], info[2], info[3]
            k_saved = info[4] if len(info) > 4 else None
            v_saved = info[5] if len(info) > 5 else None
            if not is_init or k_len == 0:
                lyr.keys = None
                lyr.values = None
                lyr.is_initialized = False
                if hasattr(lyr, "cumulative_length"):
                    lyr.cumulative_length = 0
            else:
                if k_saved is not None and v_saved is not None:
                    lyr.keys = k_saved.clone()
                    lyr.values = v_saved.clone()
                    lyr.is_initialized = True
                elif getattr(lyr, "keys", None) is not None and lyr.keys.numel() > 0:
                    lyr.keys = lyr.keys[..., :k_len, :]
                    lyr.values = lyr.values[..., :k_len, :]
                if hasattr(lyr, "cumulative_length"):
                    lyr.cumulative_length = cum_len


class Lfm3RDPreTrainedModel(PreTrainedModel):
    config: Lfm3RDConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Lfm3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Lfm3DecoderLayer,
        "attentions": Lfm3Attention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Lfm3RotaryEmbedding):
            for layer_type in module.layer_types:
                curr_inv_freq, _ = module.compute_rope_parameters(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)
        elif isinstance(module, Lfm3RDController):
            init.ones_(module.act_norm.weight)
            init.ones_(module.vel_norm.weight)
            init.normal_(module.state_proj.weight, mean=0.0, std=0.02)
            init.normal_(module.vel_proj.weight, mean=0.0, std=0.02)
            init.normal_(module.dyn_proj.weight, mean=0.0, std=0.02)
            init.zeros_(module.dyn_proj.bias)
            init.normal_(module.halting_head.weight, mean=0.0, std=0.02)
            init.constant_(module.halting_head.bias, -0.85)
            init.ones_(module.step_scales)
            init.ones_(module.inj_norm.weight)
            init.normal_(module.inj_A, mean=0.0, std=1.0 / math.sqrt(module.hidden_size))
            init.normal_(module.inj_B, mean=0.0, std=0.015)
            init.constant_(module.inj_gates, 0.20)
            init.ones_(module.ref_norm.weight)
            init.normal_(module.ref_A, mean=0.0, std=1.0 / math.sqrt(module.hidden_size))
            init.normal_(module.ref_B, mean=0.0, std=0.008)
            init.zeros_(module.refine_gates)
            init.zeros_(module.refine_token_proj.weight)
            init.zeros_(module.refine_token_proj.bias)

    @staticmethod
    def create_masks_for_generate(config, inputs_embeds, attention_mask, past_key_values, position_ids=None, **kwargs):
        return create_masks_by_layer_type(config, inputs_embeds, attention_mask, past_key_values, position_ids)


class Lfm3RDModel(Lfm3RDPreTrainedModel):
    def __init__(self, config: Lfm3RDConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        coda_start = config.num_prelude_layers + config.num_motifs * config.motif_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.prelude = nn.ModuleList([Lfm3DecoderLayer(config, i) for i in range(config.num_prelude_layers)])
        self.core = Lfm3RDCore(config)
        self.coda = nn.ModuleList([Lfm3DecoderLayer(config, i) for i in range(coda_start, config.num_hidden_layers)])
        self.rd_controller = Lfm3RDController(config)
        self.rotary_emb = Lfm3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.embedding_norm = Lfm3RMSNorm(config.hidden_size, eps=config.norm_eps)

        self.last_recurrent_steps: int | None = None
        self.act_step_history: list[int] = []

        self.post_init()

    def set_reasoning_mode(self, mode="manual", loops=None, effort=None, min_loops=None, max_loops=None):
        if effort is not None:
            if effort == "auto":
                mode = "auto"
            elif effort in EFFORT_TO_LOOPS:
                mode, loops = "manual", EFFORT_TO_LOOPS[effort]
            else:
                raise ValueError(f"Unknown `effort`={effort!r}.")
        self.config.reasoning_mode = mode
        self.config.reasoning_effort = effort
        if loops is not None:
            self.config.recurrent_loops = int(loops)
        if min_loops is not None:
            self.config.min_recurrent_loops = int(min_loops)
        if max_loops is not None:
            self.config.max_recurrent_loops = int(max_loops)

    def _complete_ffn_tail(
        self,
        hidden_states: torch.Tensor,
        next_phase_idx: int,
        pos_emb: dict | None = None,
        past_key_values: Cache | None = None,
    ) -> torch.Tensor:
        h = hidden_states
        for m_rem in range(next_phase_idx, self.config.num_motifs):
            h = self.core.phases[m_rem].forward_ffn_only(h, pos_emb=pos_emb, past_key_values=past_key_values)
        return h

    def _probs_to_halt_dist(self, probs: list[torch.Tensor]) -> torch.Tensor:
        p_stack = torch.stack(probs, dim=0)
        T_steps = p_stack.shape[0]
        if T_steps == 1:
            return torch.ones_like(p_stack)
        cdf_prev = torch.cat([torch.zeros_like(p_stack[:1]), p_stack[:-1]], dim=0)
        pmf_raw = (p_stack - cdf_prev).clamp(min=1e-4)
        rem = (1.0 - p_stack[-2:-1]).clamp(min=1e-4)
        pmf = torch.cat([pmf_raw[:-1], rem], dim=0)
        return pmf / pmf.sum(dim=0, keepdim=True).clamp(min=1e-6)

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        reasoning_mode: str | None = None,
        reasoning_effort: str | None = None,
        recurrent_loops: int | None = None,
        min_recurrent_loops: int | None = None,
        max_recurrent_loops: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Lfm3RDModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        eff = reasoning_effort if reasoning_effort is not None else self.config.reasoning_effort
        mode = reasoning_mode if reasoning_mode is not None else self.config.reasoning_mode
        T_loops = recurrent_loops if recurrent_loops is not None else self.config.recurrent_loops
        T_min = min_recurrent_loops if min_recurrent_loops is not None else self.config.min_recurrent_loops
        T_max = max_recurrent_loops if max_recurrent_loops is not None else self.config.max_recurrent_loops
        if eff is not None:
            if eff == "auto":
                mode = "auto"
            elif eff in EFFORT_TO_LOOPS:
                mode, T_loops = "manual", EFFORT_TO_LOOPS[eff]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        first_attn_idx = next((i for i, t in enumerate(self.config.layer_types) if t in self.config._attention_layer_types), 2)
        past_seen_tokens = past_key_values.get_seq_length(first_attn_idx) if past_key_values is not None else 0

        if position_ids is None:
            position_ids = (torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens).unsqueeze(0)

        if not isinstance(masks := attention_mask, dict):
            masks = create_masks_by_layer_type(
                config=self.config, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                past_key_values=past_key_values, position_ids=position_ids,
            )

        hidden_states = inputs_embeds
        pos_emb = {lt: self.rotary_emb(hidden_states, position_ids=position_ids, layer_type=lt)
                   for lt in self.rotary_emb.layer_types}

        def run(h, m, pkv):
            return self.core.phases[m](h, masks, pos_emb, position_ids, pkv, None, **kwargs)

        # === 1. PRELUDE ===
        hidden_states = _run_blocks(self.prelude, hidden_states, masks, pos_emb, position_ids, past_key_values, None, **kwargs)
        prelude_out = hidden_states

        ctrl = self.rd_controller
        n_motifs = self.config.num_motifs
        track = self.training
        probs: list[torch.Tensor] = []
        pred_depths: list[torch.Tensor] = []
        base_t5_hidden = None

        phase_snaps = {m: _snapshot_phase_cache(past_key_values, m, self.config) for m in range(n_motifs)}
        executed_phases = set()
        is_decoding = (hidden_states.shape[1] == 1)

        # === 2. RECURRENT CORE ===
        if mode == "auto" and is_decoding and not self.training:
            T_min_c = max(1, min(32, int(T_min)))
            T_max_c = max(T_min_c, min(32, int(T_max)))
            threshold = float(self.config.act_halting_threshold)
            steps_taken = 0

            h_stream = prelude_out
            h_latent = prelude_out
            h_prev_exit = prelude_out
            p_ready = None
            pass1_snaps = None
            phase_deltas: list[torch.Tensor | None] = [None] * n_motifs

            for s in range(T_max_c):
                m_idx = s % n_motifs
                if m_idx in executed_phases and past_key_values is not None:
                    _restore_phase_cache(past_key_values, phase_snaps[m_idx])
                executed_phases.add(m_idx)

                if s < n_motifs:
                    h_next = run(h_stream, m_idx, past_key_values)
                    phase_deltas[m_idx] = h_next - h_stream
                    h_stream = h_next
                    h_latent = h_stream
                    h_exit_cand = h_stream
                    if s == n_motifs - 1 and past_key_values is not None:
                        pass1_snaps = {m: _snapshot_phase_cache(past_key_values, m, self.config, save_kv_tensors=True) for m in range(n_motifs)}
                else:
                    if m_idx == 0:
                        cycle_idx = s // n_motifs
                        h_stream = ctrl.inject_feedback(prelude_out, h_latent, cycle_idx)
                    h_next = run(h_stream, m_idx, past_key_values)
                    delta_new = h_next - h_stream
                    h_stream = h_next
                    old_delta = phase_deltas[m_idx]
                    alpha_tok, step_corr = ctrl.compute_deep_refinement_step(
                        s, m_idx, h_stream, delta_new, old_delta, h_latent, p_ready
                    )
                    h_latent = h_latent + alpha_tok * step_corr
                    phase_deltas[m_idx] = old_delta + alpha_tok * (delta_new - old_delta)
                    h_exit_cand = h_latent

                steps_taken += 1
                p_ready, _ = ctrl.predict_depth_and_prob(h_exit_cand, h_prev_exit, prelude_out, step_idx=s)
                h_prev_exit = h_exit_cand
                if steps_taken >= T_min_c and bool((p_ready >= threshold).all()):
                    if steps_taken < n_motifs and past_key_values is not None:
                        hidden_states = self._complete_ffn_tail(
                            h_stream, steps_taken, pos_emb=pos_emb, past_key_values=past_key_values
                        )
                    else:
                        hidden_states = h_exit_cand
                    break
            else:
                if steps_taken < n_motifs and past_key_values is not None:
                    hidden_states = self._complete_ffn_tail(
                        h_stream, steps_taken, pos_emb=pos_emb, past_key_values=past_key_values
                    )
                else:
                    hidden_states = h_exit_cand

            if steps_taken > n_motifs and past_key_values is not None and pass1_snaps is not None:
                for m_k in range(n_motifs):
                    _restore_phase_cache(past_key_values, pass1_snaps[m_k])

            self.last_recurrent_steps = steps_taken
            self.act_step_history.append(steps_taken)
            scale_t = ctrl.step_scales[self.last_recurrent_steps - 1].to(hidden_states.dtype)
            hidden_states = prelude_out + scale_t * (hidden_states - prelude_out)
        else:
            T = max(1, min(32, int(T_loops)))
            step_states: list[torch.Tensor] = []

            h_stream = prelude_out
            h_latent = prelude_out
            h_prev_exit = prelude_out
            p_ready = None
            pass1_snaps = None
            phase_deltas: list[torch.Tensor | None] = [None] * n_motifs
            h_exit_s = prelude_out

            for s in range(T):
                m_idx = s % n_motifs
                if m_idx in executed_phases and past_key_values is not None:
                    _restore_phase_cache(past_key_values, phase_snaps[m_idx])
                executed_phases.add(m_idx)

                if s < n_motifs:
                    h_next = run(h_stream, m_idx, past_key_values)
                    phase_deltas[m_idx] = h_next - h_stream
                    h_stream = h_next
                    if s + 1 < n_motifs:
                        is_final_early_step = (s == T - 1)
                        if is_final_early_step or self.training:
                            h_exit_s = self._complete_ffn_tail(
                                h_stream,
                                s + 1,
                                pos_emb=pos_emb if is_final_early_step else None,
                                past_key_values=past_key_values if is_final_early_step else None,
                            )
                        else:
                            h_exit_s = h_stream
                    else:
                        h_latent = h_stream
                        h_exit_s = h_latent
                        base_t5_hidden = h_latent.detach()
                        if past_key_values is not None:
                            pass1_snaps = {m: _snapshot_phase_cache(past_key_values, m, self.config, save_kv_tensors=True) for m in range(n_motifs)}
                else:
                    if m_idx == 0:
                        cycle_idx = s // n_motifs
                        h_stream = ctrl.inject_feedback(prelude_out, h_latent, cycle_idx)
                    h_next = run(h_stream, m_idx, past_key_values)
                    delta_new = h_next - h_stream
                    h_stream = h_next
                    old_delta = phase_deltas[m_idx]
                    alpha_tok, step_corr = ctrl.compute_deep_refinement_step(
                        s, m_idx, h_stream, delta_new, old_delta, h_latent, p_ready
                    )
                    h_latent = h_latent + alpha_tok * step_corr
                    phase_deltas[m_idx] = old_delta + alpha_tok * (delta_new - old_delta)
                    h_exit_s = h_latent

                if track or T > n_motifs:
                    p_ready, s_pred = ctrl.predict_depth_and_prob(
                        h_exit_s.detach(), h_prev_exit.detach(), prelude_out.detach(), step_idx=s
                    )
                h_prev_exit = h_exit_s

                if track:
                    probs.append(p_ready)
                    pred_depths.append(s_pred)
                    if mode == "auto":
                        sc_s = ctrl.step_scales[s].to(h_exit_s.dtype)
                        step_states.append(prelude_out + sc_s * (h_exit_s - prelude_out))

            if T > n_motifs and past_key_values is not None and pass1_snaps is not None:
                for m_k in range(n_motifs):
                    _restore_phase_cache(past_key_values, pass1_snaps[m_k])

            hidden_states = h_exit_s
            self.last_recurrent_steps = T

            if self.training and mode == "auto" and len(step_states) > 1:
                halt_dist = self._probs_to_halt_dist(probs).to(hidden_states.dtype)
                hidden_states = (halt_dist.unsqueeze(-1) * torch.stack(step_states, dim=0)).sum(dim=0)
            else:
                scale_t = ctrl.step_scales[self.last_recurrent_steps - 1].to(hidden_states.dtype)
                hidden_states = prelude_out + scale_t * (hidden_states - prelude_out)

        base_t5_coda_hidden = None
        if self.training and base_t5_hidden is not None and self.last_recurrent_steps > n_motifs and past_key_values is None:
            with torch.no_grad():
                sc_5 = ctrl.step_scales[n_motifs - 1].detach().to(base_t5_hidden.dtype)
                h_t5_scaled = prelude_out.detach() + sc_5 * (base_t5_hidden - prelude_out.detach())
                h_t5_coda = _run_blocks(self.coda, h_t5_scaled, masks, pos_emb, position_ids, None, None, **kwargs)
                base_t5_coda_hidden = self.embedding_norm(h_t5_coda).detach()

        # === 3. CODA ===
        hidden_states = _run_blocks(self.coda, hidden_states, masks, pos_emb, position_ids, past_key_values, None, **kwargs)
        hidden_states = self.embedding_norm(hidden_states)

        return Lfm3RDModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            ponder_loss=None,
            recurrent_steps=self.last_recurrent_steps,
            step_halting_probs=probs if self.training else None,
            step_pred_depths=pred_depths if self.training else None,
            base_t5_coda_hidden=base_t5_coda_hidden,
        )


class Lfm3RDForCausalLM(Lfm3RDPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Lfm3RDConfig):
        super().__init__(config)
        self.model = Lfm3RDModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def set_reasoning_mode(self, mode="manual", loops=None, effort=None, min_loops=None, max_loops=None):
        self.model.set_reasoning_mode(mode=mode, loops=loops, effort=effort, min_loops=min_loops, max_loops=max_loops)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None,
                                      inputs_embeds=None, cache_position=None, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask,
            inputs_embeds=inputs_embeds, cache_position=cache_position, **kwargs,
        )
        for k in ("reasoning_mode", "reasoning_effort", "recurrent_loops", "min_recurrent_loops", "max_recurrent_loops"):
            if k in kwargs:
                model_inputs[k] = kwargs[k]
        return model_inputs

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        reasoning_mode: str | None = None,
        reasoning_effort: str | None = None,
        recurrent_loops: int | None = None,
        min_recurrent_loops: int | None = None,
        max_recurrent_loops: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Lfm3RDCausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            reasoning_mode=reasoning_mode,
            reasoning_effort=reasoning_effort,
            recurrent_loops=recurrent_loops,
            min_recurrent_loops=min_recurrent_loops,
            max_recurrent_loops=max_recurrent_loops,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        ponder_loss = None
        deep_gain_loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

            shift_logits_live = logits[:, :-1, :].float()
            shift_labels = labels[:, 1:]
            valid_mask = (shift_labels != -100)

            with torch.no_grad():
                shift_logits_det = shift_logits_live.detach()
                tok_nll_out = F.cross_entropy(
                    shift_logits_det.reshape(-1, shift_logits_det.size(-1)),
                    shift_labels.reshape(-1),
                    reduction="none",
                    ignore_index=-100,
                ).view_as(shift_labels)

                ref_t5_nll = None
                if outputs.base_t5_coda_hidden is not None:
                    t5_logits = self.lm_head(outputs.base_t5_coda_hidden[:, slice_indices, :])[:, :-1, :].float()
                    ref_t5_nll = F.cross_entropy(
                        t5_logits.reshape(-1, t5_logits.size(-1)),
                        shift_labels.reshape(-1),
                        reduction="none",
                        ignore_index=-100,
                    ).view_as(shift_labels)

            if self.training and ref_t5_nll is not None and valid_mask.any():
                live_tok_nll = F.cross_entropy(
                    shift_logits_live.reshape(-1, shift_logits_live.size(-1)),
                    shift_labels.reshape(-1),
                    reduction="none",
                    ignore_index=-100,
                ).view_as(shift_labels)
                valid_f = valid_mask.float()
                denom_v = valid_f.sum().clamp(min=1.0)
                med_nll = ref_t5_nll[valid_mask].median().detach()

                hard_mask = valid_mask & (ref_t5_nll > med_nll)
                easy_mask = valid_mask & (ref_t5_nll <= med_nll)

                hard_pen = (F.relu(live_tok_nll - ref_t5_nll + 0.06) * hard_mask.float()).sum() / denom_v
                easy_pen = (F.relu(live_tok_nll - ref_t5_nll) * easy_mask.float()).sum() / denom_v
                deep_gain_loss = 1.5 * hard_pen + 2.0 * easy_pen

            if self.training and outputs.step_halting_probs and outputs.step_pred_depths and valid_mask.any():
                num_steps = len(outputs.step_halting_probs)
                with torch.no_grad():
                    p_top1 = F.softmax(shift_logits_det, dim=-1).max(dim=-1).values
                    top1_uncert = (1.0 - p_top1).clamp(0.0, 1.0)

                    if ref_t5_nll is not None:
                        empirical_gain = F.relu(ref_t5_nll - tok_nll_out)
                    else:
                        empirical_gain = torch.zeros_like(tok_nll_out)

                    cog_score = tok_nll_out + 0.80 * top1_uncert + 4.0 * empirical_gain
                    abs_u = (cog_score / 2.80).clamp(0.0, 1.0)

                    valid_scores = cog_score[valid_mask]
                    if valid_scores.numel() > 1:
                        sorted_idx = torch.argsort(torch.argsort(valid_scores))
                        ranks_flat = sorted_idx.float() / float(valid_scores.numel() - 1)
                        ranks = torch.zeros_like(cog_score)
                        ranks[valid_mask] = ranks_flat
                    else:
                        ranks = abs_u

                    u = (0.80 * abs_u + 0.20 * ranks * (abs_u * 2.5).clamp(0.0, 1.0)).clamp(0.0, 1.0)
                    target_s_cont = 31.0 * (u ** 1.35)

                denom_m = valid_mask.float().sum().clamp(min=1.0)
                reg_depth_loss = 0.0
                bce_halt_loss = 0.0

                for s_i in range(num_steps):
                    s_pred_shift = outputs.step_pred_depths[s_i][:, :-1].float()
                    p_s_shift = outputs.step_halting_probs[s_i][:, :-1].float().clamp(1e-5, 1.0 - 1e-5)

                    huber_s = F.smooth_l1_loss(s_pred_shift / 31.0, target_s_cont / 31.0, reduction="none", beta=0.04)
                    reg_depth_loss = reg_depth_loss + (huber_s * valid_mask.float()).sum() / denom_m

                    target_halt = torch.sigmoid(2.5 * (float(s_i) - target_s_cont + 1.15))
                    bce_s = -(target_halt * torch.log(p_s_shift) + (1.0 - target_halt) * torch.log(1.0 - p_s_shift))
                    bce_halt_loss = bce_halt_loss + (bce_s * valid_mask.float()).sum() / denom_m

                ponder_loss = (4.0 * reg_depth_loss + 0.5 * bce_halt_loss) / num_steps

        return Lfm3RDCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            ponder_loss=ponder_loss,
            recurrent_steps=outputs.recurrent_steps,
            deep_gain_loss=deep_gain_loss,
        )


__all__ = ["Lfm3RDConfig", "Lfm3RDForCausalLM", "Lfm3RDModel", "Lfm3RDPreTrainedModel", "Lfm3RDCore", "Lfm3RDPhase",
           "Lfm3RDModelOutputWithPast", "Lfm3RDCausalLMOutputWithPast"]