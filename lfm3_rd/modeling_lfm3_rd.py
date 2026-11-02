"""
Canonical Recurrent-Depth LFM3 (Prelude-Core-Coda) Architecture.
"""

from dataclasses import dataclass
import sys

sys.modules.pop("torchvision", None)
sys.modules.pop("torchvision.transforms", None)

import transformers.utils as _tu
import transformers.utils.import_utils as _tiu
_tu.import_utils = _tiu
_tu.is_torchvision_available = lambda: False
_tiu.is_torchvision_available = lambda: False

import torch
import torch.nn.functional as F
from torch import nn

from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .configuration_lfm3_rd import Lfm3RDConfig


EFFORT_TO_LOOPS = {"low": 1, "fast": 2, "medium": 5, "high": 12, "max": 32}


@dataclass
class Lfm3RDModelOutputWithPast(BaseModelOutputWithPast):
    """
    Base model output for LFM3-RD.
    """

    recurrent_steps: int | None = None
    step_halting_probs: list[torch.Tensor] | None = None
    step_pred_depths: list[torch.Tensor] | None = None
    last_core_state: torch.Tensor | None = None


@dataclass
class Lfm3RDCausalLMOutputWithPast(CausalLMOutputWithPast):
    """
    Causal LM output for LFM3-RD.
    """

    recurrent_steps: int | None = None
    step_halting_probs: list[torch.Tensor] | None = None
    step_pred_depths: list[torch.Tensor] | None = None


class Lfm3RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dimensions of the input tensor.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    Applies Rotary Position Embedding to query and key tensors in float32.
    """
    cos = cos.unsqueeze(unsqueeze_dim).float()
    sin = sin.unsqueeze(unsqueeze_dim).float()
    q_f = q.float()
    k_f = k.float()
    q_embed = (q_f * cos) + (rotate_half(q_f) * sin)
    k_embed = (k_f * cos) + (rotate_half(k_f) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def apply_rotary_pos_emb_key_only(k, cos, sin, unsqueeze_dim=1):
    """
    Applies Rotary Position Embedding to key tensor only.
    """
    cos = cos.unsqueeze(unsqueeze_dim).float()
    sin = sin.unsqueeze(unsqueeze_dim).float()
    k_f = k.float()
    k_embed = (k_f * cos) + (rotate_half(k_f) * sin)
    return k_embed.to(k.dtype)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats heads along head dimension for grouped-query attention.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Lfm3RotaryEmbedding(nn.Module):
    """
    Rotary embedding per attention layer type with zero-tail partial rotation.
    """

    def __init__(self, config: Lfm3RDConfig, device=None):
        super().__init__()
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.layer_types = sorted(set(config.layer_types) & set(getattr(config, "rope_parameters", {})))
        self.rope_type = {}
        for layer_type in self.layer_types:
            rope_params = self.config.rope_parameters[layer_type]
            self.rope_type[layer_type] = rope_params.get("rope_type", "default")
            curr_inv_freq, curr_attention_scaling = self.compute_rope_parameters(
                config, device, layer_type=layer_type
            )
            setattr(
                self,
                f"{layer_type}_inv_freq",
                nn.Buffer(curr_inv_freq, persistent=False),
            )
            setattr(
                self,
                f"{layer_type}_original_inv_freq",
                nn.Buffer(curr_inv_freq.clone(), persistent=False),
            )
            setattr(
                self,
                f"{layer_type}_attention_scaling",
                curr_attention_scaling,
            )

    def compute_rope_parameters(
        self, config: Lfm3RDConfig, device=None, layer_type: str | None = None
    ) -> tuple[torch.Tensor, float]:
        rope_params = config.rope_parameters[layer_type]
        base = rope_params.get("rope_theta", 1000000.0)
        partial_rotary_factor = rope_params.get("partial_rotary_factor", 1.0)
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        rope_angles = int(partial_rotary_factor * dim) // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float) / dim))
        inv_freq = torch.cat((inv_freq, torch.zeros(dim // 2 - rope_angles, dtype=torch.float)))
        return inv_freq.to(device), 1.0

    @torch.no_grad()
    def forward(self, x, position_ids, layer_type):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        if inv_freq.abs().sum() == 0:
            curr_inv_freq, _ = self.compute_rope_parameters(
                self.config, device=x.device, layer_type=layer_type
            )
            inv_freq.copy_(curr_inv_freq)
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")

        freqs = position_ids[..., None].float() * inv_freq.to(device=x.device, dtype=torch.float)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * attention_scaling
        sin = emb.sin() * attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Lfm3ShortConv(nn.Module):
    """
    LFM3 Gated Short Convolution with kernel size 7 and depth-indexed cache bank.
    """

    def __init__(self, config: Lfm3RDConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.conv_kernel_size = config.conv_L_cache
        self.hidden_size = config.hidden_size
        self.bias = config.conv_bias

        self.conv = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=self.conv_kernel_size,
            padding=self.conv_kernel_size - 1,
            groups=self.hidden_size,
            bias=self.bias,
        )
        self.in_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=self.bias)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=self.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        core_depth_idx: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        mask_slice = None
        if attention_mask is not None and attention_mask.dim() == 2:
            mask_slice = attention_mask[:, -seq_len:].to(hidden_states.dtype)
            hidden_states = hidden_states * mask_slice[:, :, None]

        bcx = self.in_proj(hidden_states).transpose(-1, -2)
        b, c, x = bcx.chunk(3, dim=-2)
        bx = b * x
        if mask_slice is not None:
            bx = bx * mask_slice[:, None, :]

        if past_key_values is None:
            conv_out = self.conv(bx)[..., :seq_len]
            y = c * conv_out
            return self.out_proj(y.transpose(-1, -2).contiguous())

        cache_dict = getattr(past_key_values, "_conv_state_bank", None)
        if cache_dict is None:
            cache_dict = {}
            setattr(past_key_values, "_conv_state_bank", cache_dict)

        seen_dict = getattr(past_key_values, "_conv_seen_tokens", None)
        if seen_dict is None:
            seen_dict = {}
            setattr(past_key_values, "_conv_seen_tokens", seen_dict)

        prev_dict = getattr(past_key_values, "_conv_prev_bank", None)
        if prev_dict is None:
            prev_dict = {}
            setattr(past_key_values, "_conv_prev_bank", prev_dict)

        cache_key = (self.layer_idx, core_depth_idx)
        state_len = self.conv_kernel_size - 1

        if cache_key not in cache_dict:
            if core_depth_idx > 0 and (self.layer_idx, core_depth_idx - 1) in prev_dict:
                conv_state = prev_dict[(self.layer_idx, core_depth_idx - 1)].clone()
                seen_tokens = seen_dict.get((self.layer_idx, core_depth_idx - 1), seq_len) - seq_len
            else:
                conv_state = torch.zeros(
                    batch_size,
                    self.hidden_size,
                    state_len,
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                seen_tokens = 0
        else:
            conv_state = cache_dict[cache_key]
            seen_tokens = seen_dict.get(cache_key, 0)
            if core_depth_idx > 0 and (self.layer_idx, 0) in seen_dict:
                expected_prev = seen_dict[(self.layer_idx, 0)] - seq_len
                gap = expected_prev - seen_tokens
                if gap > 0 and (self.layer_idx, core_depth_idx - 1) in prev_dict:
                    ref_prev = prev_dict[(self.layer_idx, core_depth_idx - 1)]
                    if gap >= state_len:
                        conv_state = ref_prev.clone()
                    else:
                        conv_state = torch.cat([conv_state[:, :, gap:], ref_prev[:, :, -gap:]], dim=-1)
                    seen_tokens = expected_prev

        prev_dict[cache_key] = conv_state.detach()
        x_with_cache = torch.cat([conv_state, bx], dim=-1)
        cache_dict[cache_key] = x_with_cache[:, :, -state_len:].detach()
        seen_dict[cache_key] = seen_tokens + seq_len

        conv_out = F.conv1d(
            x_with_cache,
            self.conv.weight,
            self.conv.bias,
            padding=0,
            groups=self.hidden_size,
        )
        y = c * conv_out
        return self.out_proj(y.transpose(-1, -2).contiguous())


class Lfm3Attention(nn.Module):
    """
    Standard Single-Stream Attention used in Coda layer.
    """

    def __init__(self, config: Lfm3RDConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
            self.num_attention_heads // self.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5

        q_dim = self.num_attention_heads * self.head_dim
        kv_dim = self.num_key_value_heads * self.head_dim

        self.q_proj = nn.Linear(config.hidden_size, q_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, config.hidden_size, bias=False)
        self.gate_proj = nn.Linear(
            config.hidden_size, self.num_attention_heads, bias=False
        )
        self.q_norm = Lfm3RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_norm = Lfm3RMSNorm(self.head_dim, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> torch.Tensor:
        b, s, _ = hidden_states.shape
        q = self.q_norm(
            self.q_proj(hidden_states).view(b, s, self.num_attention_heads, self.head_dim)
        ).transpose(1, 2)
        k = self.k_norm(
            self.k_proj(hidden_states).view(b, s, self.num_key_value_heads, self.head_dim)
        ).transpose(1, 2)
        v = self.v_proj(hidden_states).view(
            b, s, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
            setattr(past_key_values, "_lfm3_rd_seq_len", k.shape[-2])
            setattr(past_key_values, "_lfm3_rd_coda_attn_idx", self.layer_idx)

        k_rep = repeat_kv(k, self.num_key_value_groups)
        v_rep = repeat_kv(v, self.num_key_value_groups)

        attn_weights = torch.matmul(q, k_rep.transpose(2, 3)) * self.scaling
        s_q = q.shape[2]
        s_k = k_rep.shape[2]
        min_val = torch.finfo(q.dtype).min / 2

        if s_q > 1:
            causal_mask = torch.triu(
                torch.ones((s_q, s_k), dtype=torch.bool, device=q.device),
                diagonal=s_k - s_q + 1,
            )
            attn_weights = attn_weights.masked_fill(causal_mask[None, None, :, :], min_val)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                pad_u = attention_mask[:, -s_k:] if attention_mask.shape[-1] >= s_k else F.pad(attention_mask, (s_k - attention_mask.shape[-1], 0), value=1)
                pad_mask = (pad_u == 0).unsqueeze(1).unsqueeze(2)
                attn_weights = attn_weights.masked_fill(pad_mask, min_val)
            elif attention_mask.dim() == 4:
                attn_weights = (attn_weights + attention_mask).clamp(min=min_val)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v_rep).transpose(1, 2).contiguous()
        gate = torch.sigmoid(self.gate_proj(hidden_states)).unsqueeze(-1)
        attn_output = (attn_output * gate).reshape(b, s, -1).contiguous()
        return self.o_proj(attn_output)


class Lfm3ContextAnchoredAttention(nn.Module):
    """
    Context-Anchored Attention with dual key-value projections over [h_t; e].
    """

    def __init__(self, config: Lfm3RDConfig, layer_idx: int, is_sliding: bool = False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_sliding = is_sliding
        self.sliding_window = config.sliding_window if is_sliding else None

        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
            self.num_attention_heads // self.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5

        q_dim = self.num_attention_heads * self.head_dim
        kv_dim = self.num_key_value_heads * self.head_dim

        self.q_proj = nn.Linear(config.hidden_size, q_dim, bias=False)
        self.k_proj_h = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.v_proj_h = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.k_proj_e = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.v_proj_e = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, config.hidden_size, bias=False)
        self.gate_proj = nn.Linear(
            config.hidden_size, self.num_attention_heads, bias=False
        )

        self.q_norm = Lfm3RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_norm = Lfm3RMSNorm(self.head_dim, eps=config.norm_eps)

    def compute_anchor_kv(
        self,
        normed_anchor: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, s, _ = normed_anchor.shape
        ke = self.k_norm(
            self.k_proj_e(normed_anchor).view(b, s, self.num_key_value_heads, self.head_dim)
        ).transpose(1, 2)
        ve = self.v_proj_e(normed_anchor).view(
            b, s, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            ke = apply_rotary_pos_emb_key_only(ke, cos, sin)
        return ke, ve

    def forward(
        self,
        hidden_states: torch.Tensor,
        anchor_states: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        core_depth_idx: int = 0,
        anchor_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        b, s, _ = hidden_states.shape
        if anchor_states is None:
            anchor_states = hidden_states

        q = self.q_norm(
            self.q_proj(hidden_states).view(b, s, self.num_attention_heads, self.head_dim)
        ).transpose(1, 2)

        kh = self.k_norm(
            self.k_proj_h(hidden_states).view(b, s, self.num_key_value_heads, self.head_dim)
        ).transpose(1, 2)
        vh = self.v_proj_h(hidden_states).view(
            b, s, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, kh = apply_rotary_pos_emb(q, kh, cos, sin)

        if anchor_kv is not None:
            ke, ve = anchor_kv
        else:
            ke, ve = self.compute_anchor_kv(anchor_states, position_embeddings)

        if past_key_values is not None:
            ca_bank = getattr(past_key_values, "_ca_kv_bank", None)
            if ca_bank is None:
                ca_bank = {}
                setattr(past_key_values, "_ca_kv_bank", ca_bank)

            anchor_bank = getattr(past_key_values, "_ca_anchor_bank", None)
            if anchor_bank is None:
                anchor_bank = {}
                setattr(past_key_values, "_ca_anchor_bank", anchor_bank)

            if core_depth_idx == 0:
                if self.layer_idx in anchor_bank:
                    prev_ke, prev_ve = anchor_bank[self.layer_idx]
                    ke_full = torch.cat([prev_ke, ke], dim=2)
                    ve_full = torch.cat([prev_ve, ve], dim=2)
                else:
                    ke_full, ve_full = ke, ve
                anchor_bank[self.layer_idx] = (ke_full.detach(), ve_full.detach())
            else:
                ke_full, ve_full = anchor_bank[self.layer_idx]

            ke, ve = ke_full, ve_full

            cache_key = (self.layer_idx, core_depth_idx)
            base_key = (self.layer_idx, 0)
            expected_prev_len = (
                ca_bank[base_key][0].shape[2] - s
                if (core_depth_idx > 0 and base_key in ca_bank)
                else 0
            )

            if cache_key not in ca_bank:
                if core_depth_idx > 0 and expected_prev_len > 0:
                    prev_kh, prev_vh = ca_bank[(self.layer_idx, core_depth_idx - 1)][:2]
                    kh = torch.cat([prev_kh[:, :, :expected_prev_len, :], kh], dim=2)
                    vh = torch.cat([prev_vh[:, :, :expected_prev_len, :], vh], dim=2)
            else:
                prev_kh, prev_vh = ca_bank[cache_key][:2]
                if core_depth_idx > 0 and prev_kh.shape[2] < expected_prev_len:
                    ref_kh, ref_vh = ca_bank[(self.layer_idx, core_depth_idx - 1)][:2]
                    cur_len = prev_kh.shape[2]
                    prev_kh = torch.cat([prev_kh, ref_kh[:, :, cur_len:expected_prev_len, :]], dim=2)
                    prev_vh = torch.cat([prev_vh, ref_vh[:, :, cur_len:expected_prev_len, :]], dim=2)
                kh = torch.cat([prev_kh, kh], dim=2)
                vh = torch.cat([prev_vh, vh], dim=2)

            ca_bank[cache_key] = (kh.detach(), vh.detach(), ke.detach(), ve.detach())

        s_q = q.shape[2]
        s_k = kh.shape[2]

        k_joint = torch.cat([kh, ke], dim=2)
        v_joint = torch.cat([vh, ve], dim=2)

        k_rep = repeat_kv(k_joint, self.num_key_value_groups)
        v_rep = repeat_kv(v_joint, self.num_key_value_groups)

        attn_weights = torch.matmul(q, k_rep.transpose(2, 3)) * self.scaling

        mask_matrix = self._build_block_causal_mask(
            s_q=s_q,
            s_k=s_k,
            dtype=q.dtype,
            device=q.device,
            user_mask=attention_mask,
        )
        min_val = torch.finfo(q.dtype).min / 2
        attn_weights = (attn_weights + mask_matrix).clamp(min=min_val)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v_rep).transpose(1, 2).contiguous()
        gate = torch.sigmoid(self.gate_proj(hidden_states)).unsqueeze(-1)
        attn_output = (attn_output * gate).reshape(b, s, -1).contiguous()
        return self.o_proj(attn_output)

    def _build_block_causal_mask(
        self,
        s_q: int,
        s_k: int,
        dtype: torch.dtype,
        device: torch.device,
        user_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        min_val = torch.finfo(dtype).min / 2
        mask = torch.full((s_q, 2 * s_k), min_val, dtype=dtype, device=device)

        q_idx = torch.arange(s_k - s_q, s_k, device=device).unsqueeze(1)
        kh_idx = torch.arange(s_k, device=device).unsqueeze(0)
        ke_idx = torch.arange(s_k, device=device).unsqueeze(0)

        causal_h = kh_idx <= q_idx
        causal_e = ke_idx <= q_idx
        if self.sliding_window is not None:
            causal_h = causal_h & ((q_idx - kh_idx) < self.sliding_window)
            causal_e = causal_e & ((q_idx - ke_idx) < self.sliding_window)

        zero_t = torch.tensor(0.0, dtype=dtype, device=device)
        anchor_prior_t = torch.tensor(-1.6094379, dtype=dtype, device=device)
        mask[:, :s_k] = torch.where(causal_h, zero_t, min_val)
        mask[:, s_k:] = torch.where(causal_e, anchor_prior_t, min_val)
        mask = mask.unsqueeze(0).unsqueeze(0)

        if user_mask is not None:
            if user_mask.dim() == 4:
                um_h = user_mask[..., -s_q:, :s_k]
                um_e = user_mask[..., -s_q:, :s_k]
                user_mask_joint = torch.cat([um_h, um_e], dim=-1)
                mask = (mask + user_mask_joint).clamp(min=min_val)
            elif user_mask.dim() == 2:
                pad_u = (
                    user_mask[:, -s_k:]
                    if user_mask.shape[-1] >= s_k
                    else F.pad(user_mask, (s_k - user_mask.shape[-1], 0), value=1)
                )
                pad_h = (pad_u == 0).unsqueeze(1).unsqueeze(2)
                pad_joint = torch.cat([pad_h, pad_h], dim=-1)
                mask = mask.expand(user_mask.shape[0], -1, -1, -1).clone()
                mask = mask.masked_fill(pad_joint, min_val)

        return mask


class Lfm3MLP(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    """

    def __init__(self, config: Lfm3RDConfig):
        super().__init__()
        self.config = config
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Lfm3DecoderLayer(nn.Module):
    """
    Individual Transformer Layer with Sandwich Norms.
    """

    def __init__(self, config: Lfm3RDConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.use_sandwich_norm = config.use_sandwich_norm

        if self.layer_type == "conv":
            self.conv = Lfm3ShortConv(config, layer_idx)
        elif self.layer_type == "full_attention":
            if 2 <= layer_idx <= 5:
                self.self_attn = Lfm3ContextAnchoredAttention(config, layer_idx, is_sliding=False)
            else:
                self.self_attn = Lfm3Attention(config, layer_idx)
        elif self.layer_type == "sliding_attention":
            self.self_attn = Lfm3ContextAnchoredAttention(config, layer_idx, is_sliding=True)
        else:
            self.conv = Lfm3ShortConv(config, layer_idx)

        self.operator_norm = Lfm3RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = Lfm3RMSNorm(config.hidden_size, eps=config.norm_eps)
        if self.use_sandwich_norm:
            self.post_operator_norm = Lfm3RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.post_ffn_norm = Lfm3RMSNorm(config.hidden_size, eps=config.norm_eps)

        self.feed_forward = Lfm3MLP(config)

    @property
    def mixer(self):
        return self.self_attn if hasattr(self, "self_attn") else self.conv

    def forward(
        self,
        hidden_states: torch.Tensor,
        anchor_states: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        core_depth_idx: int = 0,
        anchor_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        normed = self.operator_norm(hidden_states)

        if hasattr(self, "self_attn"):
            if isinstance(self.self_attn, Lfm3ContextAnchoredAttention):
                normed_anchor = (
                    self.operator_norm(anchor_states)
                    if (anchor_states is not None and anchor_kv is None)
                    else normed
                )
                mixer_out = self.self_attn(
                    hidden_states=normed,
                    anchor_states=normed_anchor,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    core_depth_idx=core_depth_idx,
                    anchor_kv=anchor_kv,
                    **kwargs,
                )
            else:
                mixer_out = self.self_attn(
                    hidden_states=normed,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    **kwargs,
                )
        else:
            mixer_out = self.conv(
                hidden_states=normed,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                core_depth_idx=core_depth_idx,
                **kwargs,
            )

        if self.use_sandwich_norm:
            mixer_out = self.post_operator_norm(mixer_out)
        hidden_states = residual + mixer_out

        ffn_res = hidden_states
        ffn_out = self.feed_forward(self.ffn_norm(hidden_states))
        if self.use_sandwich_norm:
            ffn_out = self.post_ffn_norm(ffn_out)
        return ffn_res + ffn_out


class Lfm3RDController(nn.Module):
    """
    Kinematic Adaptive Halting and Dynamic Depth Controller.
    """

    def __init__(self, config: Lfm3RDConfig):
        super().__init__()
        dim = config.hidden_size
        router_dim = getattr(config, "router_dim", 128)

        self.act_norm = Lfm3RMSNorm(dim, eps=config.norm_eps)
        self.vel_norm = Lfm3RMSNorm(dim, eps=config.norm_eps)

        self.state_proj = nn.Linear(dim, router_dim, bias=False)
        self.vel_proj = nn.Linear(dim, router_dim, bias=False)
        self.dyn_proj = nn.Linear(4, router_dim, bias=True)
        self.halting_head = nn.Linear(router_dim, 1, bias=True)
        nn.init.constant_(self.halting_head.bias, -1.91)

        step_scales_init = [1.0 / (1.0 + 0.5 * float(s)) for s in range(32)]
        self.step_scales = nn.Parameter(torch.tensor(step_scales_init, dtype=torch.float32))

    def predict_depth_and_prob(
        self,
        h_curr: torch.Tensor,
        h_prev: torch.Tensor,
        prelude_out: torch.Tensor,
        step_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w_dt = self.state_proj.weight.dtype
        hc_f = h_curr.float()
        hp_f = h_prev.float()
        pre_f = prelude_out.float()

        delta = hc_f - hp_f
        norm_c = hc_f.norm(dim=-1, keepdim=True).clamp(min=1e-5)
        norm_p = hp_f.norm(dim=-1, keepdim=True).clamp(min=1e-5)
        norm_pre = pre_f.norm(dim=-1, keepdim=True).clamp(min=1e-5)

        rel_vel = (delta.norm(dim=-1, keepdim=True) / norm_c).clamp(0.0, 5.0)
        rel_drift = ((hc_f - pre_f).norm(dim=-1, keepdim=True) / norm_pre).clamp(0.0, 10.0)
        cos_sim = ((hc_f * hp_f).sum(dim=-1, keepdim=True) / (norm_c * norm_p)).clamp(-1.0, 1.0)
        cos_instab = 1.0 - cos_sim
        depth_coord = torch.full_like(rel_vel, float(step_idx) / 31.0)

        dyn_feats = torch.cat([rel_vel, rel_drift, cos_instab, depth_coord], dim=-1).to(w_dt)
        h_normed = self.act_norm(h_curr).to(w_dt)
        v_normed = self.vel_norm(delta.to(h_curr.dtype)).to(w_dt)

        hidden_router = F.silu(
            self.state_proj(h_normed) + self.vel_proj(v_normed) + self.dyn_proj(dyn_feats)
        )
        raw_logit = self.halting_head(hidden_router).squeeze(-1).float()
        s_pred = 31.0 * torch.sigmoid(raw_logit)
        p_ready = torch.sigmoid(2.0 * (float(step_idx) - s_pred + 0.70))
        return p_ready, s_pred


class Lfm3RDPreTrainedModel(PreTrainedModel):
    """
    Abstract base class for LFM3-RD models.
    """

    config_class = Lfm3RDConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Lfm3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Lfm3RotaryEmbedding):
            for layer_type in module.layer_types:
                curr_inv_freq, _ = module.compute_rope_parameters(
                    module.config, layer_type=layer_type
                )
                getattr(module, f"{layer_type}_inv_freq").copy_(curr_inv_freq)
                getattr(module, f"{layer_type}_original_inv_freq").copy_(curr_inv_freq)
        elif isinstance(module, Lfm3RDController):
            nn.init.constant_(module.halting_head.bias, -1.91)
            step_scales_init = torch.tensor(
                [1.0 / (1.0 + 0.5 * float(s)) for s in range(32)],
                dtype=module.step_scales.dtype,
                device=module.step_scales.device,
            )
            module.step_scales.copy_(step_scales_init)


class Lfm3RDModel(Lfm3RDPreTrainedModel):
    """
    Canonical Recurrent-Depth Transformer Backbone.
    """

    def __init__(self, config: Lfm3RDConfig):
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False
        pad_idx = config.pad_token_id if (config.pad_token_id is not None and config.pad_token_id < config.vocab_size) else None
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, pad_idx)
        self.layers = nn.ModuleList([
            Lfm3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])
        self.embedding_norm = Lfm3RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.rotary_emb = Lfm3RotaryEmbedding(config)

        self.step_embeddings = nn.Parameter(torch.zeros(config.max_recurrent_loops, config.hidden_size))
        self.gate_h = nn.Parameter(torch.ones(config.hidden_size))
        self.gate_e = nn.Parameter(torch.zeros(config.hidden_size))

        self.rd_controller = Lfm3RDController(config)
        self.act_step_history = []
        self.post_init()

    @property
    def norm(self):
        return self.embedding_norm

    @staticmethod
    def _ensure_cache_patched(past_key_values: Cache | None):
        if past_key_values is not None and not getattr(past_key_values, "_lfm3_rd_patched", False):
            orig_get_seq_length = past_key_values.get_seq_length

            def _patched_get_seq_length(layer_idx: int | None = 0) -> int:
                coda_idx = getattr(past_key_values, "_lfm3_rd_coda_attn_idx", None)
                if coda_idx is not None and layer_idx in (0, None, coda_idx):
                    try:
                        val = orig_get_seq_length(coda_idx)
                        if val > 0:
                            return val
                    except Exception:
                        pass
                seq_len = getattr(past_key_values, "_lfm3_rd_seq_len", None)
                if seq_len is not None:
                    return seq_len
                return orig_get_seq_length(layer_idx) if layer_idx is not None else orig_get_seq_length()

            past_key_values.get_seq_length = _patched_get_seq_length
            setattr(past_key_values, "_lfm3_rd_patched", True)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        recurrent_loops: int | None = None,
        reasoning_effort: str | None = None,
        **kwargs,
    ) -> Lfm3RDModelOutputWithPast:
        if self.gradient_checkpointing and self.training:
            use_cache = False
        else:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        if use_cache and past_key_values is None and not self.training:
            past_key_values = DynamicCache()
        self._ensure_cache_patched(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        b, s, _ = inputs_embeds.shape
        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = (
                torch.arange(past_seen, past_seen + s, dtype=torch.long, device=inputs_embeds.device)
                .unsqueeze(0)
                .expand(b, -1)
            )
        elif position_ids.shape[-1] > s:
            position_ids = position_ids[:, -s:]

        pos_emb_full = self.rotary_emb(inputs_embeds, position_ids, layer_type="full_attention")
        pos_emb_sliding = self.rotary_emb(inputs_embeds, position_ids, layer_type="sliding_attention")

        hidden_states = inputs_embeds
        if self.gradient_checkpointing and self.training and not hidden_states.requires_grad:
            hidden_states = hidden_states.requires_grad_(True)

        for idx in range(self.config.num_prelude_layers):
            layer = self.layers[idx]
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    None,
                    pos_emb_full,
                    attention_mask,
                    None,
                    0,
                    None,
                )
            else:
                hidden_states = layer(
                    hidden_states=hidden_states,
                    position_embeddings=pos_emb_full,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    **kwargs,
                )

        anchor = hidden_states

        is_auto = (
            reasoning_effort == "auto"
            or (
                recurrent_loops is None
                and reasoning_effort is None
                and (
                    self.config.reasoning_mode == "auto"
                    or self.config.reasoning_effort == "auto"
                )
            )
        )

        if is_auto and not self.training:
            max_loops = self.config.max_recurrent_loops
        else:
            if recurrent_loops is None and reasoning_effort is not None:
                recurrent_loops = EFFORT_TO_LOOPS.get(reasoning_effort, None)
            loops_val = recurrent_loops if recurrent_loops is not None else self.config.recurrent_loops
            max_loops = max(
                self.config.min_recurrent_loops,
                min(self.config.max_recurrent_loops, int(loops_val)),
            )

        core_start = self.config.num_prelude_layers
        core_end = core_start + self.config.num_core_layers
        core_layers = self.layers[core_start:core_end]

        precomputed_anchor_kv = {}
        if not (self.gradient_checkpointing and self.training):
            for layer in core_layers:
                if hasattr(layer, "self_attn") and isinstance(layer.self_attn, Lfm3ContextAnchoredAttention):
                    pos_emb = pos_emb_sliding if layer.layer_type == "sliding_attention" else pos_emb_full
                    normed_anchor = layer.operator_norm(anchor)
                    precomputed_anchor_kv[layer.layer_idx] = layer.self_attn.compute_anchor_kv(
                        normed_anchor, pos_emb
                    )

        step_halting_probs = []
        step_pred_depths = []
        prev_state = hidden_states
        actual_steps = 0

        for step in range(max_loops):
            step_clamp = min(step, self.step_embeddings.shape[0] - 1)
            h_in = hidden_states
            h_step = h_in + self.step_embeddings[step_clamp]

            for layer in core_layers:
                pos_emb = pos_emb_sliding if layer.layer_type == "sliding_attention" else pos_emb_full
                if self.gradient_checkpointing and self.training:
                    h_step = self._gradient_checkpointing_func(
                        layer.__call__,
                        h_step,
                        anchor,
                        pos_emb,
                        attention_mask,
                        None,
                        step,
                        None,
                    )
                else:
                    h_step = layer(
                        hidden_states=h_step,
                        anchor_states=anchor,
                        position_embeddings=pos_emb,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        core_depth_idx=step,
                        anchor_kv=precomputed_anchor_kv.get(layer.layer_idx),
                        **kwargs,
                    )

            alpha_step = (
                self.rd_controller.step_scales[step_clamp].to(h_step.dtype).clamp(0.05, 1.0)
                if self.rd_controller is not None
                else 1.0
            )
            delta_core = h_step - h_in
            gate_h_eff = self.gate_h.to(h_step.dtype).clamp(0.1, 1.5)
            gate_e_eff = self.gate_e.to(h_step.dtype).clamp(-0.25, 0.25)

            if step == 0:
                hidden_states = h_in + alpha_step * (delta_core * gate_h_eff)
            else:
                hidden_states = h_in + alpha_step * (
                    delta_core * gate_h_eff + (anchor - h_in) * gate_e_eff
                )
            actual_steps = step + 1

            if self.rd_controller is not None:
                p_ready, s_pred = self.rd_controller.predict_depth_and_prob(
                    hidden_states.detach(),
                    prev_state.detach(),
                    anchor.detach(),
                    step,
                )
                step_halting_probs.append(p_ready)
                step_pred_depths.append(s_pred)

                if (
                    is_auto
                    and not self.training
                    and actual_steps >= self.config.min_recurrent_loops
                ):
                    last_tok_prob = p_ready[:, -1].min().item()
                    if last_tok_prob >= self.config.act_halting_threshold:
                        break

            prev_state = hidden_states

        core_exit_state = hidden_states

        if not self.training:
            self.act_step_history.append(actual_steps)
            if len(self.act_step_history) > 128:
                self.act_step_history.pop(0)

        coda_start = core_end
        for idx in range(coda_start, len(self.layers)):
            layer = self.layers[idx]
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    None,
                    pos_emb_full,
                    attention_mask,
                    None,
                    0,
                    None,
                )
            else:
                hidden_states = layer(
                    hidden_states=hidden_states,
                    position_embeddings=pos_emb_full,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    **kwargs,
                )

        hidden_states = self.embedding_norm(hidden_states)

        return Lfm3RDModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            recurrent_steps=actual_steps,
            step_halting_probs=step_halting_probs,
            step_pred_depths=step_pred_depths,
            last_core_state=core_exit_state,
        )


class Lfm3RDForCausalLM(Lfm3RDPreTrainedModel, GenerationMixin):
    """
    Canonical Recurrent-Depth LFM3 Model with Language Modeling Head.
    """

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Lfm3RDConfig):
        super().__init__(config)
        self.model = Lfm3RDModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        recurrent_loops: int | None = None,
        reasoning_effort: str | None = None,
        **kwargs,
    ) -> Lfm3RDCausalLMOutputWithPast:
        if recurrent_loops is None and reasoning_effort is not None and reasoning_effort != "auto":
            recurrent_loops = EFFORT_TO_LOOPS.get(reasoning_effort, None)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            recurrent_loops=recurrent_loops,
            reasoning_effort=reasoning_effort,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        return Lfm3RDCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            recurrent_steps=outputs.recurrent_steps,
            step_halting_probs=outputs.step_halting_probs,
            step_pred_depths=outputs.step_pred_depths,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        """
        Prepares model inputs for HuggingFace generation pipeline.
        """
        Lfm3RDModel._ensure_cache_patched(past_key_values)

        if past_key_values is not None:
            if hasattr(past_key_values, "get_seq_length"):
                past_length = past_key_values.get_seq_length()
            else:
                past_length = 0
            if past_length > 0:
                if input_ids.shape[1] > past_length:
                    remove_prefix_length = past_length
                else:
                    remove_prefix_length = input_ids.shape[1] - 1
                input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        if position_ids is not None:
            position_ids = position_ids[:, -input_ids.shape[1]:]

        model_inputs = {"input_ids": input_ids}
        if inputs_embeds is not None and (past_key_values is None or past_key_values.get_seq_length() == 0):
            model_inputs["inputs_embeds"] = inputs_embeds
        else:
            model_inputs["input_ids"] = input_ids

        effort = kwargs.get("reasoning_effort", None)
        loops = kwargs.get("recurrent_loops", None)
        if loops is None and effort is not None and effort != "auto":
            loops = EFFORT_TO_LOOPS.get(effort, None)

        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "recurrent_loops": loops,
            "reasoning_effort": effort,
        })
        return model_inputs


__all__ = [
    "Lfm3RDConfig",
    "Lfm3RDPreTrainedModel",
    "Lfm3RDModel",
    "Lfm3RDForCausalLM",
    "Lfm3RDController",
    "Lfm3RMSNorm",
    "Lfm3ContextAnchoredAttention",
    "Lfm3Attention",
    "Lfm3ShortConv",
    "Lfm3DecoderLayer",
]
