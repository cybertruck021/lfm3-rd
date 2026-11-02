"""
Stage 1 Recovery Distillation and Manifold Alignment for Canonical LFM3-RD.

Trains the 8-layer Recurrent-Depth student model against the frozen 24-layer
LiquidAI/LFM3-1.7B teacher model across recurrent depths T in [1..8].

Safeguards against capability degradation, coordinate shock, and catastrophic forgetting:
1. Shard-Shuffled Multi-Skill English Mixture (covers all LFM3 operating modes):
   - Stream 1 (30%): FineWeb-Edu (sample-10BT) shuffled across shards - general prose & world knowledge.
   - Stream 2 (40%): SmolTalk (all) shuffled across all 14 skill shards (openhermes-100k,
     self-oss-instruct code, systemchats-30k system prompts, apigen-80k tool-use/JSON,
     smol-constraints IFEval, smol-summarize) rendered with <think></think>.
   - Stream 3 (30%): OpenThoughts-114k (metadata) shuffled across all 4 domains (math, code,
     science, puzzle) rendered with <think>{reasoning}</think>{solution}.
2. Document-Boundary Packing:
   - Full documents are packed without slicing across sequence borders. Unfitted documents are
     deferred to subsequent rows via carry_doc and remaining positions are masked out with pad_id.
3. On-Manifold Teacher-Dominated Loss (alpha_kd=0.85, alpha_cos=0.10, alpha_ce=0.05):
   - 95% of gradient signal aligns Student directly to Teacher LFM3-1.7B soft logit distribution
     and Coda 2048-d hidden representation. For T < 5, alpha_cos is scaled quadratically by (T/5)^2.
4. Synchronous Warmup & Unlocked Step Scales:
   - Adapters (lr=8e-5), Core (lr=6e-5), and Prelude/Coda (lr=3e-5) train synchronously from step 1
     with 50 warmup steps and cosine decay to 15% floor. Controller step_scales vector is unlocked.
"""

import argparse
import gc
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

sys.modules.pop("torchvision", None)
sys.modules.pop("torchvision.transforms", None)

import transformers.utils as _tu
import transformers.utils.import_utils as _tiu
_tu.import_utils = _tiu
_tu.is_torchvision_available = lambda: False
_tiu.is_torchvision_available = lambda: False

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

from lfm3_rd.configuration_lfm3_rd import Lfm3RDConfig
from lfm3_rd.modeling_lfm3_rd import Lfm3RDForCausalLM


def has_hardware_bf16(device: torch.device) -> bool:
    """
    Returns True only if the CUDA device has native hardware BF16 Tensor Cores (Ampere sm_80+).
    Turing GPUs (e.g. NVIDIA T4, sm_75) report torch.cuda.is_bf16_supported() == True via slow
    software emulation, so checking compute capability >= 8 is mandatory for hardware speed.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    idx = device.index if device.index is not None else torch.cuda.current_device()
    major, _ = torch.cuda.get_device_capability(idx)
    return major >= 8


def resolve_execution_devices(requested_mode: str = "auto") -> tuple[torch.device, torch.device]:
    """
    Assigns hardware devices for Student and Teacher models based on available GPUs.
    When two GPUs are available (such as Kaggle 2x T4), Student runs on cuda:0 and
    Teacher runs on cuda:1 for zero-overhead device-split parallelism.
    """
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2 and requested_mode in ("auto", "split"):
        dev_student = torch.device("cuda:0")
        dev_teacher = torch.device("cuda:1")
        print(
            f"Hardware configuration: Dual-GPU device-split "
            f"(Student: {torch.cuda.get_device_name(0)}, Teacher: {torch.cuda.get_device_name(1)})"
        )
    elif num_gpus >= 1 and requested_mode != "cpu":
        dev_student = torch.device("cuda:0")
        dev_teacher = torch.device("cuda:0")
        print(
            f"Hardware configuration: Single GPU "
            f"(Student & Teacher collocated on {torch.cuda.get_device_name(0)})"
        )
    else:
        dev_student = torch.device("cpu")
        dev_teacher = torch.device("cpu")
        print("Hardware configuration: CPU fallback")
    return dev_student, dev_teacher


def configure_3tier_parameter_groups(
    student_model: nn.Module,
    lr_adapters: float = 8e-5,
    lr_core: float = 6e-5,
    lr_prelude_coda: float = 3e-5,
    weight_decay: float = 0.01,
) -> list[dict]:
    """
    Constructs 3 functional parameter tiers with separate learning rates:
    - Tier 'adapters': k_proj_e, v_proj_e, gate_h, gate_e, step_embeddings, step_scales.
    - Tier 'core': layers.2, layers.3, layers.4, layers.5 (shared recurrent block).
    - Tier 'prelude_coda': layers.0, layers.1, layers.6, layers.7 (boundary layers).
    All 1D norms, gates, step embeddings, step scales, depthwise conv kernels, and biases get weight_decay = 0.0.
    """
    adapter_keywords = {"k_proj_e", "v_proj_e", "gate_h", "gate_e", "step_embeddings", "step_scales"}
    no_decay_keywords = {"norm", "gate_h", "gate_e", "step_embeddings", "step_scales", "bias", "conv.conv"}
    core_prefixes = ("model.layers.2.", "model.layers.3.", "model.layers.4.", "model.layers.5.")

    buckets = {
        ("adapters", True): [],
        ("adapters", False): [],
        ("core", True): [],
        ("core", False): [],
        ("prelude_coda", True): [],
        ("prelude_coda", False): [],
    }

    tier_lrs = {
        "adapters": lr_adapters,
        "core": lr_core,
        "prelude_coda": lr_prelude_coda,
    }

    for name, param in student_model.named_parameters():
        if not param.requires_grad:
            continue

        is_adapter = any(k in name for k in adapter_keywords)
        is_no_decay = any(k in name.lower() for k in no_decay_keywords) or param.dim() <= 1
        use_decay = not is_no_decay

        if is_adapter:
            tier = "adapters"
        elif name.startswith(core_prefixes):
            tier = "core"
        else:
            tier = "prelude_coda"

        buckets[(tier, use_decay)].append(param)

    groups = []
    for (tier, use_decay), params in buckets.items():
        if not params:
            continue
        base_lr = tier_lrs[tier]
        wd = weight_decay if use_decay else 0.0
        groups.append({
            "params": params,
            "lr": base_lr,
            "initial_lr": base_lr,
            "base_lr": base_lr,
            "weight_decay": wd,
            "tier": tier,
        })

    return groups


def update_staged_learning_rates(
    optimizer: torch.optim.Optimizer,
    step: int,
    max_steps: int,
    adapter_warmup: int = 50,
    core_freeze_steps: int = 0,
    core_warmup: int = 50,
    pc_freeze_steps: int = 0,
    pc_warmup: int = 50,
    min_lr_ratio: float = 0.15,
) -> dict[str, float]:
    """
    Applies linear warmup + cosine decay (with min_lr_ratio floor) per tier.
    """
    def compute_mult(s: int, freeze_s: int, warmup_s: int) -> float:
        if s <= freeze_s:
            return 0.0
        eff_s = s - freeze_s
        eff_warm = max(warmup_s - freeze_s, 1)
        eff_max = max(max_steps - freeze_s, 1)
        if eff_s < eff_warm:
            return float(eff_s) / float(eff_warm)
        progress = float(eff_s - eff_warm) / float(max(1, eff_max - eff_warm))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    mults = {
        "adapters": compute_mult(step, freeze_s=0, warmup_s=adapter_warmup),
        "core": compute_mult(step, freeze_s=core_freeze_steps, warmup_s=core_warmup),
        "prelude_coda": compute_mult(step, freeze_s=pc_freeze_steps, warmup_s=pc_warmup),
    }

    current_lrs = {}
    for group in optimizer.param_groups:
        tier = group.get("tier", "core")
        new_lr = group["base_lr"] * mults[tier]
        group["lr"] = new_lr
        current_lrs[tier] = new_lr

    return current_lrs


def normalize_foreign_thought_tags(text: str) -> str:
    """
    Normalizes non-LFM3 reasoning tags into canonical <think>...</think> format.
    """
    if "<|begin_of_thought|>" in text:
        text = text.replace("<|begin_of_thought|>", "<think>").replace("<|end_of_thought|>", "</think>")
        text = text.replace("<|begin_of_solution|>", "").replace("<|end_of_solution|>", "")
    return text.strip()


def encode_chat_with_masks(
    tokenizer,
    messages: list[dict],
    assistant_header_ids: list[int],
    im_end_id: int,
    bos_id: int,
    preserve_thinking: bool = True,
) -> tuple[list[int], list[int], list[int], str]:
    """
    Formats a conversation using LFM3 chat template and constructs token masks.
    """
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        preserve_thinking=preserve_thinking,
    )
    token_ids = tokenizer.encode(rendered, add_special_tokens=False)
    n = len(token_ids)
    ce_target_mask = [0] * n
    kd_target_mask = [1] * n

    for idx, tid in enumerate(token_ids):
        if tid == bos_id:
            kd_target_mask[idx] = 0

    hlen = len(assistant_header_ids)
    i = 0
    while i <= n - hlen:
        if token_ids[i : i + hlen] == assistant_header_ids:
            j = i + hlen
            while j < n:
                ce_target_mask[j] = 1
                if token_ids[j] == im_end_id:
                    if j + 1 < n and token_ids[j + 1] != bos_id:
                        kd_target_mask[j + 1] = 0
                    j += 1
                    break
                j += 1
            i = j
        else:
            i += 1

    return token_ids, kd_target_mask, ce_target_mask, rendered


def encode_raw_text_with_masks(
    tokenizer,
    text: str,
    bos_id: int,
    im_end_id: int,
) -> tuple[list[int], list[int], list[int]]:
    """
    Tokenizes a raw educational document with bos and eos tokens.
    """
    body_ids = tokenizer.encode(text.strip(), add_special_tokens=False)
    if not body_ids:
        return [], [], []
    token_ids = [bos_id] + body_ids + [im_end_id]
    n = len(token_ids)
    kd_target_mask = [0] + [1] * (n - 1)
    ce_target_mask = [0] + [1] * (n - 1)
    return token_ids, kd_target_mask, ce_target_mask


def create_verified_multistream_generator(
    tokenizer,
    seq_len: int = 1024,
    batch_size: int = 2,
    w_edu: float = 0.30,
    w_chat: float = 0.40,
    w_think: float = 0.30,
    auth_token: str | None = None,
):
    """
    Builds and verifies 3 shard-shuffled English data streams with document-boundary packing:
    1. FineWeb-Edu (HuggingFaceFW/fineweb-edu, sample-10BT, shuffled).
    2. SmolTalk (HuggingFaceTB/smoltalk, all, shuffled across all 14 subsets).
    3. OpenThoughts (open-thoughts/OpenThoughts-114k, metadata, shuffled across all domains).
    Documents that do not fit into remaining row space are preserved intact for the subsequent row.
    """
    from datasets import load_dataset

    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 124894
    im_end_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 124900
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else im_end_id
    assistant_header_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    max_single_doc_tokens = seq_len

    def stream_edu():
        seed_offset = 0
        while True:
            try:
                ds = load_dataset(
                    "HuggingFaceFW/fineweb-edu",
                    "sample-10BT",
                    split="train",
                    streaming=True,
                    token=auth_token,
                ).shuffle(seed=42 + seed_offset, buffer_size=1000)
                seed_offset += 1
                for row in ds:
                    txt = row.get("text", "")
                    if isinstance(txt, str) and len(txt.strip()) > 100:
                        t_ids, kd_m, ce_m = encode_raw_text_with_masks(tokenizer, txt, bos_id, im_end_id)
                        if 32 <= len(t_ids):
                            yield t_ids[:max_single_doc_tokens], kd_m[:max_single_doc_tokens], ce_m[:max_single_doc_tokens], {
                                "source": "fineweb-edu",
                                "cols": list(row.keys()),
                            }
            except Exception as e:
                print(f"Warning [Edu Stream]: {e}. Retrying with wikimedia/wikipedia...")
                fb = load_dataset(
                    "wikimedia/wikipedia", "20231101.en", split="train", streaming=True, token=auth_token
                ).shuffle(seed=42, buffer_size=500)
                for row in fb:
                    txt = row.get("text", "")
                    if isinstance(txt, str) and len(txt.strip()) > 100:
                        t_ids, kd_m, ce_m = encode_raw_text_with_masks(tokenizer, txt, bos_id, im_end_id)
                        if 32 <= len(t_ids):
                            yield t_ids[:max_single_doc_tokens], kd_m[:max_single_doc_tokens], ce_m[:max_single_doc_tokens], {
                                "source": "wikipedia-en",
                                "cols": list(row.keys()),
                            }

    def stream_chat():
        seed_offset = 0
        while True:
            try:
                ds = load_dataset(
                    "HuggingFaceTB/smoltalk",
                    "all",
                    split="train",
                    streaming=True,
                    token=auth_token,
                ).shuffle(seed=100 + seed_offset, buffer_size=1500)
                seed_offset += 1
                for row in ds:
                    msgs = row.get("messages", None)
                    sub_src = row.get("source", "smoltalk")
                    if isinstance(msgs, list) and len(msgs) >= 2:
                        clean_msgs = []
                        valid = True
                        for m in msgs:
                            role = m.get("role", "")
                            content = m.get("content", "")
                            if role in ("system", "user", "assistant") and isinstance(content, str) and content.strip():
                                clean_msgs.append({"role": role, "content": normalize_foreign_thought_tags(content)})
                            else:
                                valid = False
                                break
                        if valid and any(m["role"] == "assistant" for m in clean_msgs):
                            t_ids, kd_m, ce_m, rendered = encode_chat_with_masks(
                                tokenizer, clean_msgs, assistant_header_ids, im_end_id, bos_id, preserve_thinking=False
                            )
                            if sum(ce_m) >= 8:
                                yield t_ids[:max_single_doc_tokens], kd_m[:max_single_doc_tokens], ce_m[:max_single_doc_tokens], {
                                    "source": f"smoltalk/{sub_src}",
                                    "cols": list(row.keys()),
                                    "has_empty_think": "<think></think>" in rendered,
                                }
            except Exception as e:
                print(f"Warning [Chat Stream]: {e}. Retrying with everyday-conversations subset...")
                fb = load_dataset(
                    "HuggingFaceTB/smoltalk", "everyday-conversations", split="train", streaming=True, token=auth_token
                )
                for row in fb:
                    msgs = row.get("messages", [])
                    if isinstance(msgs, list) and len(msgs) >= 2:
                        t_ids, kd_m, ce_m, rendered = encode_chat_with_masks(
                            tokenizer, msgs, assistant_header_ids, im_end_id, bos_id, preserve_thinking=False
                        )
                        if sum(ce_m) >= 8:
                            yield t_ids[:max_single_doc_tokens], kd_m[:max_single_doc_tokens], ce_m[:max_single_doc_tokens], {
                                "source": "smoltalk/everyday",
                                "cols": list(row.keys()),
                                "has_empty_think": "<think></think>" in rendered,
                            }

    def stream_think():
        seed_offset = 0
        while True:
            try:
                ds = load_dataset(
                    "open-thoughts/OpenThoughts-114k",
                    "metadata",
                    split="train",
                    streaming=True,
                    token=auth_token,
                ).shuffle(seed=200 + seed_offset, buffer_size=1000)
                seed_offset += 1
                for row in ds:
                    problem = row.get("problem", "")
                    reasoning = row.get("deepseek_reasoning", "")
                    solution = row.get("deepseek_solution", "")
                    domain = row.get("domain", "general")
                    if (
                        isinstance(problem, str)
                        and isinstance(reasoning, str)
                        and isinstance(solution, str)
                        and problem.strip()
                        and reasoning.strip()
                        and solution.strip()
                    ):
                        msgs = [
                            {"role": "user", "content": problem.strip()},
                            {
                                "role": "assistant",
                                "thinking": normalize_foreign_thought_tags(reasoning),
                                "content": normalize_foreign_thought_tags(solution),
                            },
                        ]
                        t_ids, kd_m, ce_m, rendered = encode_chat_with_masks(
                            tokenizer, msgs, assistant_header_ids, im_end_id, bos_id, preserve_thinking=True
                        )
                        if sum(ce_m) >= 16:
                            yield t_ids[:max_single_doc_tokens], kd_m[:max_single_doc_tokens], ce_m[:max_single_doc_tokens], {
                                "source": f"OpenThoughts/{domain}",
                                "cols": list(row.keys()),
                                "has_full_think": "<think>" in rendered and "</think>" in rendered and "<think></think>" not in rendered,
                            }
            except Exception as e:
                print(f"Warning [Think Stream]: {e}. Falling back to open-r1/OpenR1-Math-220k...")
                fb = load_dataset(
                    "open-r1/OpenR1-Math-220k", "default", split="train", streaming=True, token=auth_token
                ).shuffle(seed=200, buffer_size=500)
                for row in fb:
                    msgs = row.get("messages", None)
                    if isinstance(msgs, list) and len(msgs) >= 2:
                        t_ids, kd_m, ce_m, rendered = encode_chat_with_masks(
                            tokenizer, msgs, assistant_header_ids, im_end_id, bos_id, preserve_thinking=True
                        )
                        if sum(ce_m) >= 16:
                            yield t_ids[:max_single_doc_tokens], kd_m[:max_single_doc_tokens], ce_m[:max_single_doc_tokens], {
                                "source": "OpenR1-Math-220k",
                                "cols": list(row.keys()),
                                "has_full_think": "<think>" in rendered and "</think>" in rendered and "<think></think>" not in rendered,
                            }

    edu_iter = stream_edu()
    chat_iter = stream_chat()
    think_iter = stream_think()

    print("\n" + "=" * 120)
    print("PRE-FLIGHT DATASET SCHEMA, SHARD SHUFFLE, CHAT TEMPLATE & TOKEN MASK AUDIT (100% ENGLISH MIXTURE)")
    print("=" * 120)
    t_e, kd_e, ce_e, meta_e = next(edu_iter)
    print(
        f"  [1. Edu Stream  ({w_edu*100:4.0f}%)] {meta_e['source']:<30} | cols={meta_e['cols'][:4]} "
        f"| tokens={len(t_e):4d} | KD_active={sum(kd_e):4d} | CE_active={sum(ce_e):4d}"
    )
    t_c, kd_c, ce_c, meta_c = next(chat_iter)
    print(
        f"  [2. Chat Stream ({w_chat*100:4.0f}%)] {meta_c['source']:<30} | cols={meta_c['cols']} "
        f"| tokens={len(t_c):4d} | KD_active={sum(kd_c):4d} | CE_active={sum(ce_c):4d} (ans-only) | <think></think>={meta_c['has_empty_think']}"
    )
    t_r, kd_r, ce_r, meta_r = next(think_iter)
    print(
        f"  [3. Think Stream({w_think*100:4.0f}%)] {meta_r['source']:<30} | cols={meta_r['cols'][:5]} "
        f"| tokens={len(t_r):4d} | KD_active={sum(kd_r):4d} | CE_active={sum(ce_r):4d} (ans-only) | <think>...</think>={meta_r['has_full_think']}"
    )
    print("=" * 120)

    carry_doc = None

    while True:
        batch_toks = []
        batch_attn = []
        batch_kd = []
        batch_ce = []

        for _ in range(batch_size):
            row_toks = []
            row_attn = []
            row_kd = []
            row_ce = []

            while len(row_toks) < seq_len:
                if carry_doc is not None:
                    t_ids, kd_m, ce_m = carry_doc
                    carry_doc = None
                else:
                    r = random.random()
                    if r < w_edu:
                        t_ids, kd_m, ce_m, _ = next(edu_iter)
                    elif r < w_edu + w_chat:
                        t_ids, kd_m, ce_m, _ = next(chat_iter)
                    else:
                        t_ids, kd_m, ce_m, _ = next(think_iter)

                remaining_space = seq_len - len(row_toks)
                if len(t_ids) <= remaining_space:
                    row_toks.extend(t_ids)
                    row_attn.extend([1] * len(t_ids))
                    row_kd.extend(kd_m)
                    row_ce.extend(ce_m)
                else:
                    carry_doc = (t_ids, kd_m, ce_m)
                    pad_len = remaining_space
                    row_toks.extend([pad_id] * pad_len)
                    row_attn.extend([0] * pad_len)
                    row_kd.extend([0] * pad_len)
                    row_ce.extend([0] * pad_len)
                    break

            batch_toks.append(row_toks)
            batch_attn.append(row_attn)
            batch_kd.append(row_kd)
            batch_ce.append(row_ce)

        input_ids = torch.tensor(batch_toks, dtype=torch.long)
        attention_mask = torch.tensor(batch_attn, dtype=torch.long)
        kd_mask = torch.tensor(batch_kd, dtype=torch.bool)
        ce_mask = torch.tensor(batch_ce, dtype=torch.bool)

        yield {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "kd_mask": kd_mask,
            "ce_mask": ce_mask,
        }


def compute_chunked_distillation_loss(
    s_hidden: torch.Tensor,
    t_hidden: torch.Tensor,
    input_ids: torch.Tensor,
    kd_mask: torch.Tensor,
    ce_mask: torch.Tensor,
    student_lm_head: nn.Module,
    teacher_lm_head: nn.Module,
    temperature: float = 2.0,
    alpha_kd: float = 0.85,
    alpha_ce: float = 0.05,
    alpha_cos: float = 0.10,
    chunk_tokens: int = 256,
) -> tuple[torch.Tensor, float, float, float, float, float]:
    """
    Computes exact Teacher-Student KL divergence + Assistant-Masked Cross-Entropy + Hidden Cosine Loss.
    """
    valid_kd = kd_mask[:, 1:]
    valid_ce = ce_mask[:, 1:] & valid_kd

    s_h_flat = s_hidden[:, :-1, :][valid_kd]
    t_h_flat = t_hidden[:, :-1, :][valid_kd]
    lbls_all = input_ids[:, 1:].clone()
    lbls_all[~valid_ce] = -100
    lbls_flat = lbls_all[valid_kd]

    total_kd_tokens = int(s_h_flat.shape[0])
    total_ce_tokens = int((lbls_flat != -100).sum().item())

    if total_kd_tokens == 0:
        zero_loss = s_hidden.sum() * 0.0
        return zero_loss, 0.0, 0.0, 0.0, 0.0, 0.0

    cos_sim = F.cosine_similarity(s_h_flat.float(), t_h_flat.float(), dim=-1).mean()
    cos_loss = 1.0 - cos_sim

    s_h_detached = s_h_flat.detach().requires_grad_(True)

    total_kl_val = 0.0
    total_ce_val = 0.0
    total_top1_matches = 0.0
    total_top5_matches = 0.0

    for start_idx in range(0, total_kd_tokens, chunk_tokens):
        end_idx = min(start_idx + chunk_tokens, total_kd_tokens)

        s_chunk = s_h_detached[start_idx:end_idx]
        t_chunk = t_h_flat[start_idx:end_idx]
        lbl_chunk = lbls_flat[start_idx:end_idx]

        with torch.no_grad():
            t_logits_c = F.linear(t_chunk.to(teacher_lm_head.weight.dtype), teacher_lm_head.weight).float()
            t_soft_c = F.softmax(t_logits_c / temperature, dim=-1)
            t_top1_c = t_logits_c.argmax(dim=-1)
            t_top5_c = torch.topk(t_logits_c, k=5, dim=-1).indices
            del t_logits_c

        s_logits_c = student_lm_head(s_chunk.to(student_lm_head.weight.dtype)).float()
        s_log_soft_c = F.log_softmax(s_logits_c / temperature, dim=-1)

        kl_c = (
            F.kl_div(s_log_soft_c, t_soft_c, reduction="sum")
            * (temperature**2)
            / float(total_kd_tokens)
        )

        if total_ce_tokens > 0 and (lbl_chunk != -100).any():
            ce_c = (
                F.cross_entropy(s_logits_c, lbl_chunk, ignore_index=-100, reduction="sum")
                / float(total_ce_tokens)
            )
        else:
            ce_c = s_logits_c.sum() * 0.0

        chunk_loss = alpha_kd * kl_c + alpha_ce * ce_c
        chunk_loss.backward()

        with torch.no_grad():
            total_kl_val += kl_c.item()
            total_ce_val += ce_c.item()
            s_top1_c = s_logits_c.argmax(dim=-1)
            total_top1_matches += (s_top1_c == t_top1_c).float().sum().item()

            s_top5_c = torch.topk(s_logits_c, k=5, dim=-1).indices
            eq_c = (s_top5_c.unsqueeze(2) == t_top5_c.unsqueeze(1)).any(dim=-1)
            total_top5_matches += eq_c.float().sum().item() / 5.0

        del s_logits_c, s_log_soft_c, t_soft_c, kl_c, ce_c, chunk_loss

    surrogate_logit_loss = (s_h_flat * s_h_detached.grad).sum()
    total_loss_expr = surrogate_logit_loss + alpha_cos * cos_loss

    total_loss_scalar = alpha_kd * total_kl_val + alpha_ce * total_ce_val + alpha_cos * cos_loss.item()
    top1_agr_pct = (total_top1_matches / float(total_kd_tokens)) * 100.0
    top5_ovr_pct = (total_top5_matches / float(total_kd_tokens)) * 100.0

    return (
        total_loss_expr,
        total_loss_scalar,
        total_kl_val,
        total_ce_val,
        top1_agr_pct,
        top5_ovr_pct,
    )


def save_complete_rd_checkpoint(
    student_model: Lfm3RDForCausalLM,
    tokenizer,
    target_dir: str,
    save_dtype: torch.dtype = torch.bfloat16,
):
    """
    Saves standalone model checkpoint with custom modeling and configurations.
    """
    os.makedirs(target_dir, exist_ok=True)
    orig_dtype = next(student_model.parameters()).dtype
    was_training = student_model.training

    try:
        if orig_dtype == torch.float32 and save_dtype != torch.float32:
            student_model.to(save_dtype)
        student_model.save_pretrained(target_dir)
    finally:
        if orig_dtype == torch.float32 and save_dtype != torch.float32:
            student_model.to(orig_dtype)
        if was_training:
            student_model.train()

    tokenizer.save_pretrained(target_dir)

    pkg_dir = os.path.join(_repo_root, "lfm3_rd")
    for fname in ("configuration_lfm3_rd.py", "modeling_lfm3_rd.py", "__init__.py"):
        src = os.path.join(pkg_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(target_dir, fname))

    cfg_file = os.path.join(target_dir, "config.json")
    if os.path.exists(cfg_file):
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg_obj = json.load(f)
        cfg_obj["auto_map"] = {
            "AutoConfig": "configuration_lfm3_rd.Lfm3RDConfig",
            "AutoModelForCausalLM": "modeling_lfm3_rd.Lfm3RDForCausalLM",
        }
        with open(cfg_file, "w", encoding="utf-8") as f:
            json.dump(cfg_obj, f, indent=2)


def evaluate_held_out(
    student_model: Lfm3RDForCausalLM,
    tokenizer,
    device: torch.device,
    autocast_dtype: torch.dtype,
    step_num: int,
):
    """
    Evaluates student generation quality across 3 distinct task families.
    """
    print("\n" + "=" * 120)
    print(f"HELD-OUT MULTI-TASK GENERATION BENCHMARK (STEP {step_num})")
    print("=" * 120)

    benchmarks = [
        (
            "1. Direct Fact (enable_thinking=False)",
            [{"role": "user", "content": "What is the capital of France and what river flows through it?"}],
            False,
            50,
        ),
        (
            "2. Math Reasoning (enable_thinking=True)",
            [
                {
                    "role": "user",
                    "content": "A farmer has chickens and rabbits. In total, there are 37 heads and 98 legs. How many chickens and how many rabbits does he have? Solve step by step.",
                }
            ],
            True,
            120,
        ),
        (
            "3. Code & System Prompt (enable_thinking=False)",
            [
                {"role": "system", "content": "You are a concise Python coding assistant. Return only valid Python code."},
                {"role": "user", "content": "Write a Python function `is_prime(n: int) -> bool` that checks if a number is prime."},
            ],
            False,
            70,
        ),
    ]

    student_model.eval()
    for label, msgs, use_think, max_toks in benchmarks:
        formatted = tokenizer.apply_chat_template(
            msgs, enable_thinking=use_think, add_generation_prompt=True, tokenize=False
        )
        inp = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).to(device)

        print(f"\n--- {label} ---")

        for effort in ("fast", "medium", "auto"):
            student_model.model.act_step_history.clear()
            t0 = time.time()
            with torch.inference_mode(), torch.amp.autocast(
                device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"
            ):
                out_ids = student_model.generate(
                    input_ids=inp["input_ids"],
                    max_new_tokens=max_toks,
                    do_sample=False,
                    reasoning_effort=effort,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            dt = max(time.time() - t0, 1e-4)
            new_tokens = out_ids[0, inp["input_ids"].shape[1] :]
            decoded_text = (
                tokenizer.decode(new_tokens, skip_special_tokens=False)
                .replace("<|im_end|>", "")
                .replace("\n", " ")
                .strip()
            )
            speed = len(new_tokens) / dt

            trace_str = ""
            if effort == "auto" and student_model.model.act_step_history:
                hist = student_model.model.act_step_history
                trace_str = f" | steps={sum(hist)/len(hist):.1f} {hist[:6]}"

            print(f"  [{effort:<6} | {speed:5.1f} tok/s{trace_str}]: {decoded_text[:165]}")

    print("=" * 120 + "\n")
    student_model.train()


def run_stage1_distillation(
    student_model_id: str = "cybertruck32489/LFM3-1.7B-RD-v2",
    teacher_model_id: str = "LiquidAI/LFM3-1.7B",
    output_dir: str = "./LFM3-1.7B-RD-healed",
    batch_size: int = 2,
    grad_accum_steps: int = 4,
    max_steps: int = 1000,
    seq_len: int = 1024,
    lr_adapters: float = 8e-5,
    lr_core: float = 6e-5,
    lr_prelude_coda: float = 3e-5,
    weight_decay: float = 0.01,
    temperature: float = 2.0,
    alpha_kd: float = 0.85,
    alpha_ce: float = 0.05,
    alpha_cos: float = 0.10,
    w_edu: float = 0.30,
    w_chat: float = 0.40,
    w_think: float = 0.30,
    freeze_embeddings: bool = True,
    chunk_tokens: int = 256,
    adapter_warmup: int = 50,
    core_freeze_steps: int = 0,
    core_warmup: int = 50,
    pc_freeze_steps: int = 0,
    pc_warmup: int = 50,
    min_lr_ratio: float = 0.15,
    log_every: int = 5,
    eval_every: int = 100,
    save_every: int = 200,
    device_mode: str = "auto",
    hub_repo_id: str | None = None,
    private_hub: bool = False,
    token: str | None = None,
):
    """
    Executes Stage 1 Recovery Distillation training loop with full telemetry.
    """
    t_start = time.time()
    auth_token = token or os.environ.get("HF_TOKEN", None)
    if auth_token:
        try:
            auth_token.encode("ascii")
            if not auth_token.startswith("hf_"):
                auth_token = None
        except UnicodeEncodeError:
            print("Warning: HF_TOKEN contains non-ASCII characters; ignoring token for public repositories.")
            auth_token = None
    os.makedirs(output_dir, exist_ok=True)

    dev_student, dev_teacher = resolve_execution_devices(device_mode)

    print(f"\nLoading Tokenizer from: {student_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(student_model_id, token=auth_token, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    teacher_hw_bf16 = has_hardware_bf16(dev_teacher)
    teacher_dtype = torch.bfloat16 if teacher_hw_bf16 else (torch.float16 if dev_teacher.type == "cuda" else torch.float32)
    print(f"Loading Frozen Teacher Model from: {teacher_model_id} to {dev_teacher} (dtype={teacher_dtype})...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_id,
        torch_dtype=teacher_dtype,
        device_map=dev_teacher,
        token=auth_token,
        trust_remote_code=True,
    )
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad_(False)

    student_hw_bf16 = has_hardware_bf16(dev_student)
    student_weight_dtype = torch.bfloat16 if student_hw_bf16 else torch.float32
    autocast_dtype = torch.bfloat16 if student_hw_bf16 else (torch.float16 if dev_student.type == "cuda" else torch.float32)
    use_scaler = (dev_student.type == "cuda") and (not student_hw_bf16)

    print(
        f"Loading Student Model from: {student_model_id} to {dev_student} "
        f"(master_dtype={student_weight_dtype}, autocast_dtype={autocast_dtype}, grad_scaler={use_scaler})..."
    )
    try:
        student_config = Lfm3RDConfig.from_pretrained(student_model_id, token=auth_token)
        student_model = Lfm3RDForCausalLM.from_pretrained(
            student_model_id,
            config=student_config,
            torch_dtype=student_weight_dtype,
            device_map=dev_student,
            token=auth_token,
        )
    except Exception:
        student_model = AutoModelForCausalLM.from_pretrained(
            student_model_id,
            torch_dtype=student_weight_dtype,
            device_map=dev_student,
            token=auth_token,
            trust_remote_code=True,
        )

    for name, param in student_model.model.rd_controller.named_parameters():
        if "step_scales" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
    print("Stage 1 Controller State: Kinematic step_scales unlocked (requires_grad=True), router MLP weights locked.")

    if freeze_embeddings:
        student_model.model.embed_tokens.weight.requires_grad_(False)
        student_model.lm_head.weight.requires_grad_(False)
        student_model.model.embedding_norm.weight.requires_grad_(False)
        print("Stage 1 Vocabulary Anchor: embed_tokens, lm_head, and embedding_norm frozen.")

    if hasattr(student_model, "gradient_checkpointing_enable"):
        student_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("Gradient Checkpointing enabled for Student model (use_reentrant=False).")

    student_model.train()

    trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in student_model.parameters() if not p.requires_grad)
    print(f"Student Parameter Budget: {trainable_params / 1e6:.1f}M trainable | {frozen_params / 1e6:.1f}M frozen")

    param_groups = configure_3tier_parameter_groups(
        student_model,
        lr_adapters=lr_adapters,
        lr_core=lr_core,
        lr_prelude_coda=lr_prelude_coda,
        weight_decay=weight_decay,
    )

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    data_stream = create_verified_multistream_generator(
        tokenizer=tokenizer,
        seq_len=seq_len,
        batch_size=batch_size,
        w_edu=w_edu,
        w_chat=w_chat,
        w_think=w_think,
        auth_token=auth_token,
    )

    print("\n" + "=" * 135)
    print(
        f"STAGE 1 DISTILLATION LAUNCHED | STEPS: {max_steps} | SEQ_LEN: {seq_len} | EFF BATCH: {batch_size * grad_accum_steps} "
        f"| Base Weights: KD={alpha_kd:.2f}, Cos={alpha_cos:.2f}, CE={alpha_ce:.2f}"
    )
    print("=" * 135)
    print(
        f"  {'Step':<11} | {'Loss':>8} | {'KL':>8} | {'CE(Ans)':>8} | {'T':>3} | {'Top1':>7} | {'Top5':>7} "
        f"| {'GradN':>7} | {'LR_Adp':>8} | {'LR_Core':>8} | {'Speed':>9}"
    )
    print("-" * 135)

    running_total_loss = 0.0
    running_kl_loss = 0.0
    running_ce_loss = 0.0
    running_top1_agr = 0.0
    running_top5_ovr = 0.0
    step_tokens = 0
    t_lap = time.time()

    optimizer.zero_grad()

    for step in range(1, max_steps + 1):
        tier_lrs = update_staged_learning_rates(
            optimizer=optimizer,
            step=step,
            max_steps=max_steps,
            adapter_warmup=adapter_warmup,
            core_freeze_steps=core_freeze_steps,
            core_warmup=core_warmup,
            pc_freeze_steps=pc_freeze_steps,
            pc_warmup=pc_warmup,
            min_lr_ratio=min_lr_ratio,
        )

        step_sub_loss = 0.0
        step_sub_kl = 0.0
        step_sub_ce = 0.0
        step_sub_top1 = 0.0
        step_sub_top5 = 0.0
        last_T = 5

        for micro_step in range(grad_accum_steps):
            r_depth = random.random()
            if r_depth < 0.60:
                current_T = 5
            elif r_depth < 0.85:
                current_T = random.choice([1, 2, 3, 4])
            else:
                current_T = random.choice([6, 8])
            last_T = current_T

            eff_alpha_cos = alpha_cos * ((current_T / 5.0) ** 2) if current_T < 5 else alpha_cos

            batch = next(data_stream)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            kd_mask = batch["kd_mask"]
            ce_mask = batch["ce_mask"]

            with torch.no_grad():
                t_base_out = teacher_model.model(
                    input_ids=input_ids.to(dev_teacher),
                    attention_mask=attention_mask.to(dev_teacher),
                    use_cache=False,
                )
                t_hidden = t_base_out.last_hidden_state.to(dev_student, non_blocking=True)
                del t_base_out

            inp_student = input_ids.to(dev_student, non_blocking=True)
            mask_student = attention_mask.to(dev_student, non_blocking=True)
            kd_mask_student = kd_mask.to(dev_student, non_blocking=True)
            ce_mask_student = ce_mask.to(dev_student, non_blocking=True)

            with torch.amp.autocast(
                device_type=dev_student.type, dtype=autocast_dtype, enabled=dev_student.type == "cuda"
            ):
                s_base_out = student_model.model(
                    input_ids=inp_student,
                    attention_mask=mask_student,
                    recurrent_loops=current_T,
                    use_cache=False,
                )
                s_hidden = s_base_out.last_hidden_state

            (
                loss_expr,
                loss_scalar,
                kl_val,
                ce_val,
                agr,
                ovr,
            ) = compute_chunked_distillation_loss(
                s_hidden=s_hidden,
                t_hidden=t_hidden,
                input_ids=inp_student,
                kd_mask=kd_mask_student,
                ce_mask=ce_mask_student,
                student_lm_head=student_model.lm_head,
                teacher_lm_head=student_model.lm_head if freeze_embeddings else teacher_model.lm_head,
                temperature=temperature,
                alpha_kd=alpha_kd,
                alpha_ce=alpha_ce,
                alpha_cos=eff_alpha_cos,
                chunk_tokens=chunk_tokens,
            )

            scaled_loss = loss_expr / grad_accum_steps

            if use_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            step_sub_loss += loss_scalar
            step_sub_kl += kl_val
            step_sub_ce += ce_val
            step_sub_top1 += agr
            step_sub_top5 += ovr
            step_tokens += input_ids.numel()

            del s_base_out, s_hidden, t_hidden, loss_expr, scaled_loss

        if use_scaler:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=0.5).item()
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=0.5).item()
            optimizer.step()

        optimizer.zero_grad()

        running_total_loss += step_sub_loss / grad_accum_steps
        running_kl_loss += step_sub_kl / grad_accum_steps
        running_ce_loss += step_sub_ce / grad_accum_steps
        running_top1_agr += step_sub_top1 / grad_accum_steps
        running_top5_ovr += step_sub_top5 / grad_accum_steps

        if step % log_every == 0:
            dt_lap = max(time.time() - t_lap, 1e-4)
            tok_speed = step_tokens / dt_lap

            avg_loss = running_total_loss / log_every
            avg_kl = running_kl_loss / log_every
            avg_ce = running_ce_loss / log_every
            avg_top1 = running_top1_agr / log_every
            avg_top5 = running_top5_ovr / log_every

            lr_adp = tier_lrs.get("adapters", 0.0)
            lr_cor = tier_lrs.get("core", 0.0)

            vram_info = ""
            if torch.cuda.is_available():
                m0 = torch.cuda.max_memory_allocated(0) / (1024**3)
                vram_info = f" | G0={m0:.1f}G"
                if torch.cuda.device_count() >= 2:
                    m1 = torch.cuda.max_memory_allocated(1) / (1024**3)
                    vram_info += f" G1={m1:.1f}G"

            print(
                f"  [{step:05d}/{max_steps:05d}] | {avg_loss:8.4f} | {avg_kl:8.4f} | {avg_ce:8.4f} | {last_T:3d} "
                f"| {avg_top1:6.1f}% | {avg_top5:6.1f}% | {grad_norm:7.3f} | {lr_adp:8.1e} | {lr_cor:8.1e} "
                f"| {tok_speed:5.0f} t/s{vram_info}"
            )

            running_total_loss = 0.0
            running_kl_loss = 0.0
            running_ce_loss = 0.0
            running_top1_agr = 0.0
            running_top5_ovr = 0.0
            step_tokens = 0
            t_lap = time.time()

        if step % eval_every == 0 or step == max_steps:
            evaluate_held_out(student_model, tokenizer, dev_student, autocast_dtype, step)

        if step % save_every == 0 and step < max_steps:
            ckpt_path = os.path.join(output_dir, f"checkpoint-step-{step}")
            print(f"Saving intermediate checkpoint to {ckpt_path}...")
            save_complete_rd_checkpoint(student_model, tokenizer, ckpt_path)

    print("\n" + "=" * 100)
    print("STAGE 1 RECOVERY DISTILLATION COMPLETE: SAVING MASTER ARTIFACTS")
    print("=" * 100)

    final_model_dir = output_dir
    save_complete_rd_checkpoint(student_model, tokenizer, final_model_dir)

    train_meta_file = os.path.join(final_model_dir, "training_meta.json")
    with open(train_meta_file, "w", encoding="utf-8") as f:
        json.dump({
            "stage": 1,
            "total_steps": max_steps,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "effective_batch_size": batch_size * grad_accum_steps,
            "lr_adapters": lr_adapters,
            "lr_core": lr_core,
            "lr_prelude_coda": lr_prelude_coda,
            "temperature": temperature,
            "alpha_kd": alpha_kd,
            "alpha_ce": alpha_ce,
            "alpha_cos": alpha_cos,
            "mixture_weights": {"fineweb_edu": w_edu, "smoltalk": w_chat, "openthoughts": w_think},
            "freeze_embeddings": freeze_embeddings,
            "elapsed_seconds": time.time() - t_start,
        }, f, indent=2)

    print(f"Master checkpoint saved successfully to {final_model_dir}")

    if hub_repo_id:
        print(f"\nUploading Stage 1 Healed Model to Hugging Face Hub: {hub_repo_id}...")
        api = HfApi(token=auth_token)
        api.create_repo(repo_id=hub_repo_id, token=auth_token, private=private_hub, exist_ok=True)
        api.upload_folder(
            folder_path=final_model_dir,
            repo_id=hub_repo_id,
            token=auth_token,
            ignore_patterns=["checkpoint-step-*"],
        )
        print(f"Successfully published to https://huggingface.co/{hub_repo_id}")

    total_time = time.time() - t_start
    print(f"\nAll Stage 1 operations completed in {total_time / 60:.2f} minutes.")


def main():
    """
    CLI interface for Stage 1 Recovery Distillation.
    """
    parser = argparse.ArgumentParser(description="Stage 1 Recovery Distillation for Canonical LFM3-RD")
    parser.add_argument("--student-model", type=str, default="cybertruck32489/LFM3-1.7B-RD-v2", help="Hugging Face repo or local path of converted RD student")
    parser.add_argument("--teacher-model", type=str, default="LiquidAI/LFM3-1.7B", help="Hugging Face repo of unrolled teacher model")
    parser.add_argument("--output-dir", type=str, default="./LFM3-1.7B-RD-healed", help="Output directory for healed model checkpoint")
    parser.add_argument("--batch-size", type=int, default=2, help="Micro-batch size per forward step")
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max-steps", type=int, default=1000, help="Total optimization steps")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length for packed training chunks")
    parser.add_argument("--lr-adapters", type=float, default=8e-5, help="Peak LR for recurrent bridge adapters")
    parser.add_argument("--lr-core", type=float, default=6e-5, help="Peak LR for shared recurrent Core layers")
    parser.add_argument("--lr-prelude-coda", type=float, default=3e-5, help="Peak LR for Prelude and Coda")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for 2D matrix weights")
    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation temperature for soft logit targets")
    parser.add_argument("--alpha-kd", type=float, default=0.85, help="Loss weight for Teacher-Student KL divergence")
    parser.add_argument("--alpha-ce", type=float, default=0.05, help="Loss weight for assistant-masked Cross-Entropy")
    parser.add_argument("--alpha-cos", type=float, default=0.10, help="Loss weight for Coda hidden-state cosine alignment")
    parser.add_argument("--w-edu", type=float, default=0.30, help="Mixture weight for FineWeb-Edu stream")
    parser.add_argument("--w-chat", type=float, default=0.40, help="Mixture weight for SmolTalk multi-skill chat stream")
    parser.add_argument("--w-think", type=float, default=0.30, help="Mixture weight for OpenThoughts <think> reasoning stream")
    parser.add_argument("--train-embeddings", action="store_true", help="Unfreeze embed_tokens/lm_head during Stage 1 (default: frozen)")
    parser.add_argument("--chunk-tokens", type=int, default=256, help="Token chunk size for memory-efficient vocabulary KL+CE")
    parser.add_argument("--adapter-warmup", type=int, default=50, help="Warmup steps for adapter tier")
    parser.add_argument("--core-freeze-steps", type=int, default=0, help="Initial steps to keep Core frozen")
    parser.add_argument("--core-warmup", type=int, default=50, help="Step at which Core reaches peak LR")
    parser.add_argument("--pc-freeze-steps", type=int, default=0, help="Initial steps to keep Prelude/Coda frozen")
    parser.add_argument("--pc-warmup", type=int, default=50, help="Step at which Prelude/Coda reach peak LR")
    parser.add_argument("--min-lr-ratio", type=float, default=0.15, help="Minimum LR ratio for cosine floor")
    parser.add_argument("--log-every", type=int, default=5, help="Logging frequency in optimization steps")
    parser.add_argument("--eval-every", type=int, default=100, help="Held-out evaluation frequency in optimization steps")
    parser.add_argument("--save-every", type=int, default=200, help="Checkpoint saving frequency in optimization steps")
    parser.add_argument("--device-mode", type=str, default="auto", choices=["auto", "split", "single", "cpu"], help="Device placement mode")
    parser.add_argument("--hub-repo-id", type=str, default=None, help="Target Hugging Face repository ID to upload healed model")
    parser.add_argument("--private-hub", action="store_true", help="Set uploaded Hugging Face repository as private")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face authentication token")
    args = parser.parse_args()

    run_stage1_distillation(
        student_model_id=args.student_model,
        teacher_model_id=args.teacher_model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        max_steps=args.max_steps,
        seq_len=args.seq_len,
        lr_adapters=args.lr_adapters,
        lr_core=args.lr_core,
        lr_prelude_coda=args.lr_prelude_coda,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        alpha_kd=args.alpha_kd,
        alpha_ce=args.alpha_ce,
        alpha_cos=args.alpha_cos,
        w_edu=args.w_edu,
        w_chat=args.w_chat,
        w_think=args.w_think,
        freeze_embeddings=not args.train_embeddings,
        chunk_tokens=args.chunk_tokens,
        adapter_warmup=args.adapter_warmup,
        core_freeze_steps=args.core_freeze_steps,
        core_warmup=args.core_warmup,
        pc_freeze_steps=args.pc_freeze_steps,
        pc_warmup=args.pc_warmup,
        min_lr_ratio=args.min_lr_ratio,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        device_mode=args.device_mode,
        hub_repo_id=args.hub_repo_id,
        private_hub=args.private_hub,
        token=args.token,
    )


if __name__ == "__main__":
    main()
