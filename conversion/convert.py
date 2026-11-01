# Copyright 2026 LFM3-RD Authors. All rights reserved.
"""
CLI entrypoint to convert `LiquidAI/LFM3-1.7B` into `LFM3-1.7B-RD`
(Prelude -> Shared Recurrent Core + Phase LoRAs -> Coda).

Usage:
    python -m conversion.convert --base-model LiquidAI/LFM3-1.7B --output-dir ./LFM3-1.7B-RD-export
"""

import argparse
import gc
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from conversion.calibration import build_calibration_domains, compute_cholesky_factors
from conversion.decomposition import align_conv_bx_gauge, balance_ba, decompose_5_motifs_rd


def normalize_lfm3_layer_types(num_layers: int = 24) -> list[str]:
    return [
        "full_attention" if i % 4 == 2 else "sliding_attention" if i % 4 == 0 and i > 0 else "conv"
        for i in range(num_layers)
    ]


def build_rd_controller_state_dict(D: int = 2048, n_motifs: int = 5, router_dim: int = 128) -> dict[str, torch.Tensor]:
    """Initializes the exact state_dict keys for the final Lfm3RDController architecture."""
    torch.manual_seed(102)
    inj_rank = 64
    ref_rank = 64
    return {
        "model.rd_controller.act_norm.weight":          torch.ones(D, dtype=torch.bfloat16),
        "model.rd_controller.vel_norm.weight":          torch.ones(D, dtype=torch.bfloat16),
        "model.rd_controller.state_proj.weight":        (torch.randn(router_dim, D) * 0.02).to(torch.bfloat16),
        "model.rd_controller.vel_proj.weight":          (torch.randn(router_dim, D) * 0.02).to(torch.bfloat16),
        "model.rd_controller.dyn_proj.weight":          (torch.randn(router_dim, 4) * 0.02).to(torch.bfloat16),
        "model.rd_controller.dyn_proj.bias":            torch.zeros(router_dim, dtype=torch.bfloat16),
        "model.rd_controller.halting_head.weight":      (torch.randn(1, router_dim) * 0.02).to(torch.bfloat16),
        "model.rd_controller.halting_head.bias":        torch.tensor([-0.85], dtype=torch.bfloat16),
        "model.rd_controller.step_scales":              torch.ones(32, dtype=torch.bfloat16),
        "model.rd_controller.inj_norm.weight":          torch.ones(D, dtype=torch.bfloat16),
        "model.rd_controller.inj_A":                    (torch.randn(inj_rank, D) / math.sqrt(D)).to(torch.bfloat16),
        "model.rd_controller.inj_B":                    torch.zeros((D, inj_rank), dtype=torch.bfloat16),
        "model.rd_controller.inj_gates":                torch.full((6,), 0.20, dtype=torch.bfloat16),
        "model.rd_controller.ref_norm.weight":          torch.ones(D, dtype=torch.bfloat16),
        "model.rd_controller.ref_A":                    (torch.randn(n_motifs, ref_rank, D) / math.sqrt(D)).to(torch.bfloat16),
        "model.rd_controller.ref_B":                    torch.zeros((n_motifs, D, ref_rank), dtype=torch.bfloat16),
        "model.rd_controller.refine_gates":             torch.zeros(32, dtype=torch.bfloat16),
        "model.rd_controller.refine_token_proj.weight": torch.zeros((1, D), dtype=torch.bfloat16),
        "model.rd_controller.refine_token_proj.bias":   torch.zeros(1, dtype=torch.bfloat16),
    }


def convert_lfm3_to_rd(
    base_model_id: str = "LiquidAI/LFM3-1.7B",
    output_dir: str = "./LFM3-1.7B-RD-export",
    rank_attn: int = 640,
    rank_conv: int = 640,
    rank_kv: int = 512,
    samples_per_dataset: int = 60,
    device: str = "cuda",
):
    t0 = time.time()
    p_layers, n_motifs, motif_size = 2, 5, 4

    print(f"⏳ [1/4] Downloading base model `{base_model_id}`...")
    orig_snap_dir = snapshot_download(repo_id=base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(orig_snap_dir, trust_remote_code=True)

    # Load and sanitize base config so 'linear_attention' in layer_types is mapped to 'conv'
    base_cfg = AutoConfig.from_pretrained(orig_snap_dir, trust_remote_code=True)
    clean_layer_types = normalize_lfm3_layer_types(base_cfg.num_hidden_layers)
    base_cfg.layer_types = clean_layer_types

    base_model = AutoModelForCausalLM.from_pretrained(
        orig_snap_dir,
        config=base_cfg,
        dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    base_model.config.layer_types = clean_layer_types
    if hasattr(base_model, "model") and hasattr(base_model.model, "config"):
        base_model.model.config.layer_types = clean_layer_types
    base_model.eval()
    D = base_model.config.hidden_size

    print(f"\n⏳ [2/4] Building stratified calibration set ({samples_per_dataset} samples x 5 domains)...")
    tokenized_domains, total_tokens = build_calibration_domains(tokenizer, samples_per_dataset=samples_per_dataset)
    print(f"  ✅ Collected {total_tokens} tokens (N_eff/D ≈ {(total_tokens * n_motifs) / D:.0f}x)")

    print("\n⏳ [3/4] Computing Cholesky whitening factors and decomposing 5 motifs...")
    S_op_in_pos, S_op_mid_pos = compute_cholesky_factors(
        base_model, tokenized_domains, p_layers=p_layers, n_motifs=n_motifs, motif_size=motif_size, device=device
    )
    del tokenized_domains
    gc.collect()
    torch.cuda.empty_cache()

    sd: dict[str, torch.Tensor] = {}
    pos_names = ["FullAttn (Pos 0)", "ShortConv (Pos 1)", "SlidingAttn (Pos 2)", "ShortConv (Pos 3)"]
    shared_idx = 0

    for pos in range(motif_size):
        is_attn = pos in (0, 2)
        S_in = S_op_in_pos[pos]
        S_mid = S_op_mid_pos[pos]
        post_op_w = [
            base_model.model.layers[p_layers + m * motif_size + pos].post_operator_norm.weight.detach().float()
            for m in range(n_motifs)
        ]
        pos_eng = []

        if is_attn:
            for p_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                W_l = []
                for m in range(n_motifs):
                    attn_mod = base_model.model.layers[p_layers + m * motif_size + pos].self_attn
                    W_l.append(getattr(attn_mod, p_name).weight.detach().float())
                    delattr(attn_mod, p_name)
                r_u = rank_kv if p_name in ("k_proj", "v_proj") else rank_attn
                S_use = S_mid if p_name == "o_proj" else S_in
                rw_l = post_op_w if p_name == "o_proj" else None
                W_sh, B_l, A_l, eng = decompose_5_motifs_rd(W_l, S_use, r_u, row_w_list=rw_l)
                del W_l
                pos_eng.append(eng)

                sd[f"model.core.shared.weights.{shared_idx}"] = W_sh.to(torch.bfloat16).contiguous().cpu()
                shared_idx += 1
                for m in range(n_motifs):
                    b_bal, a_bal = balance_ba(B_l[m], A_l[m])
                    prefix = f"model.core.phases.{m}.blocks.{pos}.self_attn.{p_name}"
                    sd[f"{prefix}.B"] = b_bal
                    sd[f"{prefix}.A"] = a_bal
                del W_sh, B_l, A_l
        else:
            W_in_l = []
            for m in range(n_motifs):
                conv_mod = base_model.model.layers[p_layers + m * motif_size + pos].conv
                W_in_l.append(conv_mod.in_proj.weight.detach().float())
                delattr(conv_mod, "in_proj")

            W_in_l = align_conv_bx_gauge(W_in_l, S_in, D)

            W_sh_chunks, B_chunks_m, A_chunks_m = [], [[] for _ in range(n_motifs)], [[] for _ in range(n_motifs)]
            for ch_i in range(3):
                W_ch = [W_in_l[m][ch_i * D : (ch_i + 1) * D, :] for m in range(n_motifs)]
                W_sh_c, B_c_l, A_c_l, eng = decompose_5_motifs_rd(W_ch, S_in, rank_conv)
                pos_eng.append(eng)
                W_sh_chunks.append(W_sh_c)
                for m in range(n_motifs):
                    b_bal, a_bal = balance_ba(B_c_l[m], A_c_l[m])
                    B_chunks_m[m].append(b_bal)
                    A_chunks_m[m].append(a_bal)
                del W_ch, W_sh_c, B_c_l, A_c_l
            del W_in_l

            sd[f"model.core.shared.weights.{shared_idx}"] = torch.cat(W_sh_chunks, dim=0).to(torch.bfloat16).contiguous().cpu()
            del W_sh_chunks
            shared_idx += 1
            for m in range(n_motifs):
                prefix = f"model.core.phases.{m}.blocks.{pos}.conv.in_proj"
                sd[f"{prefix}.B"] = torch.stack(B_chunks_m[m], dim=0).contiguous()
                sd[f"{prefix}.A"] = torch.stack(A_chunks_m[m], dim=0).contiguous()

            W_out_l = []
            for m in range(n_motifs):
                conv_mod = base_model.model.layers[p_layers + m * motif_size + pos].conv
                W_out_l.append(conv_mod.out_proj.weight.detach().float())
                delattr(conv_mod, "out_proj")
            W_sh_o, B_o_l, A_o_l, eng = decompose_5_motifs_rd(W_out_l, S_mid, rank_conv, row_w_list=post_op_w)
            del W_out_l
            pos_eng.append(eng)
            sd[f"model.core.shared.weights.{shared_idx}"] = W_sh_o.to(torch.bfloat16).contiguous().cpu()
            shared_idx += 1
            for m in range(n_motifs):
                b_bal, a_bal = balance_ba(B_o_l[m], A_o_l[m])
                prefix = f"model.core.phases.{m}.blocks.{pos}.conv.out_proj"
                sd[f"{prefix}.B"] = b_bal
                sd[f"{prefix}.A"] = a_bal
            del W_sh_o, B_o_l, A_o_l

        print(f"  • {pos_names[pos]}: retained {sum(pos_eng)/len(pos_eng):.2f}% spectral energy")

    del S_op_in_pos, S_op_mid_pos
    sd["model.embed_tokens.weight"] = base_model.model.embed_tokens.weight.detach().to(torch.bfloat16).contiguous().cpu()
    sd["model.embedding_norm.weight"] = base_model.model.embedding_norm.weight.detach().to(torch.bfloat16).contiguous().cpu()

    for idx in range(24):
        lyr = base_model.model.layers[idx]
        for k, v in lyr.state_dict().items():
            t_cpu = v.detach().to(torch.bfloat16).contiguous().cpu()
            if idx < p_layers:
                sd[f"model.prelude.{idx}.{k}"] = t_cpu
            elif idx < p_layers + n_motifs * motif_size:
                m, pos = divmod(idx - p_layers, motif_size)
                sd[f"model.core.phases.{m}.blocks.{pos}.{k}"] = t_cpu
            else:
                c_idx = idx - (p_layers + n_motifs * motif_size)
                sd[f"model.coda.{c_idx}.{k}"] = t_cpu
        base_model.model.layers[idx] = None

    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    sd.update(build_rd_controller_state_dict(D=D, n_motifs=n_motifs))

    print(f"\n💾 [4/4] Exporting checkpoint to `{output_dir}`...")
    os.makedirs(output_dir, exist_ok=True)
    for fname in [".gitattributes", "LICENSE", "chat_template.jinja", "lfm3_conv.py", "generation_config.json",
                  "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]:
        src_p = os.path.join(orig_snap_dir, fname)
        if os.path.exists(src_p):
            shutil.copy2(src_p, os.path.join(output_dir, fname))

    pkg_dir = Path(__file__).resolve().parent.parent / "lfm3_rd"
    shutil.copy2(pkg_dir / "configuration_lfm3_rd.py", os.path.join(output_dir, "configuration_lfm3_rd.py"))
    shutil.copy2(pkg_dir / "modeling_lfm3_rd.py", os.path.join(output_dir, "modeling_lfm3_rd.py"))

    with open(os.path.join(orig_snap_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.update({
        "architectures": ["Lfm3RDForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_lfm3_rd.Lfm3RDConfig",
            "AutoModel": "modeling_lfm3_rd.Lfm3RDModel",
            "AutoModelForCausalLM": "modeling_lfm3_rd.Lfm3RDForCausalLM",
        },
        "model_type": "lfm3_rd",
        "layer_types": clean_layer_types,
        "num_prelude_layers": p_layers,
        "num_motifs": n_motifs,
        "motif_size": motif_size,
        "num_coda_layers": 2,
        "rank_attn": rank_attn,
        "rank_conv": rank_conv,
        "rank_kv": rank_kv,
        "conv_lora_groups": 3,
        "reasoning_mode": "manual",
        "reasoning_effort": None,
        "recurrent_loops": 5,
        "min_recurrent_loops": 1,
        "max_recurrent_loops": 32,
        "act_halting_threshold": 0.80,
        "act_kl_prior_lambda": 0.20,
    })
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    safetensors_path = os.path.join(output_dir, "model.safetensors")
    save_file(sd, safetensors_path, metadata={"format": "pt"})
    n_params_b = sum(v.numel() for v in sd.values()) / 1e9
    sz_gb = os.path.getsize(safetensors_path) / (1024 ** 3)
    print(f"  ✅ Exported {len(sd)} tensors ({n_params_b:.3f}B params, {sz_gb:.2f} GB) in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LiquidAI/LFM3-1.7B to parameter-shared LFM3-1.7B-RD")
    parser.add_argument("--base-model", type=str, default="LiquidAI/LFM3-1.7B", help="Base model Hugging Face repo ID or local path")
    parser.add_argument("--output-dir", type=str, default="./LFM3-1.7B-RD-export", help="Output export directory")
    parser.add_argument("--rank-attn", type=int, default=640, help="Bottleneck rank for attention projections")
    parser.add_argument("--rank-conv", type=int, default=640, help="Bottleneck rank for short-conv projections")
    parser.add_argument("--rank-kv", type=int, default=512, help="Bottleneck rank for KV projections")
    parser.add_argument("--num-calib-samples", "--samples-per-dataset", dest="samples_per_dataset", type=int, default=60, help="Calibration samples per domain")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for calibration and SVD surgery")
    args = parser.parse_args()

    convert_lfm3_to_rd(
        base_model_id=args.base_model,
        output_dir=args.output_dir,
        rank_attn=args.rank_attn,
        rank_conv=args.rank_conv,
        rank_kv=args.rank_kv,
        samples_per_dataset=args.samples_per_dataset,
        device=args.device,
    )
