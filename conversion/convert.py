# Copyright 2026 LFM3-RD Authors. All rights reserved.
"""
CLI entrypoint to convert `LiquidAI/LFM3-1.7B` into lossless `LFM3-1.7B-RD`
(Prelude -> 5-Phase Recurrent Core -> Coda).

Usage:
    python -m conversion.convert --base-model LiquidAI/LFM3-1.7B --output-dir ./LFM3-1.7B-RD-export
"""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
import torch


def normalize_lfm3_layer_types(num_layers: int = 24) -> list[str]:
    return [
        "full_attention"
        if i % 4 == 2
        else "sliding_attention"
        if i % 4 == 0 and i > 0
        else "conv"
        for i in range(num_layers)
    ]


def build_rd_controller_state_dict(
    D: int = 2048, n_motifs: int = 5, router_dim: int = 128
) -> dict[str, torch.Tensor]:
    """Initializes the state_dict keys for the Lfm3RDController architecture."""
    torch.manual_seed(102)
    inj_rank = 64
    ref_rank = 64
    return {
        "model.rd_controller.act_norm.weight": torch.ones(
            D, dtype=torch.bfloat16
        ),
        "model.rd_controller.vel_norm.weight": torch.ones(
            D, dtype=torch.bfloat16
        ),
        "model.rd_controller.state_proj.weight": (
            torch.randn(router_dim, D) * 0.02
        ).to(torch.bfloat16),
        "model.rd_controller.vel_proj.weight": (
            torch.randn(router_dim, D) * 0.02
        ).to(torch.bfloat16),
        "model.rd_controller.dyn_proj.weight": (
            torch.randn(router_dim, 4) * 0.02
        ).to(torch.bfloat16),
        "model.rd_controller.dyn_proj.bias": torch.zeros(
            router_dim, dtype=torch.bfloat16
        ),
        "model.rd_controller.halting_head.weight": (
            torch.randn(1, router_dim) * 0.02
        ).to(torch.bfloat16),
        "model.rd_controller.halting_head.bias": torch.tensor(
            [-0.85], dtype=torch.bfloat16
        ),
        "model.rd_controller.step_scales": torch.ones(32, dtype=torch.bfloat16),
        "model.rd_controller.inj_norm.weight": torch.ones(
            D, dtype=torch.bfloat16
        ),
        "model.rd_controller.inj_A": (
            torch.randn(inj_rank, D) / math.sqrt(D)
        ).to(torch.bfloat16),
        "model.rd_controller.inj_B": torch.zeros(
            (D, inj_rank), dtype=torch.bfloat16
        ),
        "model.rd_controller.inj_gates": torch.full(
            (6,), 0.20, dtype=torch.bfloat16
        ),
        "model.rd_controller.ref_norm.weight": torch.ones(
            D, dtype=torch.bfloat16
        ),
        "model.rd_controller.ref_A": (
            torch.randn(n_motifs, ref_rank, D) / math.sqrt(D)
        ).to(torch.bfloat16),
        "model.rd_controller.ref_B": torch.zeros(
            (n_motifs, D, ref_rank), dtype=torch.bfloat16
        ),
        "model.rd_controller.refine_gates": torch.zeros(
            32, dtype=torch.bfloat16
        ),
        "model.rd_controller.refine_token_proj.weight": torch.zeros(
            (1, D), dtype=torch.bfloat16
        ),
        "model.rd_controller.refine_token_proj.bias": torch.zeros(
            1, dtype=torch.bfloat16
        ),
    }


def convert_lfm3_to_rd(
    base_model_id: str = "LiquidAI/LFM3-1.7B",
    output_dir: str = "./LFM3-1.7B-RD-export",
):
    t0 = time.time()
    p_layers, n_motifs, motif_size = 2, 5, 4

    print(f"⏳ [1/3] Downloading base model `{base_model_id}`...")
    orig_snap_dir = snapshot_download(repo_id=base_model_id)

    with open(
        os.path.join(orig_snap_dir, "config.json"), "r", encoding="utf-8"
    ) as f:
        cfg = json.load(f)

    num_layers = cfg.get("num_hidden_layers", 24)
    D = cfg.get("hidden_size", 2048)
    clean_layer_types = normalize_lfm3_layer_types(num_layers)

    print(
        "\n⏳ [2/3] Remapping 24 layers into Prelude (2) -> Core (5x4) -> Coda (2)..."
    )
    sd: dict[str, torch.Tensor] = {}
    src_safetensors = os.path.join(orig_snap_dir, "model.safetensors")

    with safe_open(src_safetensors, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key).to(torch.bfloat16).contiguous()
            if key in ("model.embed_tokens.weight", "model.embedding_norm.weight"):
                sd[key] = tensor
            elif key.startswith("model.layers."):
                parts = key.split(".")
                idx = int(parts[2])
                sub_key = ".".join(parts[3:])
                if idx < p_layers:
                    sd[f"model.prelude.{idx}.{sub_key}"] = tensor
                elif idx < p_layers + n_motifs * motif_size:
                    m, pos = divmod(idx - p_layers, motif_size)
                    sd[f"model.core.phases.{m}.blocks.{pos}.{sub_key}"] = tensor
                else:
                    c_idx = idx - (p_layers + n_motifs * motif_size)
                    sd[f"model.coda.{c_idx}.{sub_key}"] = tensor
            elif key == "lm_head.weight" and not cfg.get(
                "tie_word_embeddings", True
            ):
                sd[key] = tensor

    sd.update(build_rd_controller_state_dict(D=D, n_motifs=n_motifs))
    gc.collect()

    print(f"\n💾 [3/3] Exporting checkpoint to `{output_dir}`...")
    os.makedirs(output_dir, exist_ok=True)
    for fname in [
        ".gitattributes",
        "LICENSE",
        "chat_template.jinja",
        "lfm3_conv.py",
        "generation_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]:
        src_p = os.path.join(orig_snap_dir, fname)
        if os.path.exists(src_p):
            shutil.copy2(src_p, os.path.join(output_dir, fname))

    pkg_dir = Path(__file__).resolve().parent.parent / "lfm3_rd"
    for mod_file in ("__init__.py", "configuration_lfm3_rd.py", "modeling_lfm3_rd.py"):
        if (pkg_dir / mod_file).exists():
            shutil.copy2(pkg_dir / mod_file, os.path.join(output_dir, mod_file))

    for legacy_key in (
        "rank_attn",
        "rank_conv",
        "rank_kv",
        "conv_lora_groups",
        "act_kl_prior_lambda",
    ):
        cfg.pop(legacy_key, None)

    cfg.update(
        {
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
            "reasoning_mode": "manual",
            "reasoning_effort": None,
            "recurrent_loops": 5,
            "min_recurrent_loops": 1,
            "max_recurrent_loops": 32,
            "act_halting_threshold": 0.80,
            "mono_loss_weight": 0.25,
            "settle_loss_weight": 0.10,
            "halt_loss_weight": 0.50,
        }
    )
    with open(
        os.path.join(output_dir, "config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    safetensors_path = os.path.join(output_dir, "model.safetensors")
    save_file(sd, safetensors_path, metadata={"format": "pt"})
    n_params_b = sum(v.numel() for v in sd.values()) / 1e9
    sz_gb = os.path.getsize(safetensors_path) / (1024**3)
    print(
        f"  ✅ Exported {len(sd)} tensors ({n_params_b:.3f}B params, {sz_gb:.2f} GB) in {time.time() - t0:.1f}s!"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert LiquidAI/LFM3-1.7B to lossless LFM3-1.7B-RD"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="LiquidAI/LFM3-1.7B",
        help="Base model Hugging Face repo ID or local path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./LFM3-1.7B-RD-export",
        help="Output export directory",
    )
    args, _ = parser.parse_known_args()

    convert_lfm3_to_rd(
        base_model_id=args.base_model,
        output_dir=args.output_dir,
    )