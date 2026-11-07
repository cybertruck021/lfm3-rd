# Copyright 2026 LFM3-RD Authors. All rights reserved.

"""Verification and benchmark suite for LFM3-1.7B-RD across recurrent depths T in [1..32] and Auto mode."""

import argparse
from collections import Counter
import math
from pathlib import Path
import sys
import time

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch
import torch.nn.functional as F
import transformers.utils.import_utils as _tiu

_tiu._torchaudio_available = False
_tiu._torchvision_available = False
_tiu.is_torchaudio_available = lambda: False
_tiu.is_torchvision_available = lambda: False

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


def verify_model(model_path: str, device: str = "cuda"):
    print(f"🔍 Loading `{model_path}` on `{device}`...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    use_bf16 = device == "cpu" or (
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    )
    model_dtype = torch.bfloat16 if use_bf16 else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=model_dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    print("\n" + "=" * 110)
    print("STAGE 1: ARCHITECTURE & PARAMETER AUDIT")
    print("=" * 110)
    unique_params = sum(p.numel() for p in model.parameters())
    ctrl_params = sum(p.numel() for p in model.model.rd_controller.parameters())
    print(f"  • Model Class            : {type(model).__name__} ({model_dtype})")
    print(f"  • Total Physical Params  : {unique_params / 1e6:.2f}M")
    print(f"  • RD Controller Params   : {ctrl_params / 1e6:.2f}M")
    print(
        f"  • Structure              : Prelude({len(model.model.prelude)}) -> Core({len(model.model.core.phases)}x4) -> Coda({len(model.model.coda)})"
    )

    print("\n" + "=" * 110)
    print("STAGE 2: DEPTH CONVERGENCE & SETTLING TRAJECTORY (T in [1..32])")
    print("=" * 110)
    eval_text = (
        "The capital of France is Paris, and the Seine river flows through the city. "
        "In mathematics, the derivative of sin(x) with respect to x is cos(x), "
        "and a system of linear equations can be solved via Gaussian elimination."
    )
    eval_ids = tokenizer(eval_text, return_tensors="pt")["input_ids"].to(
        model.device
    )

    with torch.inference_mode():
        ref_out = model(
            input_ids=eval_ids,
            labels=eval_ids,
            reasoning_mode="manual",
            recurrent_loops=5,
            use_cache=False,
        )
        ref_logits = ref_out.logits.float()
        ref_top1 = ref_logits.argmax(dim=-1)
        ref_probs = F.softmax(ref_logits, dim=-1)

        print(
            f"  {'Depth T':<10} | {'Loss':>8} | {'PPL':>9} | {'KL vs T=5':>10} | {'Top-1 vs T=5':>13} | {'Logit Cos':>10}"
        )
        print("  " + "-" * 72)
        for t_steps in (1, 2, 3, 4, 5, 8, 12, 16, 32):
            out_t = model(
                input_ids=eval_ids,
                labels=eval_ids,
                reasoning_mode="manual",
                recurrent_loops=t_steps,
                use_cache=False,
            )
            logits_t = out_t.logits.float()
            loss_t = out_t.loss.item()
            ppl_t = math.exp(min(loss_t, 20.0))
            kl_t = F.kl_div(
                F.log_softmax(logits_t, dim=-1),
                ref_probs,
                reduction="batchmean",
            ).item()
            top1_agr = (
                (logits_t.argmax(dim=-1) == ref_top1).float().mean().item()
                * 100.0
            )
            cos_t = (
                F.cosine_similarity(
                    logits_t.reshape(-1, logits_t.size(-1)),
                    ref_logits.reshape(-1, ref_logits.size(-1)),
                    dim=-1,
                )
                .mean()
                .item()
            )
            print(
                f"  T = {t_steps:<6d} | {loss_t:8.4f} | {ppl_t:9.3f} | {kl_t:10.4f} | {top1_agr:12.1f}% | {cos_t:10.5f}"
            )

    print("\n" + "=" * 110)
    print("STAGE 3: DYNAMIC CACHE & VARIABLE-DEPTH TRANSITION AUDIT")
    print("=" * 110)
    with torch.inference_mode():
        cache = DynamicCache(config=model.config)
        prefix_ids = eval_ids[:, :10]
        _ = model(
            input_ids=prefix_ids,
            past_key_values=cache,
            use_cache=True,
            reasoning_mode="manual",
            recurrent_loops=5,
        )
        transition_schedule = [1, 2, 5, 12, 3, 8]
        for idx, t_var in enumerate(transition_schedule):
            next_tok = eval_ids[:, 10 + idx : 11 + idx]
            step_out = model(
                input_ids=next_tok,
                past_key_values=cache,
                use_cache=True,
                reasoning_mode="manual",
                recurrent_loops=t_var,
            )
            if (
                torch.isnan(step_out.logits).any()
                or torch.isinf(step_out.logits).any()
            ):
                raise RuntimeError(
                    f"NaN/Inf detected during variable-depth cache transition at T={t_var}!"
                )
    print(
        f"  ✅ Variable-depth cache transition {transition_schedule} passed with zero NaN/Inf."
    )

    prompts = {
        "Easy Fact": [
            {
                "role": "user",
                "content": (
                    "What is the capital of France and what river runs through"
                    " it?"
                ),
            }
        ],
        "Math System (Farmer)": [
            {
                "role": "user",
                "content": (
                    "A farmer has chickens and rabbits. In total, there are 37"
                    " heads and 98 legs. How many rabbits and how many chickens"
                    " does he have? Solve step by step."
                ),
            }
        ],
        "Logic Trap (3 Boxes)": [
            {
                "role": "user",
                "content": (
                    "There are three boxes: Red, Blue, and Green. Gold is in"
                    " exactly one of them.\n- Red box: 'The gold is in this"
                    " box.'\n- Blue box: 'The gold is not in this box.'\n-"
                    " Green box: 'The gold is not in the Red box.'\nExactly ONE"
                    " statement is true. Which box has the gold?"
                ),
            }
        ],
    }

    for name, msgs in prompts.items():
        use_think = name != "Easy Fact"
        prompt_str = tokenizer.apply_chat_template(
            msgs,
            enable_thinking=use_think,
            add_generation_prompt=True,
            tokenize=False,
        )
        inp_ids = tokenizer(
            prompt_str, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(model.device)

        print(f"\n==================== {name} ====================")
        for effort in ("low", "fast", "medium", "high", "auto"):
            model.model.act_step_history.clear()
            t0 = time.time()
            with torch.inference_mode():
                gen_ids = model.generate(
                    input_ids=inp_ids,
                    max_new_tokens=180 if use_think else 60,
                    do_sample=False,
                    reasoning_effort=effort,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            dt = time.time() - t0
            new_toks = gen_ids[0][inp_ids.shape[1] :]
            out_txt = (
                tokenizer.decode(new_toks, skip_special_tokens=False)
                .replace("<|im_end|>", "")
                .strip()
            )

            step_info = ""
            if effort == "auto" and model.model.act_step_history:
                hist = model.model.act_step_history
                counts = dict(sorted(Counter(hist).items()))
                step_info = (
                    f" | avg={sum(hist)/len(hist):.2f} (min={min(hist)},"
                    f" max={max(hist)}, dist={counts}, trace={hist[:12]})"
                )

            print(
                f"  • [{effort:6s} | {len(new_toks)/max(dt, 1e-6):5.1f}"
                f" tok/s{step_info}]: {out_txt.replace(chr(10), ' ')[:200]}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate LFM3-1.7B-RD across recurrent steps and dynamic halting"
        )
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./LFM3-1.7B-RD-export",
        help="Hugging Face repo ID or local checkpoint path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    args = parser.parse_args()
    verify_model(args.model_path, device=args.device)