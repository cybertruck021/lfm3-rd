
# Copyright 2026 LFM3-RD Authors. All rights reserved.

"""Verification and benchmark suite for LFM3-1.7B-RD across recurrent depths T in [1..32] and Auto mode."""

import argparse
import sys
import time
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers.cache_utils import DynamicCache

def verify_model(model_path: str, device: str = "cuda"):

    print(f"🔍 Loading `{model_path}`...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(

        model_path, dtype=torch.bfloat16, device_map=device, trust_remote_code=True

    )

    model.eval()

    prompts = {

        "Easy Fact": [

            {"role": "user", "content": "What is the capital of France and what river runs through it?"}

        ],

        "Math System (Farmer)": [

            {"role": "user", "content": "A farmer has chickens and rabbits. In total, there are 37 heads and 98 legs. How many rabbits and how many chickens does he have? Solve step by step."}

        ],

        "Logic Trap (3 Boxes)": [

            {"role": "user", "content": (

                "There are three boxes: Red, Blue, and Green. Gold is in exactly one of them.\n"

                "- Red box: 'The gold is in this box.'\n"

                "- Blue box: 'The gold is not in this box.'\n"

                "- Green box: 'The gold is not in the Red box.'\n"

                "Exactly ONE statement is true. Which box has the gold?"

            )}

        ],

    }

    for name, msgs in prompts.items():

        use_think = name != "Easy Fact"

        prompt_str = tokenizer.apply_chat_template(

            msgs, enable_thinking=use_think, add_generation_prompt=True, tokenize=False

        )

        inp_ids = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

        print(f"\n==================== {name} ====================")

        for effort in ("fast", "medium", "high", "auto"):

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

            new_toks = gen_ids[0][inp_ids.shape[1]:]

            out_txt = tokenizer.decode(new_toks, skip_special_tokens=False).replace("<|im_end|>", "").strip()

            step_info = ""

            if effort == "auto" and model.model.act_step_history:

                hist = model.model.act_step_history

                step_info = f" | avg_steps={sum(hist)/len(hist):.2f} (min={min(hist)}, max={max(hist)}, trace={hist[:15]})"

            print(f"  • [{effort:6s} | {len(new_toks)/max(dt, 1e-6):5.1f} tok/s{step_info}]: {out_txt.replace(chr(10), ' ')[:220]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LFM3-1.7B-RD across recurrent steps and dynamic halting")
    parser.add_argument("--model-path", type=str, default="cybertruck32489/LFM3-1.7B-RD", help="Hugging Face repo ID or local checkpoint path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    args = parser.parse_args()
    verify_model(args.model_path, device=args.device)
