
# Copyright 2026 LFM3-RD Authors. All rights reserved.

"""Stratified multi-domain activation covariance collection and Cholesky factor computation."""

import gc

import json

import re

import urllib.parse

import urllib.request

from typing import Callable

import torch

from datasets import load_dataset

def render_lfm3_chat(

    tokenizer,

    messages: list[dict],

    tools: list[dict] | None = None,

    enable_thinking: bool = False,

    preserve_thinking: bool = False,

    add_generation_prompt: bool = False,

) -> str:

    return tokenizer.apply_chat_template(

        messages,

        tools=tools,

        enable_thinking=enable_thinking,

        preserve_thinking=preserve_thinking,

        add_generation_prompt=add_generation_prompt,

        tokenize=False,

    )

def fetch_hf_rows(

    dataset_id: str,

    config: str = "default",

    split: str = "train",

    n: int = 60,

    revision: str | None = None,

) -> list[dict]:

    rows = []

    try:

        q = urllib.parse.urlencode({

            "dataset": dataset_id,

            "config": config,

            "split": split,

            "offset": 0,

            "length": min(n + 15, 100),

        })

        req = urllib.request.Request(

            f"https://datasets-server.huggingface.co/rows?{q}",

            headers={"User-Agent": "Mozilla/5.0"},

        )

        with urllib.request.urlopen(req, timeout=15) as resp:

            data = json.loads(resp.read().decode("utf-8"))

            rows = [item["row"] for item in data.get("rows", [])]

    except Exception:

        pass

    if len(rows) < n:

        try:

            kw = {"split": split, "streaming": True}

            if revision:

                kw["revision"] = revision

            ds = load_dataset(dataset_id, config if config != "default" else None, **kw)

            for r in ds:

                rows.append(dict(r))

                if len(rows) >= n + 10:

                    break

            del ds

        except Exception as e:

            print(f"  [Calibration Warning] Fallback on {dataset_id}: {e}")

    gc.collect()

    return rows

def build_calibration_domains(

    tokenizer,

    samples_per_dataset: int = 60,

    max_seq_len: int = 512,

) -> tuple[dict[str, tuple[float, list[torch.Tensor]]], int]:

    """Loads 5 stratified calibration domains and returns tokenized input_id tensors."""

    def load_alpaca(n: int) -> list[str]:

        texts = []

        for row in fetch_hf_rows("yahma/alpaca-cleaned", "default", "train", n):

            instr = (row.get("instruction") or "").strip()

            inp = (row.get("input") or "").strip()

            out = (row.get("output") or "").strip()

            if not instr or not out:

                continue

            user_msg = f"{instr}\n\n{inp}" if inp else instr

            texts.append(render_lfm3_chat(tokenizer, [{"role": "user", "content": user_msg}, {"role": "assistant", "content": out}]))

            if len(texts) >= n:

                break

        return texts

    def load_openhermes(n: int) -> list[str]:

        texts = []

        role_map = {"system": "system", "human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}

        for row in fetch_hf_rows("teknium/OpenHermes-2.5", "default", "train", n, revision="refs/convert/parquet"):

            msgs = []

            for turn in (row.get("conversations") or []):

                r = role_map.get(turn.get("from", ""))

                val = (turn.get("value") or "").strip()

                if r and val:

                    msgs.append({"role": r, "content": val})

            if len(msgs) >= 2 and any(m["role"] == "assistant" for m in msgs):

                try:

                    texts.append(render_lfm3_chat(tokenizer, msgs))

                except Exception:

                    continue

            if len(texts) >= n:

                break

        return texts

    def load_gsm8k(n: int) -> list[str]:

        texts = []

        for row in fetch_hf_rows("openai/gsm8k", "main", "train", n):

            q = (row.get("question") or "").strip()

            a = (row.get("answer") or "").strip()

            if not q or not a:

                continue

            if "####" in a:

                think_part, final_ans = [x.strip() for x in a.split("####", 1)]

                clean_ans = f"{think_part}\n\nTherefore, the final answer is **{final_ans}**."

            else:

                think_part, clean_ans = a, a

            texts.append(render_lfm3_chat(

                tokenizer,

                [{"role": "user", "content": q}, {"role": "assistant", "thinking": think_part, "content": clean_ans}],

                enable_thinking=True,

                preserve_thinking=True,

            ))

            if len(texts) >= n:

                break

        return texts

    def load_opencode(n: int) -> list[str]:

        texts = []

        for row in fetch_hf_rows("marcuscedricridia/OpenCodeInstruct-1000-sample", "default", "train", n):

            inp = (row.get("input") or row.get("instruction") or row.get("prompt") or "").strip()

            out = (row.get("output") or row.get("response") or "").strip()

            if not inp or not out:

                continue

            texts.append(render_lfm3_chat(tokenizer, [{"role": "user", "content": inp}, {"role": "assistant", "content": out}]))

            if len(texts) >= n:

                break

        return texts

    def load_hermes_tools(n: int) -> list[str]:

        texts = []

        think_re = re.compile(r"<think>(.*?)</think>", re.DOTALL)

        tc_re = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

        for row in fetch_hf_rows("interstellarninja/hermes_reasoning_tool_use", "default", "train", n):

            raw_tools = row.get("tools")

            try:

                tools_list = json.loads(raw_tools) if isinstance(raw_tools, str) else raw_tools

                if not isinstance(tools_list, list) or not tools_list:

                    continue

            except Exception:

                continue

            msgs = []

            for turn in (row.get("conversations") or []):

                frm = turn.get("from", "")

                val = (turn.get("value") or "").strip()

                if frm == "system":

                    msgs.append({"role": "system", "content": "You are a helpful AI assistant with access to tools."})

                elif frm in ("human", "user"):

                    msgs.append({"role": "user", "content": val})

                elif frm in ("gpt", "assistant"):

                    th_m = think_re.search(val)

                    thinking = th_m.group(1).strip() if th_m else ""

                    tcs = []

                    for tc_str in tc_re.findall(val):

                        try:

                            tc_obj = json.loads(tc_str)

                            fn_name = tc_obj.get("name")

                            fn_args = tc_obj.get("arguments", {})

                            if isinstance(fn_args, str):

                                fn_args = json.loads(fn_args) if fn_args.strip() else {}

                            if fn_name and isinstance(fn_args, dict):

                                tcs.append({"name": fn_name, "arguments": fn_args})

                        except Exception:

                            pass

                    rest = tc_re.sub("", think_re.sub("", val)).strip()

                    m_dict = {"role": "assistant", "content": rest}

                    if thinking:

                        m_dict["thinking"] = thinking

                    if tcs:

                        m_dict["tool_calls"] = tcs

                    msgs.append(m_dict)

                elif frm in ("tool", "observation"):

                    msgs.append({"role": "tool", "content": val})

            if len(msgs) >= 2 and any(m["role"] == "assistant" for m in msgs):

                try:

                    texts.append(render_lfm3_chat(tokenizer, msgs, tools=tools_list, enable_thinking=True, preserve_thinking=True))

                except Exception:

                    continue

            if len(texts) >= n:

                break

        return texts

    loaders: dict[str, tuple[float, Callable[[int], list[str]]]] = {

        "1_alpaca_instruct":    (0.20, load_alpaca),

        "2_openhermes_general": (0.20, load_openhermes),

        "3_gsm8k_math_think":   (0.20, load_gsm8k),

        "4_opencode_instruct":  (0.20, load_opencode),

        "5_hermes_tool_use":    (0.20, load_hermes_tools),

    }

    tokenized_domains: dict[str, tuple[float, list[torch.Tensor]]] = {}

    total_tokens = 0

    for d_name, (w_d, loader_fn) in loaders.items():

        samples = loader_fn(samples_per_dataset)

        id_list = [

            tokenizer(txt, return_tensors="pt", truncation=True, max_length=max_seq_len, add_special_tokens=False)["input_ids"]

            for txt in samples if len(txt) > 16

        ]

        n_tok = sum(x.shape[1] for x in id_list)

        tokenized_domains[d_name] = (w_d, id_list)

        total_tokens += n_tok

        print(f"  • {d_name:22s}: {len(id_list):3d} samples | {n_tok:6d} tokens")

    return tokenized_domains, total_tokens

@torch.no_grad()

def compute_cholesky_factors(

    base_model,

    tokenized_domains: dict[str, tuple[float, list[torch.Tensor]]],

    p_layers: int = 2,

    n_motifs: int = 5,

    motif_size: int = 4,

    damping: float = 1.5e-3,

    device: str = "cuda",

) -> tuple[list[torch.Tensor], list[torch.Tensor]]:

    """

    Collects stratified second-moment matrices E[x x^T] across the 5 motifs and returns

    lower-triangular Cholesky whitening factors S_in[pos] and S_mid[pos] for each motif position pos in 0..3.

    """

    D = base_model.config.hidden_size

    cov_in_total = [torch.zeros((D, D), dtype=torch.float32, device=device) for _ in range(motif_size)]

    cov_mid_total = [torch.zeros((D, D), dtype=torch.float32, device=device) for _ in range(motif_size)]

    cov_in_dom = [torch.zeros((D, D), dtype=torch.float32, device=device) for _ in range(motif_size)]

    cov_mid_dom = [torch.zeros((D, D), dtype=torch.float32, device=device) for _ in range(motif_size)]

    def make_in_hook(pos_idx: int):

        def _hook(mod, inp, out):

            x = out.detach().reshape(-1, D).float()

            cov_in_dom[pos_idx].add_(x.T @ x)

        return _hook

    def make_mid_hook(pos_idx: int):

        def _hook(mod, inp, out):

            x = inp[0].detach().reshape(-1, D).float()

            cov_mid_dom[pos_idx].add_(x.T @ x)

        return _hook

    hooks = []

    try:

        for m in range(n_motifs):

            for pos in range(motif_size):

                lyr = base_model.model.layers[p_layers + m * motif_size + pos]

                hooks.append(lyr.operator_norm.register_forward_hook(make_in_hook(pos)))

                target_proj = lyr.self_attn.o_proj if pos in (0, 2) else lyr.conv.out_proj

                hooks.append(target_proj.register_forward_hook(make_mid_hook(pos)))

        for _, (w_d, id_list) in tokenized_domains.items():

            for pos in range(motif_size):

                cov_in_dom[pos].zero_()

                cov_mid_dom[pos].zero_()

            n_dom_tok = max(1, sum(x.shape[1] for x in id_list))

            for cid in id_list:

                _ = base_model(input_ids=cid.to(device), use_cache=False, logits_to_keep=1)

            scale = w_d / (float(n_motifs) * n_dom_tok)

            for pos in range(motif_size):

                cov_in_total[pos].add_(cov_in_dom[pos] * scale)

                cov_mid_total[pos].add_(cov_mid_dom[pos] * scale)

    finally:

        for h in hooks:

            h.remove()

    S_in_pos, S_mid_pos = [], []

    eye_D = torch.eye(D, dtype=torch.float32, device=device)

    for pos in range(motif_size):

        C_in = cov_in_total[pos]

        S_in_pos.append(torch.linalg.cholesky(C_in + damping * torch.diagonal(C_in).mean() * eye_D))

        C_mid = cov_mid_total[pos]

        S_mid_pos.append(torch.linalg.cholesky(C_mid + damping * torch.diagonal(C_mid).mean() * eye_D))

    return S_in_pos, S_mid_pos
