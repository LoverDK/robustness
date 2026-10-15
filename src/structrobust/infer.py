"""Batched greedy generation with Hugging Face transformers (optionally with a LoRA adapter)."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load_model(model_name: str = DEFAULT_MODEL, adapter: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    return tok, model


def chat(tok, prompt: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True)


def generate(tok, model, items: list[dict], out_path: Path, max_new_tokens: int = 512,
             batch_size: int = 48, shuffle_seed: int = 0, log_every: int = 5) -> list[dict]:
    """items: dicts with keys 'uid' and 'prompt'. Results are appended to out_path (resumable).

    Evaluation order is randomly shuffled so that no condition is systematically
    batched together; within a chunk items are length-sorted to reduce padding.
    """
    import torch

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["uid"]] = r
    todo = [it for it in items if it["uid"] not in done]
    random.Random(shuffle_seed).shuffle(todo)
    chunk = batch_size * 8
    t0 = time.time()
    n_tok = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for c0 in range(0, len(todo), chunk):
            block = sorted(todo[c0:c0 + chunk], key=lambda it: len(it["prompt"]))
            for b0 in range(0, len(block), batch_size):
                batch = block[b0:b0 + batch_size]
                texts = [chat(tok, it["prompt"]) for it in batch]
                enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
                with torch.no_grad():
                    out = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                                         pad_token_id=tok.pad_token_id, temperature=None,
                                         top_p=None, top_k=None)
                gen = out[:, enc["input_ids"].shape[1]:]
                for it, g in zip(batch, gen):
                    text = tok.decode(g, skip_special_tokens=True)
                    n_tok += int((g != tok.pad_token_id).sum())
                    rec = {"uid": it["uid"], "generation": text}
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    done[it["uid"]] = rec
                fh.flush()
            el = time.time() - t0
            print(f"[gen] {min(c0 + chunk, len(todo))}/{len(todo)}  {el:.0f}s  {n_tok / max(el, 1):.0f} tok/s",
                  flush=True)
    return [done[it["uid"]] for it in items]
