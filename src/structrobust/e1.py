"""E1 pilot: equal-budget SFT on depth-2 seen tasks.

Conditions share one latent pool, two supervised sequences per latent, and the
same optimizer-step schedule. Evaluation uses the same chat template as
training. Shifted cells are the generator's reversed presentation.

Training permutations for B2/B3/Ours shuffle every core and other rule.
Depth 2 only has two core orders inside the generator's fixed other-rule
slots, so that narrower shuffle cannot supply K=8 distinct presentations.
The generator itself is unchanged, and the test cells still use it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import logic
from .burden import burden, burden_score
from .prompts import _norm_num, arith_prompt, extract_number, extract_yes_no, logic_prompt
from .stats import mean_ci

PROJECT = Path(__file__).resolve().parents[2]
PILOT = PROJECT / "experiments" / "e1" / "pilot"
BUILD = PROJECT / "build" / "e1" / "pilot"
LOG_PATH = PROJECT / "build" / "logs" / "e1_pilot.log"

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 0
DEPTH = 2
N_TRAIN = 500
N_VAL = 200
N_TEST = 200
N_GSM = 500
K_MINE = 8
MAX_LEN = 1024
MAX_NEW_TOKENS = 512
LR = 2e-4
WARMUP_FRAC = 0.03
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.0
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
EFFECTIVE_BATCH = 16
MICRO_BATCH = 4
LAMBDA_CONSISTENCY = 0.1
VAL_WORSE_MEAN = 0.05
TIME_LIMIT_S = 6 * 3600
TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)
CONDITIONS = ("b0", "b1", "b2", "b3", "ours")
MIX_CONDITIONS = ("b1", "b2", "b3", "ours")
GSM_MIX_FRACTION = 0.15
GSM_HELDOUT_PATH = PROJECT / "experiments" / "e1" / "pilot" / "gsm8k_gold.json"
_CALC = re.compile(r"<<[^>]*>>")


class PilotStop(Exception):
    """Stop the pilot without writing a successful metrics file."""


def log(msg: str) -> None:
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + msg
    print(line, flush=True)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _rules(rules) -> list:
    return [[list(premises), conclusion] for premises, conclusion in rules]


def graph_hash(lat: logic.Latent) -> str:
    payload = {
        "family": lat.family,
        "depth": lat.depth,
        "label": lat.label,
        "entity": lat.entity,
        "facts": list(lat.facts),
        "core_rules": _rules(lat.core_rules),
        "other_rules": _rules(lat.other_rules),
        "extra_rules": _rules(lat.extra_distractors),
        "extra_facts": list(lat.meta.get("extra_facts", ())),
        "query": lat.query,
        "vocab": lat.vocab,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _mean(xs) -> float:
    return float(sum(xs) / len(xs))


def zscore(values: list[float]) -> list[float]:
    if not values:
        return []
    mu = _mean(values)
    var = sum((v - mu) ** 2 for v in values) / len(values)
    if var < 1e-12:
        return [0.0] * len(values)
    sd = math.sqrt(var)
    return [(v - mu) / sd for v in values]


def rank_presentations(nlls: list[float], burdens: list[float]) -> tuple[list[float], list[int]]:
    """Higher score is mined first. Ties break toward higher burden, then lower index."""
    if len(nlls) != len(burdens):
        raise ValueError("nll and burden lists differ in length")
    scores = [a + b for a, b in zip(zscore(nlls), zscore(burdens))]
    order = sorted(range(len(scores)), key=lambda i: (scores[i], burdens[i], -i), reverse=True)
    return scores, order


def optimizer_step_plan(n_seq: int = N_TRAIN * 2, effective: int = EFFECTIVE_BATCH):
    """Every sequence is used once. The final step is shorter when n_seq is not divisible."""
    sizes = [effective] * (n_seq // effective)
    if n_seq % effective:
        sizes.append(n_seq % effective)
    remine_after = len(sizes) // 2
    return sizes, remine_after


def warmup_steps(n_steps: int) -> int:
    return max(1, round(WARMUP_FRAC * n_steps))


def lr_at(step: int, n_steps: int, warmup: int, base: float = LR) -> float:
    if step < warmup:
        return base * float(step + 1) / float(warmup)
    progress = float(step - warmup) / float(max(1, n_steps - warmup))
    return base * max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))


def render_order(lat: logic.Latent, order: list[tuple[str, int]]) -> dict:
    lookup = {"core": lat.core_rules, "other": lat.other_rules}
    fact_lines = [f"{lat.entity} is {fact}." for fact in lat.facts]
    rule_lines = [logic.rule_text(lat.entity, lookup[kind][idx]) for kind, idx in order]
    text = (
        "Facts:\n" + "\n".join(fact_lines) + "\n\nRules:\n" + "\n".join(rule_lines)
        + f"\n\nQuestion: Is it true that {lat.entity} is {lat.query}?"
    )
    core_pos = {idx: pos for pos, (kind, idx) in enumerate(order) if kind == "core"}
    positions = [core_pos[i] for i in range(len(lat.core_rules))]
    features = burden(lat, positions)
    return {
        "text": text,
        "core_positions": positions,
        "burden": features,
        "burden_score": burden_score(features),
        "order": order,
    }


def canonical_render(lat: logic.Latent) -> dict:
    rendered = logic.render(lat, "aligned")
    features = burden(lat, rendered["core_positions"])
    return {
        "text": rendered["text"],
        "core_positions": list(rendered["core_positions"]),
        "burden": features,
        "burden_score": burden_score(features),
        "order": None,
    }


def sample_renders(lat: logic.Latent, k: int, key: str) -> list[dict]:
    items = [("core", i) for i in range(len(lat.core_rules))]
    items += [("other", j) for j in range(len(lat.other_rules))]
    out, seen = [], set()
    attempt = 0
    while len(out) < k:
        rng = random.Random(logic.stable_seed(lat.task_id, key, attempt))
        order = items.copy()
        rng.shuffle(order)
        rendered = render_order(lat, order)
        attempt += 1
        if rendered["text"] in seen:
            if attempt > max(2000, k * 50):
                raise RuntimeError(f"{lat.task_id} produced only {len(out)} unique presentations")
            continue
        seen.add(rendered["text"])
        out.append(rendered)
    return out


def training_renders(lat: logic.Latent, condition: str) -> tuple[dict, dict]:
    if condition == "b1":
        aligned = canonical_render(lat)
        return aligned, dict(aligned)
    if condition in ("b2", "b3"):
        drawn = sample_renders(lat, 2, "b2-random")
        return drawn[0], drawn[1]
    raise ValueError(condition)


def slot_random_unique(lat: logic.Latent, n_seeds: int = 16) -> int:
    return len({logic.render(lat, "random", perm_seed=i)["text"] for i in range(n_seeds)})


def _audit_pool(pool, vocab: str, families) -> dict:
    labels = [lat.label for lat in pool]
    graphs = [graph_hash(lat) for lat in pool]
    return {
        "n": len(pool),
        "n_unique_graph": len(set(graphs)),
        "n_unique_task_id": len({lat.task_id for lat in pool}),
        "n_yes": labels.count("yes"),
        "n_no": labels.count("no"),
        "depths": sorted({lat.depth for lat in pool}),
        "families": sorted({lat.family for lat in pool}),
        "vocabs": sorted({lat.vocab for lat in pool}),
        "expected_vocab": vocab,
        "expected_families": list(families),
    }


def build_pools(seed: int = SEED):
    n_seen = len(logic.SEEN_FAMILIES)
    n_unseen = len(logic.UNSEEN_FAMILIES)
    if N_TRAIN % n_seen or N_VAL % n_seen or N_TEST % n_seen or N_TEST % n_unseen:
        raise RuntimeError("pool size is not divisible by the number of families")
    train = logic.make_pool(logic.SEEN_FAMILIES, [DEPTH], N_TRAIN // n_seen, seed, "train", "e1train")
    val = logic.make_pool(logic.SEEN_FAMILIES, [DEPTH], N_VAL // n_seen, seed, "train", "e1val")
    test_seen = logic.make_pool(logic.SEEN_FAMILIES, [DEPTH], N_TEST // n_seen, seed, "eval", "e1test")
    test_unseen = logic.make_pool(
        logic.UNSEEN_FAMILIES, [DEPTH], N_TEST // n_unseen, seed, "eval", "e1test_unseen",
    )
    e0 = logic.make_pool(logic.SEEN_FAMILIES, [DEPTH], N_TEST // n_seen, 0, "eval", "e0")
    pools = {
        "train": train,
        "val": val,
        "test_seen": test_seen,
        "test_unseen": test_unseen,
        "e0_seen_depth2": e0,
    }
    summary = isolation_summary(pools)
    if not summary["ok"]:
        raise RuntimeError("pool isolation failed: " + json.dumps(summary["overlaps"]))
    kept = {name: pools[name] for name in ("train", "val", "test_seen", "test_unseen")}
    return kept, summary


def isolation_summary(pools: dict) -> dict:
    graphs = {name: {graph_hash(lat) for lat in pool} for name, pool in pools.items()}
    pairs = (
        ("train", "val"),
        ("train", "test_seen"),
        ("train", "test_unseen"),
        ("val", "test_seen"),
        ("val", "test_unseen"),
        ("train", "e0_seen_depth2"),
        ("val", "e0_seen_depth2"),
        ("test_seen", "e0_seen_depth2"),
        ("test_unseen", "e0_seen_depth2"),
        ("test_seen", "test_unseen"),
    )
    overlaps = {f"{a}__{b}": len(graphs[a] & graphs[b]) for a, b in pairs if a in graphs and b in graphs}
    audits = {
        "train": _audit_pool(pools["train"], "train", logic.SEEN_FAMILIES),
        "val": _audit_pool(pools["val"], "train", logic.SEEN_FAMILIES),
        "test_seen": _audit_pool(pools["test_seen"], "eval", logic.SEEN_FAMILIES),
        "test_unseen": _audit_pool(pools["test_unseen"], "eval", logic.UNSEEN_FAMILIES),
    }
    if "e0_seen_depth2" in pools:
        audits["e0_seen_depth2"] = _audit_pool(pools["e0_seen_depth2"], "eval", logic.SEEN_FAMILIES)
    slot_counts = [slot_random_unique(lat) for lat in pools["train"]]
    ok = all(v == 0 for v in overlaps.values())
    for audit in audits.values():
        ok = ok and audit["n"] == audit["n_unique_graph"] == audit["n_unique_task_id"]
        ok = ok and audit["n_yes"] == audit["n_no"] == audit["n"] // 2
        ok = ok and audit["depths"] == [DEPTH]
        ok = ok and audit["families"] == sorted(audit["expected_families"])
        ok = ok and audit["vocabs"] == [audit["expected_vocab"]]
    expected_n = {"train": N_TRAIN, "val": N_VAL, "test_seen": N_TEST, "test_unseen": N_TEST}
    for name, n in expected_n.items():
        ok = ok and audits[name]["n"] == n
    return {
        "ok": ok,
        "seed": SEED,
        "depth": DEPTH,
        "tags": {
            "train": "e1train",
            "val": "e1val",
            "test_seen": "e1test",
            "test_unseen": "e1test_unseen",
            "excluded_e0_tag": "e0",
        },
        "overlaps": overlaps,
        "pools": audits,
        "shifted_presentation": "reversed",
        "slot_random_unique_min": min(slot_counts),
        "slot_random_unique_mean": _mean(slot_counts),
        "slot_random_unique_max": max(slot_counts),
        "slot_random_seeds_per_latent": 16,
        "training_permutation": (
            "B1 uses logic.render aligned twice. B2 and B3 share two uniform "
            "shuffles of all core and other rules. Ours draws K=8 such shuffles. "
            "Facts stay fixed. Extra distractors are not used in training."
        ),
    }


def build_eval_items(pool, split: str) -> tuple[list[dict], dict]:
    items, meta = [], {}
    for lat in pool:
        digest = graph_hash(lat)
        for presentation, suffix in (("aligned", "aligned"), ("reversed", "shifted")):
            rendered = logic.render(lat, presentation)
            features = burden(lat, rendered["core_positions"])
            cell = f"{split}_{suffix}"
            uid = f"{cell}|{digest}"
            items.append({"uid": uid, "prompt": logic_prompt(rendered["text"])})
            meta[uid] = {
                "uid": uid,
                "cell": cell,
                "presentation": presentation,
                "label": lat.label,
                "family": lat.family,
                "depth": lat.depth,
                "vocab": lat.vocab,
                "task_id": lat.task_id,
                "graph_hash": digest,
                "back_reads": features["back_reads"],
                "n_core": features["n_core"],
            }
    return items, meta


def _as_ids(encoded) -> list[int]:
    ids = encoded["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def yes_no_token_ids(tok) -> tuple[int, int]:
    yes = tok.encode(" yes", add_special_tokens=False)
    no = tok.encode(" no", add_special_tokens=False)
    if len(yes) != 1 or len(no) != 1:
        raise RuntimeError(f"yes/no are not single tokens: {yes} {no}")
    return yes[0], no[0]


def encode_chat_trace(tok, user_text: str, trace: str) -> dict:
    """Tokenize the eval chat string plus a gold trace. Loss starts at the assistant tokens."""
    from .infer import chat

    prompt_str = chat(tok, user_text)
    full_str = tok.apply_chat_template(
        [{"role": "user", "content": user_text}, {"role": "assistant", "content": trace}],
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_ids = _as_ids(tok(prompt_str))
    full_ids = _as_ids(tok(full_str))
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            f"train/eval token prefix mismatch (prompt {len(prompt_ids)}, full {len(full_ids)})"
        )
    if len(full_ids) > MAX_LEN:
        raise RuntimeError(f"sequence length {len(full_ids)} exceeds {MAX_LEN}")
    if len(full_ids) < 2:
        raise RuntimeError("supervised sequence is too short")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {
        "input_ids": full_ids,
        "labels": labels,
        "answer_pos": len(full_ids) - 1,
        "prompt_len": len(prompt_ids),
    }


def encode_supervised(tok, user_text: str, trace: str, label: str, yes_id: int, no_id: int) -> dict:
    encoded = encode_chat_trace(tok, user_text, trace)
    full_ids = encoded["input_ids"]
    labels = encoded["labels"]
    gold = yes_id if label == "yes" else no_id
    answer_pos = None
    for i in range(len(full_ids) - 1, encoded["prompt_len"] - 1, -1):
        if full_ids[i] in (yes_id, no_id):
            answer_pos = i
            break
    if answer_pos is None or full_ids[answer_pos] != gold:
        raise RuntimeError("gold answer token was not found after the prompt")
    if labels[answer_pos] != gold:
        raise RuntimeError("answer token is masked out of the loss")
    return {"input_ids": full_ids, "labels": labels, "answer_pos": answer_pos}


def examples_from_renders(tok, lat: logic.Latent, renders: list[dict], yes_id: int, no_id: int) -> list[dict]:
    trace = logic.solver_trace(lat)
    if not trace.endswith(f"Answer: {lat.label}"):
        raise RuntimeError(f"solver trace does not end with the declared label {lat.label}")
    user_cache = {}
    out = []
    for rendered in renders:
        text = rendered["text"]
        if text not in user_cache:
            user_cache[text] = encode_supervised(
                tok, logic_prompt(text), trace, lat.label, yes_id, no_id,
            )
        base = user_cache[text]
        out.append({
            "input_ids": list(base["input_ids"]),
            "labels": list(base["labels"]),
            "answer_pos": base["answer_pos"],
            "task_id": lat.task_id,
            "graph_hash": graph_hash(lat),
            "burden_score": float(rendered["burden_score"]),
            "text": text,
            "label": lat.label,
        })
    return out


def paired_stream(pairs: list[tuple[dict, dict]], seed: int) -> list[dict]:
    rng = random.Random(seed)
    order = list(range(len(pairs)))
    rng.shuffle(order)
    seq = []
    for i in order:
        left, right = pairs[i]
        if left["task_id"] != right["task_id"]:
            raise RuntimeError("a training pair mixes two latent tasks")
        seq.append(left)
        seq.append(right)
    return seq


def _quiet_hf() -> None:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        from transformers.utils.logging import disable_progress_bar
        disable_progress_bar()
    except Exception:
        pass


def _install_log() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    class _Tee:
        def __init__(self, stream, path):
            self.stream = stream
            self.file = open(path, "a", encoding="utf-8", buffering=1)

        def write(self, data):
            self.stream.write(data)
            self.file.write(data)
            self.flush()
            return len(data)

        def flush(self):
            self.stream.flush()
            self.file.flush()

        def isatty(self):
            return False

    sys.stdout = _Tee(sys.stdout, LOG_PATH)
    sys.stderr = _Tee(sys.stderr, LOG_PATH)


def _finite(value) -> bool:
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or exc.__class__.__name__ == "OutOfMemoryError"


def _free(model) -> None:
    import gc
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_trainable(quant: str):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.padding_side = "right"
    if quant == "nf4":
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=cfg, device_map="cuda",
        )
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    elif quant == "bf16":
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cuda")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        raise ValueError(quant)
    model.config.use_cache = False
    if tok.pad_token_id is not None:
        model.config.pad_token_id = tok.pad_token_id
    lora = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=list(TARGET_MODULES),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)
    model.train()
    return tok, model


def load_for_eval(adapter: str | None):
    from .infer import load_model

    try:
        tok, model = load_model(MODEL_NAME, adapter)
    except Exception as exc:
        if not adapter:
            raise
        log(f"merged eval load failed ({type(exc).__name__}: {exc}); loading the adapter without merging")
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cuda")
        model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        model.config.use_cache = True
        del torch
    return tok, model


def collate(examples: list[dict], pad_id: int, device):
    import torch

    width = max(len(ex["input_ids"]) for ex in examples)
    batch = len(examples)
    ids = torch.full((batch, width), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((batch, width), dtype=torch.long, device=device)
    labels = torch.full((batch, width), -100, dtype=torch.long, device=device)
    answer = torch.zeros(batch, dtype=torch.long, device=device)
    for i, ex in enumerate(examples):
        n = len(ex["input_ids"])
        ids[i, :n] = torch.tensor(ex["input_ids"], dtype=torch.long, device=device)
        attn[i, :n] = 1
        labels[i, :n] = torch.tensor(ex["labels"], dtype=torch.long, device=device)
        answer[i] = ex["answer_pos"]
    return ids, attn, labels, answer


def forward_loss(model, tok, examples, lam: float, yes_id: int, no_id: int):
    import torch
    import torch.nn.functional as F

    if len(examples) % 2:
        raise RuntimeError("training batch is not a set of pairs")
    for i in range(0, len(examples), 2):
        if examples[i]["task_id"] != examples[i + 1]["task_id"]:
            raise RuntimeError("microbatch broke a latent-task pair")
    ids, attn, labels, answer = collate(examples, tok.pad_token_id, "cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(input_ids=ids, attention_mask=attn, labels=labels, use_cache=False)
        sft = out.loss
        kl = torch.zeros((), device=sft.device, dtype=sft.dtype)
        if lam:
            row = torch.arange(ids.size(0), device=sft.device)
            chosen = out.logits[row, answer - 1]
            pair = torch.stack((chosen[:, yes_id], chosen[:, no_id]), dim=-1).float()
            logp = F.log_softmax(pair, dim=-1)
            left, right = logp[0::2], logp[1::2]
            kl_value = (left.exp() * (left - right)).sum(-1) + (right.exp() * (right - left)).sum(-1)
            kl = kl_value.mean().to(dtype=sft.dtype)
        loss = sft + (lam * kl)
    return loss, sft.detach(), kl.detach()


def score_nll(model, tok, flat: list[dict], batch_size: int) -> list[float]:
    import torch
    import torch.nn.functional as F

    model.eval()
    values = [None] * len(flat)
    sizes = [batch_size, max(1, batch_size // 2), 1]
    # Try the requested batch, then smaller ones, without dropping candidates.
    cursor = 0
    bs_index = 0
    with torch.no_grad():
        while cursor < len(flat):
            bs = sizes[min(bs_index, len(sizes) - 1)]
            batch = flat[cursor:cursor + bs]
            try:
                ids, attn, labels, _ = collate(batch, tok.pad_token_id, "cuda")
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=ids, attention_mask=attn, use_cache=False).logits
                shift_logits = logits[:, :-1, :].float()
                shift_labels = labels[:, 1:]
                token_loss = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    reduction="none",
                    ignore_index=-100,
                ).view_as(shift_labels)
                mask = shift_labels.ne(-100)
                denom = mask.sum(dim=1).clamp(min=1)
                nll = ((token_loss * mask).sum(dim=1) / denom).tolist()
                for offset, value in enumerate(nll):
                    values[cursor + offset] = float(value)
                cursor += len(batch)
            except RuntimeError as exc:
                if not _is_oom(exc) or bs == 1:
                    raise
                log(f"mining OOM at batch {bs}; retrying smaller")
                torch.cuda.empty_cache()
                bs_index += 1
    model.train()
    model.config.use_cache = False
    if any(v is None or not _finite(v) for v in values):
        raise PilotStop("mining NLL was missing or not finite")
    return values


def select_pairs(candidates: list[list[dict]], nll_rows: list[list[float]]):
    pairs = []
    selected_burden, all_burden, selected_nll = [], [], []
    for examples, nlls in zip(candidates, nll_rows):
        burdens = [ex["burden_score"] for ex in examples]
        _, order = rank_presentations(nlls, burdens)
        chosen = (examples[order[0]], examples[order[1]])
        pairs.append(chosen)
        selected_burden.extend(ex["burden_score"] for ex in chosen)
        selected_nll.extend(nlls[i] for i in order[:2])
        all_burden.extend(burdens)
    stats = {
        "selected_burden_mean": _mean(selected_burden),
        "candidate_burden_mean": _mean(all_burden),
        "selected_nll_mean": _mean(selected_nll),
    }
    return pairs, stats


def _condition_pairs(tok, pool, condition: str, yes_id: int, no_id: int):
    pairs = []
    for lat in pool:
        left, right = training_renders(lat, condition)
        examples = examples_from_renders(tok, lat, [left, right], yes_id, no_id)
        pairs.append((examples[0], examples[1]))
    return pairs


def _ours_candidates(tok, pool, yes_id: int, no_id: int):
    groups = []
    for lat in pool:
        renders = sample_renders(lat, K_MINE, "bgm-k8")
        groups.append(examples_from_renders(tok, lat, renders, yes_id, no_id))
    return groups


def gsm_mix_counts(n_synthetic: int = N_TRAIN * 2, fraction: float = GSM_MIX_FRACTION) -> dict:
    """Keep every synthetic sequence. Add the even GSM8K count closest to `fraction` of the total."""
    if n_synthetic < 2 or not 0 < fraction < 1:
        raise ValueError("mix counts need a positive synthetic budget and a fraction inside (0, 1)")
    target = n_synthetic * fraction / (1.0 - fraction)
    nearest = int(round(target))
    even_choices = [n for n in range(max(2, nearest - 2), nearest + 3) if n % 2 == 0]
    n_seq = min(even_choices, key=lambda n: (abs(n - target), n))
    total = n_synthetic + n_seq
    return {
        "n_synthetic_sequences": n_synthetic,
        "n_gsm_sequences": n_seq,
        "n_gsm_questions": n_seq // 2,
        "n_total_sequences": total,
        "gsm_fraction": n_seq / total,
        "target_gsm_sequences": target,
        "fraction_requested": fraction,
    }


def normalize_question(text: str) -> str:
    return " ".join(text.split())


def gsm_supervised_trace(answer: str) -> str:
    """Drop calculator markup and end on the same Answer line the eval prompt requests."""
    gold = gsm_gold(answer)
    body = answer.split("####", 1)[0]
    body = _CALC.sub("", body)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in body.splitlines()]
    body = "\n".join(line for line in lines if line).strip()
    if not body:
        raise RuntimeError("GSM8K rationale is empty")
    return f"{body}\nAnswer: {gold}"


def select_gsm_train_indices(questions, blocked, n_questions: int, seed: int, accept=None) -> list[int]:
    """Sample train-split rows. `blocked` is held-out question text, not test-split index numbers."""
    if n_questions < 1:
        raise ValueError(n_questions)
    blocked_norm = {normalize_question(q) for q in blocked}
    rng = random.Random(seed)
    order = list(range(len(questions)))
    rng.shuffle(order)
    chosen = []
    for index in order:
        if normalize_question(questions[index]) in blocked_norm:
            continue
        if accept is not None and not accept(index):
            continue
        chosen.append(index)
        if len(chosen) == n_questions:
            return sorted(chosen)
    raise RuntimeError(f"only {len(chosen)} GSM8K train questions were usable; needed {n_questions}")


def held_out_question_set(test_dataset, gold: dict) -> set[str]:
    indices = [int(i) for i in gold["indices"]]
    if gold.get("split") != "test" or gold.get("n") != N_GSM or len(indices) != N_GSM or len(gold["gold"]) != N_GSM:
        raise RuntimeError("held-out GSM8K file is not the saved 500-question test index")
    if len(set(indices)) != N_GSM:
        raise RuntimeError("held-out GSM8K indices are not unique")
    return {normalize_question(test_dataset[i]["question"]) for i in indices}


def encode_gsm_example(tok, question: str, answer: str, task_id: str) -> dict:
    trace = gsm_supervised_trace(answer)
    gold = gsm_gold(answer)
    if not trace.endswith(f"Answer: {gold}"):
        raise RuntimeError("GSM8K trace does not end with the gold answer")
    encoded = encode_chat_trace(tok, arith_prompt(question), trace)
    return {
        "input_ids": list(encoded["input_ids"]),
        "labels": list(encoded["labels"]),
        "answer_pos": encoded["answer_pos"],
        "task_id": task_id,
        "graph_hash": hashlib.sha256(normalize_question(question).encode("utf-8")).hexdigest(),
        "burden_score": 0.0,
        "text": question,
        "label": "gsm8k",
    }


def gsm_training_pair(example: dict) -> tuple[dict, dict]:
    """Two identical copies, so the pair batch and the consistency term stay well-defined."""
    left = {
        "input_ids": list(example["input_ids"]),
        "labels": list(example["labels"]),
        "answer_pos": example["answer_pos"],
        "task_id": example["task_id"],
        "graph_hash": example["graph_hash"],
        "burden_score": 0.0,
        "text": example["text"],
        "label": "gsm8k",
    }
    right = dict(left)
    right["input_ids"] = list(left["input_ids"])
    right["labels"] = list(left["labels"])
    return left, right


def _with_extra(pairs, extra_pairs):
    if not extra_pairs:
        return pairs
    return list(pairs) + list(extra_pairs)


def train_condition(cond: str, pool, quant: str, micro: int, extra_pairs=None) -> dict:
    import torch

    n_extra = 0 if not extra_pairs else 2 * len(extra_pairs)
    sizes, remine_after = optimizer_step_plan(N_TRAIN * 2 + n_extra)
    if any(n % micro != 0 for n in sizes):
        raise RuntimeError(f"micro-batch {micro} does not divide the step sizes {sizes}")
    if extra_pairs:
        for left, right in extra_pairs:
            if left["task_id"] != right["task_id"] or left["input_ids"] != right["input_ids"]:
                raise RuntimeError("GSM8K training pair is not two copies of one train question")
    set_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    tok, model = load_trainable(quant)
    yes_id, no_id = yes_no_token_ids(tok)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(
        f"{cond} trainable tensors {n_trainable} quant {quant} micro {micro} "
        f"sequences synthetic {N_TRAIN * 2} gsm {n_extra} steps {len(sizes)}"
    )
    mine_seconds = 0.0
    mine_stats = []
    selections = []
    t0 = time.perf_counter()

    if cond == "ours":
        candidates = _ours_candidates(tok, pool, yes_id, no_id)

        def current_pairs(round_id: int):
            nonlocal mine_seconds
            started = time.perf_counter()
            flat = [ex for group in candidates for ex in group]
            nll_flat = score_nll(model, tok, flat, batch_size=8)
            rows, cursor = [], 0
            for group in candidates:
                rows.append(nll_flat[cursor:cursor + len(group)])
                cursor += len(group)
            pairs, stats = select_pairs(candidates, rows)
            mine_seconds += time.perf_counter() - started
            stats["round"] = round_id
            stats["seconds"] = mine_seconds
            mine_stats.append(stats)
            selections.append([(a["text"], b["text"]) for a, b in pairs])
            log(
                f"{cond} mine round {round_id} selected burden {stats['selected_burden_mean']:.4f} "
                f"vs candidate {stats['candidate_burden_mean']:.4f}"
            )
            return pairs

        pairs = _with_extra(current_pairs(0), extra_pairs)
        stream_seed = logic.stable_seed(cond, "shuffle", 0)
    else:
        pairs = _with_extra(_condition_pairs(tok, pool, cond, yes_id, no_id), extra_pairs)
        stream_seed = logic.stable_seed(cond, "shuffle", 0)

    seq = paired_stream(pairs, stream_seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    warmup = warmup_steps(len(sizes))
    log_path = BUILD / cond / "train_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    consumed_tokens = 0
    consumed_seqs = 0
    cursor = 0
    records = []
    with log_path.open("w", encoding="utf-8") as handle:
        for step, n_seq in enumerate(sizes):
            if time.perf_counter() - t0 > TIME_LIMIT_S:
                raise PilotStop(f"{cond} training exceeded 6 hours at step {step}")
            if cond == "ours" and step == remine_after:
                pairs = _with_extra(current_pairs(1), extra_pairs)
                seq = paired_stream(pairs, logic.stable_seed(cond, "shuffle", 1))
                cursor = 0
            chunk = seq[cursor:cursor + n_seq]
            cursor += n_seq
            if len(chunk) != n_seq:
                raise RuntimeError(f"{cond} ran out of sequences at step {step}")
            n_micro = n_seq // micro
            opt.zero_grad(set_to_none=True)
            sft_sum = 0.0
            kl_sum = 0.0
            for m in range(n_micro):
                batch = chunk[m * micro:(m + 1) * micro]
                loss, sft, kl = forward_loss(model, tok, batch, LAMBDA_CONSISTENCY if cond == "b3" else 0.0, yes_id, no_id)
                if not _finite(loss):
                    raise PilotStop(f"{cond} loss is not finite at step {step}")
                (loss / n_micro).backward()
                sft_sum += float(sft)
                kl_sum += float(kl)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            if not _finite(grad_norm):
                raise PilotStop(f"{cond} gradient is not finite at step {step}")
            lr = lr_at(step, len(sizes), warmup)
            for group in opt.param_groups:
                group["lr"] = lr
            opt.step()
            for ex in chunk:
                consumed_tokens += sum(token != -100 for token in ex["labels"])
                consumed_seqs += 1
            rec = {
                "step": step,
                "sft": sft_sum / n_micro,
                "kl": kl_sum / n_micro,
                "loss": (sft_sum + (LAMBDA_CONSISTENCY if cond == "b3" else 0.0) * kl_sum) / n_micro,
                "lr": lr,
                "grad_norm": float(grad_norm),
                "seconds": time.perf_counter() - t0,
            }
            records.append(rec)
            handle.write(json.dumps(rec) + "\n")
            handle.flush()
            if step % 5 == 0 or step + 1 == len(sizes):
                log(
                    f"{cond} step {step + 1}/{len(sizes)} loss {rec['loss']:.4f} "
                    f"sft {rec['sft']:.4f} kl {rec['kl']:.4f} lr {lr:.6g}"
                )
    if consumed_seqs != sum(sizes):
        raise PilotStop(f"{cond} consumed {consumed_seqs} sequences, expected {sum(sizes)}")
    adapter = BUILD / cond / "adapter"
    if adapter.exists():
        shutil.rmtree(adapter)
    model.save_pretrained(adapter)
    # Adapter only: peft save_pretrained does not write the base weights.
    peak = int(torch.cuda.max_memory_allocated())
    _free(model)
    changed = None
    if len(selections) == 2:
        changed = sum(a != b for a, b in zip(selections[0], selections[1]))
    return {
        "train_seconds": time.perf_counter() - t0,
        "mine_seconds": mine_seconds,
        "peak_vram_bytes": peak,
        "steps": len(records),
        "step_sizes": sizes,
        "remine_after_steps": remine_after if cond == "ours" else None,
        "micro_batch": micro,
        "grad_accum_full_step": EFFECTIVE_BATCH // micro,
        "loss_first": records[0]["loss"],
        "loss_last": records[-1]["loss"],
        "sft_first": records[0]["sft"],
        "sft_last": records[-1]["sft"],
        "kl_last": records[-1]["kl"],
        "consumed_sequences": consumed_seqs,
        "consumed_supervised_tokens": consumed_tokens,
        "n_trainable": n_trainable,
        "mine_stats": mine_stats,
        "n_latents_selection_changed": changed,
        "quant": quant,
        "adapter": str(adapter.relative_to(PROJECT)).replace("\\", "/"),
        "n_synthetic_sequences": N_TRAIN * 2,
        "n_gsm_sequences": n_extra,
        "n_total_sequences": sum(sizes),
    }


def run_generate(tok, model, items: list[dict], path: Path, seed: int) -> int:
    import torch
    from .infer import generate

    path.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for batch_size in (32, 16, 8, 4):
        try:
            generate(
                tok, model, items, path,
                max_new_tokens=MAX_NEW_TOKENS,
                batch_size=batch_size,
                shuffle_seed=seed,
            )
            return batch_size
        except RuntimeError as exc:
            last = exc
            if not _is_oom(exc) or batch_size == 4:
                raise
            log(f"eval OOM at batch {batch_size}: {exc}")
            torch.cuda.empty_cache()
    raise last


def read_jsonl(path: Path) -> dict:
    found = {}
    if not path.exists():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            found[row["uid"]] = row
    return found


def score_logic(meta: dict, gens: dict) -> dict:
    missing = [uid for uid in meta if uid not in gens]
    if missing:
        raise PilotStop(f"logic eval missing {len(missing)} generations")
    grouped = defaultdict(list)
    for uid, row in meta.items():
        pred = extract_yes_no(gens[uid]["generation"])
        grouped[row["cell"]].append({
            "correct": int(pred == row["label"]),
            "pred": pred,
            "label": row["label"],
            "presentation": row["presentation"],
            "back_reads": row["back_reads"],
            "graph_hash": row["graph_hash"],
            "family": row["family"],
        })
    cells = {}
    for cell, rows in sorted(grouped.items()):
        acc, lo, hi = mean_ci([r["correct"] for r in rows], seed=SEED)
        cells[cell] = {
            "acc": acc,
            "lo": lo,
            "hi": hi,
            "n": len(rows),
            "n_yes": sum(r["label"] == "yes" for r in rows),
            "n_no": sum(r["label"] == "no" for r in rows),
            "extract_fail_rate": float(np.mean([r["pred"] == "" for r in rows])),
            "pred_yes_rate": float(np.mean([r["pred"] == "yes" for r in rows])),
            "presentation": rows[0]["presentation"],
            "back_reads_mean": float(np.mean([r["back_reads"] for r in rows])),
        }
    return cells


def gsm_gold(answer: str) -> str:
    if "####" not in answer:
        raise RuntimeError("GSM8K answer has no #### delimiter")
    number = _norm_num(answer.split("####")[-1])
    if number == "":
        raise RuntimeError("GSM8K gold number did not parse")
    return number


def _load_hf_split(split: str):
    from datasets import load_dataset

    errors = []

    def attempt(label: str):
        for name in ("openai/gsm8k", "gsm8k"):
            try:
                return load_dataset(name, "main", split=split), f"{label}|{name}"
            except Exception as exc:
                message = f"{label} {name} split={split}: {type(exc).__name__}: {exc}"
                errors.append(message)
                log(f"GSM8K download failed ({message})")
        return None

    found = attempt(os.environ.get("HF_ENDPOINT", "hub-default"))
    if found:
        return found[0], found[1], errors
    if os.environ.get("HF_ENDPOINT") != "https://hf-mirror.com":
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        found = attempt("https://hf-mirror.com")
        if found:
            return found[0], found[1], errors
    return None, None, errors


def load_gsm8k():
    dataset, label, errors = _load_hf_split("test")
    if dataset is not None:
        return dataset, label
    note = {
        "status": "not_downloaded",
        "reason": "GSM8K could not be downloaded. Synthetic evaluation continues. No GSM8K score is reported.",
        "attempts": errors,
    }
    write_json(PILOT / "gsm8k_unavailable.json", note)
    return None, note


def gsm_index_and_gold(dataset) -> dict:
    path = PILOT / "gsm8k_gold.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if len(saved["indices"]) != N_GSM or len(saved["gold"]) != N_GSM:
            raise RuntimeError("existing GSM8K index does not have 500 items")
        return saved
    rng = random.Random(SEED)
    indices = sorted(rng.sample(range(len(dataset)), N_GSM))
    gold = [gsm_gold(dataset[i]["answer"]) for i in indices]
    saved = {"indices": indices, "gold": gold, "n": N_GSM, "split": "test", "seed": SEED}
    write_json(path, saved)
    return saved


def score_gsm(golds: list[str], gens: dict, uids: list[str]) -> dict:
    missing = [uid for uid in uids if uid not in gens]
    if missing:
        raise PilotStop(f"GSM8K eval missing {len(missing)} generations")
    correct, fail = [], 0
    for uid, gold in zip(uids, golds):
        pred = extract_number(gens[uid]["generation"])
        if pred == "":
            fail += 1
        correct.append(int(pred == gold))
    acc, lo, hi = mean_ci(correct, seed=SEED)
    return {
        "acc": acc,
        "lo": lo,
        "hi": hi,
        "n": len(correct),
        "extract_fail_rate": fail / len(correct),
        "n_extract_fail": fail,
    }


def _cell_mean(metrics: dict, keys: tuple[str, str]) -> float:
    return 0.5 * (metrics["val"][keys[0]]["acc"] + metrics["val"][keys[1]]["acc"])


def maybe_lambda_note(b3_metrics: dict) -> None:
    b2_path = PILOT / "b2" / "metrics.json"
    if not b2_path.exists():
        log("B2 metrics missing, so the lambda note was not written")
        return
    b2 = json.loads(b2_path.read_text(encoding="utf-8"))
    b2_mean = _cell_mean(b2, ("val_aligned", "val_shifted"))
    b3_mean = _cell_mean(b3_metrics, ("val_aligned", "val_shifted"))
    gap = b2_mean - b3_mean
    comparison = {
        "lambda": LAMBDA_CONSISTENCY,
        "threshold_mean_acc": VAL_WORSE_MEAN,
        "b2_val_mean_acc": b2_mean,
        "b3_val_mean_acc": b3_mean,
        "b2_minus_b3": gap,
        "b2_val": b2["val"],
        "b3_val": b3_metrics["val"],
        "lambda_0_5_ran": False,
    }
    b3_metrics["lambda_comparison"] = comparison
    if gap >= VAL_WORSE_MEAN:
        comparison["note"] = (
            f"λ=0.1 的验证集 aligned/shifted 平均准确率比 B2 低 {gap:.4f}，"
            "达到预先写下的 0.05 阈值。没有跑 λ=0.5。"
        )
        write_json(PILOT / "b3_lambda_note.json", comparison)
        log("B3 validation is worse than B2 by the preset margin; lambda 0.5 was not run")
    else:
        log(f"B3 validation mean minus comparison gap {gap:.4f}; lambda 0.5 was not run")


def run_condition(cond: str, pools, dataset, gsm_saved: dict, quant: str, extra_pairs=None, mix_note=None) -> None:
    import torch

    t0 = time.perf_counter()
    out_dir = BUILD / cond
    out_dir.mkdir(parents=True, exist_ok=True)
    done_path = out_dir / "TRAIN_DONE.json"
    train_info = {"train_seconds": 0.0, "peak_vram_bytes": 0, "adapter": None, "steps": 0}
    if cond != "b0":
        if done_path.exists():
            train_info = json.loads(done_path.read_text(encoding="utf-8"))
            log(f"{cond} training already finished; resuming at evaluation")
        else:
            micro = MICRO_BATCH
            try:
                train_info = train_condition(cond, pools["train"], quant, micro, extra_pairs)
            except RuntimeError as exc:
                if not _is_oom(exc):
                    raise
                log(f"{cond} CUDA OOM during training; restarting this condition at micro-batch 2")
                torch.cuda.empty_cache()
                try:
                    train_info = train_condition(cond, pools["train"], quant, micro=2, extra_pairs=extra_pairs)
                    train_info["micro_batch_retry"] = True
                except RuntimeError as second:
                    if _is_oom(second):
                        raise PilotStop(f"{cond} CUDA OOM again at micro-batch 2") from second
                    raise
            train_info["micro_batch_retry"] = train_info.get("micro_batch_retry", False)
            write_json(done_path, train_info)
            log(f"{cond} training saved {train_info['adapter']}")
    if time.perf_counter() - t0 > TIME_LIMIT_S:
        raise PilotStop(f"{cond} exceeded 6 hours before evaluation finished")

    torch.cuda.reset_peak_memory_stats()
    eval_t0 = time.perf_counter()
    adapter = None if cond == "b0" else train_info["adapter"]
    adapter_abs = None if adapter is None else str(PROJECT / adapter)
    tok, model = load_for_eval(adapter_abs)

    logic_items, logic_meta = [], {}
    for split, pool in (("seen", pools["test_seen"]), ("unseen", pools["test_unseen"])):
        items, meta = build_eval_items(pool, split)
        logic_items.extend(items)
        logic_meta.update(meta)
    val_items, val_meta = build_eval_items(pools["val"], "val")
    # Val cells are named val_aligned / val_shifted by build_eval_items.
    logic_bs = run_generate(tok, model, logic_items, out_dir / "logic.jsonl", SEED)
    if time.perf_counter() - t0 > TIME_LIMIT_S:
        raise PilotStop(f"{cond} exceeded 6 hours during logic eval")
    val_bs = run_generate(tok, model, val_items, out_dir / "val.jsonl", SEED + 1)
    if time.perf_counter() - t0 > TIME_LIMIT_S:
        raise PilotStop(f"{cond} exceeded 6 hours during validation eval")
    gsm_bs = None
    gsm_path = None
    if dataset is None or gsm_saved is None:
        unavailable = PILOT / "gsm8k_unavailable.json"
        reason = "GSM8K was not downloaded."
        if unavailable.exists():
            reason = json.loads(unavailable.read_text(encoding="utf-8")).get("reason", reason)
        gsm = {
            "status": "not_downloaded",
            "acc": None,
            "n": None,
            "reason": reason,
            "file": "experiments/e1/pilot/gsm8k_unavailable.json",
        }
    else:
        gsm_items = []
        for index in gsm_saved["indices"]:
            gsm_items.append({
                "uid": f"gsm8k|{index}",
                "prompt": arith_prompt(dataset[int(index)]["question"]),
            })
        gsm_path = out_dir / "gsm8k.jsonl"
        gsm_bs = run_generate(tok, model, gsm_items, gsm_path, SEED + 2)
        gsm = score_gsm(gsm_saved["gold"], read_jsonl(gsm_path), [it["uid"] for it in gsm_items])
        gsm["status"] = "scored"
    cells = score_logic(logic_meta, read_jsonl(out_dir / "logic.jsonl"))
    val_cells = score_logic(val_meta, read_jsonl(out_dir / "val.jsonl"))
    expected = {"seen_aligned", "seen_shifted", "unseen_aligned", "unseen_shifted"}
    if set(cells) != expected or any(cells[name]["n"] < N_TEST for name in expected):
        raise PilotStop(f"{cond} logic cells incomplete: { {k: v['n'] for k, v in cells.items()} }")
    eval_peak = int(torch.cuda.max_memory_allocated())
    _free(model)
    metrics = {
        "condition": cond,
        "status": "complete",
        "seed": SEED,
        "depth": DEPTH,
        "model": MODEL_NAME,
        "quant": quant if cond != "b0" else "bf16-eval",
        "adapter": adapter,
        "train_seconds": train_info.get("train_seconds", 0.0),
        "mine_seconds": train_info.get("mine_seconds", 0.0),
        "eval_seconds": time.perf_counter() - eval_t0,
        "wall_seconds": time.perf_counter() - t0,
        "peak_vram_bytes": max(int(train_info.get("peak_vram_bytes", 0)), eval_peak),
        "peak_vram_mib": max(int(train_info.get("peak_vram_bytes", 0)), eval_peak) / (1024 ** 2),
        "train_peak_vram_mib": int(train_info.get("peak_vram_bytes", 0)) / (1024 ** 2),
        "eval_peak_vram_mib": eval_peak / (1024 ** 2),
        "train": {k: v for k, v in train_info.items() if k != "peak_vram_bytes"},
        "cells": cells,
        "val": val_cells,
        "gsm8k": gsm,
        "eval_batch_size": {"logic": logic_bs, "val": val_bs, "gsm8k": gsm_bs},
        "decoding": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS},
        "files": {
            "logic_generations": str((out_dir / "logic.jsonl").relative_to(PROJECT)).replace("\\", "/"),
            "val_generations": str((out_dir / "val.jsonl").relative_to(PROJECT)).replace("\\", "/"),
            "gsm8k_generations": None if gsm_path is None else str(gsm_path.relative_to(PROJECT)).replace("\\", "/"),
            "train_log": None if cond == "b0" else str((out_dir / "train_log.jsonl").relative_to(PROJECT)).replace("\\", "/"),
            "train_done": None if cond == "b0" else str(done_path.relative_to(PROJECT)).replace("\\", "/"),
        },
    }
    if mix_note is not None:
        metrics["gsm_mix"] = mix_note
    if cond == "b3":
        maybe_lambda_note(metrics)
    write_json(PILOT / cond / "metrics.json", metrics)
    if gsm.get("status") == "scored":
        gsm_bit = f" gsm8k={gsm['acc']:.4f}"
    else:
        gsm_bit = " gsm8k=not_downloaded"
    log(
        f"CONDITION_DONE {cond} cells "
        + " ".join(f"{name}={metrics['cells'][name]['acc']:.4f}" for name in sorted(expected))
        + gsm_bit
    )


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True,
        ).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def package_versions() -> dict:
    import torch
    import transformers
    import peft

    versions = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        import bitsandbytes
        versions["bitsandbytes"] = bitsandbytes.__version__
    except Exception as exc:
        versions["bitsandbytes"] = f"import failed: {exc}"
    return versions


def smoke_quant(quant: str) -> dict:
    import torch

    set_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    pools, _ = build_pools()
    tok, model = load_trainable(quant)
    yes_id, no_id = yes_no_token_ids(tok)
    lat = pools["train"][0]
    rendered = canonical_render(lat)
    examples = examples_from_renders(tok, lat, [rendered, rendered], yes_id, no_id)
    loss, sft, kl = forward_loss(model, tok, examples, 0.0, yes_id, no_id)
    if not _finite(loss):
        raise RuntimeError(f"{quant} smoke loss is not finite: {float(loss)}")
    loss.backward()
    adapter = BUILD / "_check" / "adapter"
    if adapter.exists():
        shutil.rmtree(adapter)
    model.save_pretrained(adapter)
    peak = int(torch.cuda.max_memory_allocated())
    _free(model)
    tok, model = load_for_eval(str(adapter))
    from .infer import generate

    generate(
        tok, model,
        [{"uid": "smoke", "prompt": logic_prompt(rendered["text"])}],
        BUILD / "_check" / "gen.jsonl",
        max_new_tokens=8,
        batch_size=1,
        shuffle_seed=SEED,
    )
    _free(model)
    shutil.rmtree(BUILD / "_check", ignore_errors=True)
    return {
        "quant": quant,
        "smoke_loss": float(loss.detach()),
        "smoke_sft": float(sft),
        "smoke_peak_vram_bytes": peak,
        "kl": float(kl),
    }


def resolve_quant() -> dict:
    path = BUILD / "quant.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("ok"):
            log(f"reusing quantization {saved['quant']}")
            return saved
    info = {"ok": False, "nf4_error": None}
    try:
        info.update(smoke_quant("nf4"))
        info["ok"] = True
        log(f"NF4 smoke ok loss {info['smoke_loss']:.4f}")
    except Exception as exc:
        info["nf4_error"] = f"{type(exc).__name__}: {exc}"
        log(f"NF4 smoke failed: {info['nf4_error']}")
        log(traceback.format_exc())
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        info.update(smoke_quant("bf16"))
        info["ok"] = True
        info["fallback"] = "bf16 LoRA without quantization"
        log(f"bf16 LoRA smoke ok loss {info['smoke_loss']:.4f}")
    write_json(path, info)
    return info


def write_config(summary: dict, quant_info: dict, gsm_endpoint: str) -> None:
    sizes, remine_after = optimizer_step_plan()
    write_json(PILOT / "config.json", {
        "model": MODEL_NAME,
        "seed": SEED,
        "depth": DEPTH,
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "n_test_per_split": N_TEST,
        "n_gsm8k": N_GSM,
        "k_mine": K_MINE,
        "sequences_per_latent": 2,
        "step_sizes": sizes,
        "n_steps": len(sizes),
        "remine_after_completed_steps": remine_after,
        "effective_batch": EFFECTIVE_BATCH,
        "micro_batch": MICRO_BATCH,
        "lr": LR,
        "warmup_fraction": WARMUP_FRAC,
        "warmup_steps": warmup_steps(len(sizes)),
        "grad_clip": GRAD_CLIP,
        "weight_decay": WEIGHT_DECAY,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": list(TARGET_MODULES),
        "max_length": MAX_LEN,
        "max_new_tokens": MAX_NEW_TOKENS,
        "lambda_consistency": LAMBDA_CONSISTENCY,
        "lambda_definition": "KL(p||q)+KL(q||p) on the two-way softmax of the yes and no answer tokens",
        "val_worse_mean_threshold": VAL_WORSE_MEAN,
        "quant": quant_info.get("quant"),
        "nf4_error": quant_info.get("nf4_error"),
        "gsm8k_endpoint": gsm_endpoint,
        "git_hash_at_start": git_hash(),
        "versions": package_versions(),
        "shifted_presentation": "reversed",
        "training_permutation": summary["training_permutation"],
        "slot_random_unique_max": summary["slot_random_unique_max"],
    })


def update_status(completed: list[str]) -> None:
    write_json(PILOT / "status.json", {"completed": completed, "updated": time.strftime("%Y-%m-%d %H:%M:%S")})


def pilot(conditions: list[str]) -> int:
    _quiet_hf()
    _install_log()
    log("PILOT START " + " ".join(conditions))
    stopped = PILOT / "STOPPED.json"
    if stopped.exists():
        stopped.unlink()
    try:
        pools, summary = build_pools()
        write_json(PILOT / "data_summary.json", summary)
        log(
            "pools ok "
            f"slot-random unique min/mean/max "
            f"{summary['slot_random_unique_min']}/"
            f"{summary['slot_random_unique_mean']:.3f}/"
            f"{summary['slot_random_unique_max']}"
        )
        dataset, gsm_meta = load_gsm8k()
        if dataset is None:
            gsm_saved = None
            endpoint = "not_downloaded"
            log("GSM8K not downloaded; synthetic conditions will still run")
        else:
            endpoint = gsm_meta
            gsm_saved = gsm_index_and_gold(dataset)
            gsm_saved["endpoint"] = endpoint
            write_json(PILOT / "gsm8k_gold.json", gsm_saved)
            log(f"GSM8K ready via {endpoint} n={len(gsm_saved['indices'])}")
        quant_info = resolve_quant()
        write_config(summary, quant_info, endpoint)
        completed = []
        for cond in CONDITIONS:
            metrics_path = PILOT / cond / "metrics.json"
            if metrics_path.exists():
                completed.append(cond)
        for cond in conditions:
            metrics_path = PILOT / cond / "metrics.json"
            if metrics_path.exists():
                saved = json.loads(metrics_path.read_text(encoding="utf-8"))
                if saved.get("status") == "complete":
                    log(f"skip complete {cond}")
                    if cond not in completed:
                        completed.append(cond)
                    continue
            run_condition(cond, pools, dataset, gsm_saved, quant_info["quant"])
            if cond not in completed:
                completed.append(cond)
            update_status(completed)
        log("PILOT COMPLETE " + " ".join(completed))
        return 0
    except PilotStop as exc:
        log(f"PILOT STOP {exc}")
        write_json(PILOT / "STOPPED.json", {"reason": str(exc), "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        return 2
    except Exception as exc:
        log(f"PILOT STOP unexpected {type(exc).__name__}: {exc}")
        log(traceback.format_exc())
        write_json(PILOT / "STOPPED.json", {
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        return 2


def _online_env() -> None:
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[key] = "0"
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare_gsm_training_pairs(tok, train_ds, test_ds, gold: dict) -> tuple[list, dict]:
    counts = gsm_mix_counts()
    blocked = held_out_question_set(test_ds, gold)
    for index, expected in zip(gold["indices"], gold["gold"]):
        actual = gsm_gold(test_ds[int(index)]["answer"])
        if actual != expected:
            raise RuntimeError(
                f"held-out GSM8K item {index} gold {expected!r} does not match the test split {actual!r}"
            )
    path = PILOT / "gsm8k_train.json"
    skipped = 0
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        chosen = [int(i) for i in saved["indices"]]
        if saved.get("split") != "train" or len(chosen) != counts["n_gsm_questions"]:
            raise RuntimeError("saved GSM8K train mix does not match the 15% plan")
        skipped = int(saved.get("n_skipped", 0))
    else:
        questions = [train_ds[i]["question"] for i in range(len(train_ds))]

        def accept(index: int) -> bool:
            nonlocal skipped
            try:
                encode_gsm_example(
                    tok, train_ds[index]["question"], train_ds[index]["answer"], f"gsm8k-train|{index}",
                )
                return True
            except RuntimeError as exc:
                text = str(exc)
                if "exceeds" in text or "rationale is empty" in text or "too short" in text:
                    skipped += 1
                    return False
                raise

        chosen = select_gsm_train_indices(questions, blocked, counts["n_gsm_questions"], SEED, accept)
    selected = {normalize_question(train_ds[i]["question"]) for i in chosen}
    overlap = selected & blocked
    if overlap:
        raise RuntimeError(f"{len(overlap)} selected GSM8K train questions match the held-out test set")
    pairs = []
    rows = []
    for index in chosen:
        question = train_ds[index]["question"]
        answer = train_ds[index]["answer"]
        example = encode_gsm_example(tok, question, answer, f"gsm8k-train|{index}")
        pairs.append(gsm_training_pair(example))
        rows.append({
            "index": index,
            "gold": gsm_gold(answer),
            "sha256": hashlib.sha256(normalize_question(question).encode("utf-8")).hexdigest(),
            "n_tokens": len(example["input_ids"]),
        })
    if len(pairs) != counts["n_gsm_questions"]:
        raise RuntimeError("GSM8K pair count does not match the mix plan")
    sizes, remine_after = optimizer_step_plan(counts["n_total_sequences"])
    record = {
        "split": "train",
        "seed": SEED,
        "fraction_requested": counts["fraction_requested"],
        "gsm_fraction": counts["gsm_fraction"],
        "target_gsm_sequences": counts["target_gsm_sequences"],
        "n_gsm_questions": counts["n_gsm_questions"],
        "n_gsm_sequences": counts["n_gsm_sequences"],
        "n_synthetic_sequences": counts["n_synthetic_sequences"],
        "n_total_sequences": counts["n_total_sequences"],
        "n_steps": len(sizes),
        "step_sizes": sizes,
        "remine_after_completed_steps": remine_after,
        "indices": chosen,
        "n_skipped": skipped,
        "held_out_test_file": "experiments/e1/pilot/gsm8k_gold.json",
        "held_out_test_n": N_GSM,
        "held_out_test_sha256": _sha256(GSM_HELDOUT_PATH),
        "selected_text_overlap_with_held_out": 0,
        "index_space": (
            "Indices are positions in the GSM8K train split. "
            "The 500 test-split indices in gsm8k_gold.json are not used to read training rows."
        ),
        "pairing": "Each selected train question is copied twice so every microbatch stays a same-task pair.",
        "lambda_0_5_ran": False,
        "questions": rows,
    }
    write_json(path, record)
    mix_note = {k: v for k, v in record.items() if k != "questions"}
    mix_note["indices_file"] = "experiments/e1/pilot_mix15/gsm8k_train.json"
    return pairs, mix_note


def mix15(conditions: list[str]) -> int:
    global PILOT, BUILD, LOG_PATH
    _online_env()
    PILOT = PROJECT / "experiments" / "e1" / "pilot_mix15"
    BUILD = PROJECT / "build" / "e1" / "pilot_mix15"
    LOG_PATH = PROJECT / "build" / "logs" / "e1_mix15.log"
    if PILOT == PROJECT / "experiments" / "e1" / "pilot":
        raise RuntimeError("mix15 output path collided with the finished pilot")
    _quiet_hf()
    _install_log()
    log("MIX15 START " + " ".join(conditions))
    log(
        "offline "
        f"hub={os.environ.get('HF_HUB_OFFLINE')} "
        f"datasets={os.environ.get('HF_DATASETS_OFFLINE')} "
        f"tf={os.environ.get('TRANSFORMERS_OFFLINE')} "
        f"endpoint={os.environ.get('HF_ENDPOINT')}"
    )
    stopped = PILOT / "STOPPED.json"
    if stopped.exists():
        stopped.unlink()
    try:
        if not GSM_HELDOUT_PATH.exists():
            raise PilotStop(f"missing held-out GSM8K index {GSM_HELDOUT_PATH}")
        heldout_hash = _sha256(GSM_HELDOUT_PATH)
        pools, summary = build_pools()
        write_json(PILOT / "data_summary.json", summary)
        train_ds, train_label, train_errors = _load_hf_split("train")
        test_ds, test_label, test_errors = _load_hf_split("test")
        if train_ds is None or test_ds is None:
            note = {
                "status": "not_downloaded",
                "reason": "GSM8K train or test split could not be downloaded. No mix15 score is reported.",
                "train_attempts": train_errors,
                "test_attempts": test_errors,
            }
            write_json(PILOT / "gsm8k_unavailable.json", note)
            raise PilotStop(note["reason"])
        log(f"GSM8K train via {train_label} n={len(train_ds)}; test via {test_label} n={len(test_ds)}")
        gold = json.loads(GSM_HELDOUT_PATH.read_text(encoding="utf-8"))
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        extra_pairs, mix_note = prepare_gsm_training_pairs(tok, train_ds, test_ds, gold)
        del tok
        if _sha256(GSM_HELDOUT_PATH) != heldout_hash:
            raise PilotStop("held-out GSM8K index changed while the mix was being prepared")
        log(
            f"GSM8K mix questions {mix_note['n_gsm_questions']} sequences {mix_note['n_gsm_sequences']} "
            f"fraction {mix_note['gsm_fraction']:.6f} steps {mix_note['n_steps']} skipped {mix_note['n_skipped']}"
        )
        prior_quant = PROJECT / "build" / "e1" / "pilot" / "quant.json"
        quant_path = BUILD / "quant.json"
        if prior_quant.exists() and not quant_path.exists():
            quant_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(prior_quant, quant_path)
        quant_info = resolve_quant()
        sizes, remine_after = optimizer_step_plan(mix_note["n_total_sequences"])
        write_json(PILOT / "config.json", {
            "model": MODEL_NAME,
            "seed": SEED,
            "depth": DEPTH,
            "n_train": N_TRAIN,
            "n_val": N_VAL,
            "n_test_per_split": N_TEST,
            "n_gsm8k_eval": N_GSM,
            "gsm_mix": mix_note,
            "k_mine": K_MINE,
            "sequences_per_latent": 2,
            "step_sizes": sizes,
            "n_steps": len(sizes),
            "remine_after_completed_steps": remine_after,
            "effective_batch": EFFECTIVE_BATCH,
            "micro_batch": MICRO_BATCH,
            "lr": LR,
            "warmup_fraction": WARMUP_FRAC,
            "warmup_steps": warmup_steps(len(sizes)),
            "grad_clip": GRAD_CLIP,
            "weight_decay": WEIGHT_DECAY,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": list(TARGET_MODULES),
            "max_length": MAX_LEN,
            "max_new_tokens": MAX_NEW_TOKENS,
            "lambda_consistency": LAMBDA_CONSISTENCY,
            "lambda_0_5_ran": False,
            "quant": quant_info.get("quant"),
            "nf4_error": quant_info.get("nf4_error"),
            "gsm8k_train_endpoint": train_label,
            "gsm8k_test_endpoint": test_label,
            "b0_metrics": "experiments/e1/pilot/b0/metrics.json",
            "held_out_test_file": "experiments/e1/pilot/gsm8k_gold.json",
            "held_out_test_sha256": heldout_hash,
            "git_hash_at_start": git_hash(),
            "versions": package_versions(),
            "shifted_presentation": "reversed",
            "training_permutation": summary["training_permutation"],
            "conditions": list(conditions),
        })
        completed = []
        for cond in conditions:
            metrics_path = PILOT / cond / "metrics.json"
            if metrics_path.exists() and json.loads(metrics_path.read_text(encoding="utf-8")).get("status") == "complete":
                log(f"skip complete {cond}")
                completed.append(cond)
                continue
            run_condition(cond, pools, test_ds, gold, quant_info["quant"], extra_pairs, mix_note)
            completed.append(cond)
            update_status(completed)
        if _sha256(GSM_HELDOUT_PATH) != heldout_hash:
            raise PilotStop("held-out GSM8K index changed during mix15")
        b0_path = PROJECT / "experiments" / "e1" / "pilot" / "b0" / "metrics.json"
        b0_acc = json.loads(b0_path.read_text(encoding="utf-8"))["gsm8k"]["acc"]
        for cond in completed:
            acc = json.loads((PILOT / cond / "metrics.json").read_text(encoding="utf-8"))["gsm8k"]["acc"]
            log(f"GSM8K_COMPARE {cond} {acc:.4f} b0 {b0_acc:.4f} drop {b0_acc - acc:.4f}")
        log("MIX15 COMPLETE " + " ".join(completed) + "; mix fraction was not increased")
        return 0
    except PilotStop as exc:
        log(f"MIX15 STOP {exc}")
        write_json(PILOT / "STOPPED.json", {"reason": str(exc), "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        return 2
    except Exception as exc:
        log(f"MIX15 STOP unexpected {type(exc).__name__}: {exc}")
        log(traceback.format_exc())
        write_json(PILOT / "STOPPED.json", {
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        return 2


def check() -> int:
    _quiet_hf()
    _install_log()
    log("CHECK START")
    try:
        pools, summary = build_pools()
        if summary["slot_random_unique_max"] >= K_MINE:
            raise PilotStop("slot-constrained random already supplies K distinct texts; revisit the permutation choice")
        info = resolve_quant()
        log(f"CHECK_OK quant {info['quant']} loss {info['smoke_loss']:.4f}")
        # Touch one encode path on every training family.
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        yes_id, no_id = yes_no_token_ids(tok)
        lengths = []
        for lat in pools["train"][:8]:
            renders = [canonical_render(lat)] + sample_renders(lat, 2, "b2-random")
            for ex in examples_from_renders(tok, lat, renders, yes_id, no_id):
                lengths.append(len(ex["input_ids"]))
        log(f"CHECK lengths min/max {min(lengths)}/{max(lengths)} n_train {len(pools['train'])}")
        return 0
    except Exception as exc:
        log(f"CHECK FAIL {type(exc).__name__}: {exc}")
        log(traceback.format_exc())
        return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check", "pilot", "mix15"))
    parser.add_argument("--conditions", nargs="*", default=None)
    args = parser.parse_args()
    if args.command == "check":
        raise SystemExit(check())
    if args.command == "mix15":
        conditions = list(MIX_CONDITIONS if args.conditions is None else args.conditions)
        unknown = [name for name in conditions if name not in MIX_CONDITIONS]
        if unknown:
            raise SystemExit(f"mix15 conditions must be among {list(MIX_CONDITIONS)}, got {unknown}")
        raise SystemExit(mix15(conditions))
    conditions = list(CONDITIONS if args.conditions is None else args.conditions)
    unknown = [name for name in conditions if name not in CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown conditions {unknown}")
    raise SystemExit(pilot(conditions))


if __name__ == "__main__":
    main()
