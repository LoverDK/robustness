"""Synonym control for the saved E1 pilot B1 adapter.

B1 training_renders returns the canonical aligned presentation twice.
Aligned and reversed answers stay in build/e1/pilot/b1/logic.jsonl.
Sentence construction, graph-hash pairing, SF, bootstrap, and load_for_eval
come from e1_synonym. This module does not run B0 or B2.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .e1 import PROJECT, build_pools, read_jsonl
from .e1_synonym import (
    BATCH_SIZES,
    BOOT_SEED,
    MAX_NEW_TOKENS,
    N_BOOT,
    N_TEST,
    POOL_NAME,
    SPLITS,
    _close,
    _flip_counts,
    _free,
    acc_interval,
    build_synonym_items,
    check_pools,
    correctness_vectors,
    interval_relation,
    line_stats,
    load_rescore,
    run_generate,
    sf_interval,
)

FAMILIES = ("chain", "conj_fact", "tree", "diamond")
EXPECTED_LINES = 800
EXPECTED_UIDS = 800
LOGIC_PATH = PROJECT / "build" / "e1" / "pilot" / "b1" / "logic.jsonl"
ADAPTER_PATH = PROJECT / "build" / "e1" / "pilot" / "b1" / "adapter"
GEN_PATH = PROJECT / "build" / "e1" / "synonym" / "b1" / "generations.jsonl"
METRICS_PATH = PROJECT / "experiments" / "e1" / "synonym" / "b1_metrics.json"
LOG_PATH = PROJECT / "build" / "logs" / "e1_synonym_b1.log"
RESCORE_KEY = ("rounds", "pilot", "conditions", "b1")
B1_SANITY = {
    "seen": {
        "aligned_acc": 0.97,
        "reversed_SF": (
            0.26804123711340205,
            0.20833333333333334,
            0.33160687850269643,
        ),
        "n_aligned_correct": 194,
        "n_flip": 52,
    },
    "unseen": {
        "aligned_acc": 0.88,
        "reversed_SF": (
            0.36363636363636365,
            0.2921116804543771,
            0.4375,
        ),
        "n_aligned_correct": 176,
        "n_flip": 64,
    },
}
LOAD_NOTE = (
    "Pilot metrics record B1 quant as nf4 because training used nf4. "
    "Aligned generations were produced by e1.run_condition, which calls "
    "e1.load_for_eval. That loads the base model in bfloat16 and, when an "
    "adapter is set, merges it with PeftModel.merge_and_unload. Synonym "
    "generation uses the same function on build/e1/pilot/b1/adapter. "
    "Aligned and reversed answers are the saved generations in "
    "build/e1/pilot/b1/logic.jsonl; this run does not regenerate them."
)


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S") + " " + msg, flush=True)


def guard_b1_output(path: Path) -> Path:
    """Allow only the B1 synonym generations, metrics, and log."""
    resolved = path if path.is_absolute() else PROJECT / path
    resolved = resolved.resolve()
    allowed = {
        GEN_PATH.resolve(),
        METRICS_PATH.resolve(),
        METRICS_PATH.with_suffix(METRICS_PATH.suffix + ".tmp").resolve(),
        LOG_PATH.resolve(),
    }
    if resolved not in allowed:
        raise SystemExit(f"refusing to write {resolved}")
    return resolved


def _triple_close(got, expected) -> bool:
    if len(got) != 3 or len(expected) != 3:
        return False
    return all(_close(left, right) for left, right in zip(got, expected))


def reversed_control_record(aligned, reversed_) -> dict:
    """Aligned accuracy and reversed SF for one split, in the given row order."""
    aligned_v = np.asarray(aligned, dtype=float)
    reversed_v = np.asarray(reversed_, dtype=float)
    counts = _flip_counts(aligned_v, reversed_v)
    return {
        "aligned_acc": acc_interval(aligned_v),
        "reversed_acc": acc_interval(reversed_v),
        "reversed_SF": sf_interval(aligned_v, reversed_v),
        "n_aligned_correct": counts["n_aligned_correct"],
        "n_flip": counts["n_flip"],
        "n_stay": counts["n_stay"],
        "n": counts["n"],
    }


def assert_b1_control_numbers(recomputed: dict, saved_block: dict) -> None:
    """Stop when rebuilt B1 numbers disagree with the stated control or rescore."""
    if saved_block.get("quant") != "nf4":
        raise RuntimeError(f"rescore B1 quant is {saved_block.get('quant')}")
    problems = []
    for split, sanity in B1_SANITY.items():
        got = recomputed[split]
        saved_sf = saved_block["SF"][split]
        saved_aligned = saved_block["acc"][split]["aligned"]
        saved_reversed = saved_block["acc"][split]["reversed"]
        saved_counts = saved_block["counts"][split]
        if not _close(got["aligned_acc"][0], sanity["aligned_acc"]):
            problems.append(f"{split} rebuilt aligned acc {got['aligned_acc'][0]}")
        if not _triple_close(got["reversed_SF"], sanity["reversed_SF"]):
            problems.append(f"{split} rebuilt reversed SF {got['reversed_SF']}")
        if (
            got["n_aligned_correct"] != sanity["n_aligned_correct"]
            or got["n_flip"] != sanity["n_flip"]
        ):
            problems.append(
                f"{split} rebuilt counts "
                f"{got['n_aligned_correct']} {got['n_flip']}"
            )
        if not _close(saved_aligned[0], sanity["aligned_acc"]):
            problems.append(f"{split} rescore aligned acc {saved_aligned[0]}")
        if not _triple_close(saved_sf, sanity["reversed_SF"]):
            problems.append(f"{split} rescore reversed SF {saved_sf}")
        if (
            saved_counts["n_aligned_correct"] != sanity["n_aligned_correct"]
            or saved_counts["n_flip"] != sanity["n_flip"]
        ):
            problems.append(f"{split} rescore counts {saved_counts}")
        if not _triple_close(got["reversed_SF"], saved_sf):
            problems.append(f"{split} reversed SF != rescore")
        if not _triple_close(got["aligned_acc"], saved_aligned):
            problems.append(f"{split} aligned acc interval != rescore")
        if not _triple_close(got["reversed_acc"], saved_reversed):
            problems.append(f"{split} reversed acc interval != rescore")
        if (
            got["n_aligned_correct"] != saved_counts["n_aligned_correct"]
            or got["n_flip"] != saved_counts["n_flip"]
            or got["n_stay"] != saved_counts["n_stay"]
        ):
            problems.append(f"{split} flip counts != rescore")
    if problems:
        raise RuntimeError("; ".join(problems))


def family_conditional_sf(rows: list[dict]) -> dict:
    """Per-family synonym and reversed SF.

    Bootstrap resamples latent tasks inside the family. Row order inside a
    family is sorted graph_hash, with the same n_boot and seed as sf_interval.
    """
    grouped = {family: [] for family in FAMILIES}
    for row in rows:
        family = row["family"]
        if family not in grouped:
            raise ValueError(family)
        grouped[family].append(row)
    out = {}
    for family in FAMILIES:
        group = grouped[family]
        if not group:
            raise RuntimeError(f"empty family {family}")
        group = sorted(group, key=lambda row: row["graph_hash"])
        splits = {row["split"] for row in group}
        if len(splits) != 1:
            raise RuntimeError(f"{family} spans {sorted(splits)}")
        aligned = np.asarray([row["aligned_correct"] for row in group], dtype=float)
        reversed_ = np.asarray([row["reversed_correct"] for row in group], dtype=float)
        synonym = np.asarray([row["synonym_correct"] for row in group], dtype=float)
        synonym_sf = sf_interval(aligned, synonym)
        reversed_sf = sf_interval(aligned, reversed_)
        relation = interval_relation(synonym_sf, reversed_sf)
        synonym_counts = _flip_counts(aligned, synonym)
        reversed_counts = _flip_counts(aligned, reversed_)
        extract_fail = int(sum(row["synonym_extract_fail"] for row in group))
        out[family] = {
            "pool": next(iter(splits)),
            "n": int(aligned.size),
            "synonym_SF": synonym_sf,
            "reversed_SF": reversed_sf,
            "interval_relation": relation,
            "overlaps_reversed_SF": relation == "重叠",
            "n_aligned_correct": synonym_counts["n_aligned_correct"],
            "synonym_n_flip": synonym_counts["n_flip"],
            "reversed_n_flip": reversed_counts["n_flip"],
            "synonym_acc": acc_interval(synonym),
            "synonym_extract_fail": extract_fail,
        }
    return out


def _saved_block(rescore: dict) -> dict:
    block = rescore
    for key in RESCORE_KEY:
        block = block[key]
    return block


def _install_log() -> None:
    path = guard_b1_output(LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    class _Tee:
        def __init__(self, stream, file_path):
            self.stream = stream
            self.file = open(file_path, "a", encoding="utf-8", buffering=1)

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

    sys.stdout = _Tee(sys.stdout, path)
    sys.stderr = _Tee(sys.stderr, path)


def _write_json(path: Path, obj) -> None:
    path = guard_b1_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = guard_b1_output(path.with_suffix(path.suffix + ".tmp"))
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _assert_merged_bf16(model) -> None:
    import torch
    from peft import PeftModel

    if isinstance(model, PeftModel) or hasattr(model, "peft_config"):
        raise RuntimeError("load_for_eval left the B1 adapter unmerged")
    dtype = next(model.parameters()).dtype
    if dtype != torch.bfloat16:
        raise RuntimeError(f"eval dtype is {dtype}")


def _logic_gate(rescore: dict) -> dict:
    stats = line_stats(LOGIC_PATH)
    if stats["n_lines"] != EXPECTED_LINES or stats["n_uid"] != EXPECTED_UIDS:
        raise RuntimeError(f"b1 logic.jsonl counts {stats} != {(EXPECTED_LINES, EXPECTED_UIDS)}")
    block = _saved_block(rescore)
    if block.get("n_logic_lines") not in (None, EXPECTED_LINES):
        raise RuntimeError(f"rescore n_logic_lines is {block.get('n_logic_lines')}")
    if block.get("n_logic_uid") not in (None, EXPECTED_UIDS):
        raise RuntimeError(f"rescore n_logic_uid is {block.get('n_logic_uid')}")
    gens = read_jsonl(LOGIC_PATH)
    if len(gens) != EXPECTED_UIDS:
        raise RuntimeError(f"read_jsonl returned {len(gens)} b1 uids")
    return gens


def _reversed_by_split(pools: dict, pilot_gens: dict) -> tuple[dict, list[dict]]:
    recomputed = {}
    rows = []
    for split in SPLITS:
        vectors = correctness_vectors(pools[POOL_NAME[split]], split, pilot_gens, {})
        if len(vectors["hashes"]) != N_TEST:
            raise RuntimeError(f"{split} has {len(vectors['hashes'])} paired tasks")
        recomputed[split] = reversed_control_record(vectors["aligned"], vectors["reversed"])
        rows.extend(vectors["rows"])
    family_n = {family: 0 for family in FAMILIES}
    for row in rows:
        if row["family"] not in family_n:
            raise RuntimeError(f"unexpected family {row['family']}")
        family_n[row["family"]] += 1
    expected_n = N_TEST // 2
    for family, count in family_n.items():
        if count != expected_n:
            raise RuntimeError(f"{family} has {count} tasks")
    return recomputed, rows


def _pending_count(items: list[dict], path: Path) -> int:
    done = read_jsonl(path) if path.exists() else {}
    return sum(1 for item in items if item["uid"] not in done)


def generate_b1(items: list[dict]) -> dict:
    from .e1 import load_for_eval

    gen_path = guard_b1_output(GEN_PATH)
    pending = _pending_count(items, gen_path)
    log(f"b1 pending {pending} / {len(items)}")
    if pending == 0:
        info = {
            "condition": "b1",
            "skipped": True,
            "batch_size": None,
            "reason": "generations already complete",
            "max_new_tokens": MAX_NEW_TOKENS,
            "decoding": "greedy",
            "cot": True,
            "adapter": "build/e1/pilot/b1/adapter",
            "load": "e1.load_for_eval",
            "base_dtype": "bfloat16",
            "adapter_application": "merge_and_unload",
        }
        log("b1 generation skipped because generations already complete")
        return info
    if not ADAPTER_PATH.exists():
        raise RuntimeError(f"missing adapter {ADAPTER_PATH}")
    log("b1 load_for_eval adapter=build/e1/pilot/b1/adapter")
    _tok, model = load_for_eval(str(ADAPTER_PATH))
    try:
        _assert_merged_bf16(model)
        batch_size = run_generate(_tok, model, items, gen_path)
    finally:
        _free(model)
    info = {
        "condition": "b1",
        "skipped": False,
        "batch_size": batch_size,
        "reason": None,
        "max_new_tokens": MAX_NEW_TOKENS,
        "decoding": "greedy",
        "cot": True,
        "adapter": "build/e1/pilot/b1/adapter",
        "load": "e1.load_for_eval",
        "base_dtype": "bfloat16",
        "adapter_application": "merge_and_unload",
        "batch_schedule": list(BATCH_SIZES),
    }
    log(f"b1 generation finished batch_size={batch_size}")
    return info


def _split_summary(vectors: dict, reversed_sf: list[float]) -> dict:
    synonym_sf = sf_interval(vectors["aligned"], vectors["synonym"])
    relation = interval_relation(synonym_sf, reversed_sf)
    counts = _flip_counts(vectors["aligned"], vectors["synonym"])
    extract_fail = int(sum(row["synonym_extract_fail"] for row in vectors["rows"]))
    return {
        "synonym_SF": synonym_sf,
        "reversed_SF": [float(value) for value in reversed_sf],
        "interval_relation": relation,
        "overlaps_reversed_SF": relation == "重叠",
        "counts": counts,
        "synonym_acc": acc_interval(vectors["synonym"]),
        "aligned_acc": acc_interval(vectors["aligned"]),
        "synonym_extract_fail": extract_fail,
        "synonym_extract_fail_rate": extract_fail / N_TEST,
    }


def summarize(pools: dict, pilot_gens: dict, run_info: dict, recomputed: dict) -> dict:
    synonym_gens = read_jsonl(GEN_PATH)
    per_split = {}
    rows = []
    for split in SPLITS:
        vectors = correctness_vectors(
            pools[POOL_NAME[split]], split, pilot_gens, synonym_gens,
        )
        if vectors["synonym"] is None or vectors["synonym"].size != N_TEST:
            raise RuntimeError(f"b1 {split} synonym generations are incomplete")
        again = sf_interval(vectors["aligned"], vectors["reversed"])
        if not _triple_close(again, recomputed[split]["reversed_SF"]):
            raise RuntimeError(f"b1 {split} reversed SF changed while scoring synonyms")
        per_split[split] = _split_summary(vectors, recomputed[split]["reversed_SF"])
        rows.extend(vectors["rows"])
    families = family_conditional_sf(rows)
    expected_n = N_TEST // 2
    for family, block in families.items():
        if block["n"] != expected_n:
            raise RuntimeError(f"{family} n is {block['n']}")
    payload = {
        "training": False,
        "round": "pilot",
        "condition": "b1",
        "training_renders": "canonical aligned presentation twice",
        "conditions_run": ["b1"],
        "SF_definition": "P(synonym wrong | aligned right)",
        "interval": "[point, low, high]",
        "interval_relation_rule": "overlap if the intervals touch, otherwise not",
        "n_boot": N_BOOT,
        "bootstrap_seed": BOOT_SEED,
        "paired_unit": "latent_task",
        "hash_order": "sorted graph_hash",
        "family_bootstrap": "latent tasks inside one family, sorted by graph_hash",
        "pool_seed": 0,
        "decoding": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS, "cot": True},
        "prompt": "logic_prompt",
        "presentation": "synonym",
        "synonym_scope": "proof step 0 only; premises and conclusion unchanged",
        "ambiguous_included": False,
        "eval_load": {
            "function": "e1.load_for_eval",
            "adapter": "build/e1/pilot/b1/adapter",
            "base_dtype": "bfloat16",
            "adapter_application": "merge_and_unload",
            "source_metrics_quant": "nf4",
            "note": LOAD_NOTE,
        },
        "reversed_SF_source": "experiments/e1/rescore/metrics.json rounds.pilot.conditions.b1",
        "reversed_SF_check": "recomputed from build/e1/pilot/b1/logic.jsonl before generation",
        "aligned_source": "build/e1/pilot/b1/logic.jsonl",
        "generations": "build/e1/synonym/b1/generations.jsonl",
        "run": run_info,
        "splits": per_split,
        "families": families,
    }
    _write_json(METRICS_PATH, payload)
    log("wrote experiments/e1/synonym/b1_metrics.json")
    return payload


def main() -> None:
    import os

    os.environ["HF_HOME"] = str(PROJECT / ".cache" / "huggingface")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _install_log()
    try:
        log("b1 synonym control start; training false")
        rescore = load_rescore()
        pilot_gens = _logic_gate(rescore)
        pools, summary = build_pools(seed=0)
        if not summary.get("ok"):
            raise RuntimeError("pool isolation failed")
        log("checking synonym sentences against the solver")
        check_pools(pools)
        log("solver check passed")
        recomputed, _rows = _reversed_by_split(pools, pilot_gens)
        assert_b1_control_numbers(recomputed, _saved_block(rescore))
        log("b1 reversed SF matches experiments/e1/rescore/metrics.json")
        for split in SPLITS:
            record = recomputed[split]
            log(
                f"b1 {split} aligned_acc={record['aligned_acc'][0]} "
                f"reversed_SF={record['reversed_SF']} "
                f"n_aligned_correct={record['n_aligned_correct']} n_flip={record['n_flip']}"
            )
        items, _meta = build_synonym_items(pools)
        if len(items) != N_TEST * len(SPLITS):
            raise RuntimeError(f"expected {N_TEST * len(SPLITS)} synonym items, got {len(items)}")
        run_info = generate_b1(items)
        payload = summarize(pools, pilot_gens, run_info, recomputed)
        brief = {
            split: {
                "synonym_SF": payload["splits"][split]["synonym_SF"],
                "reversed_SF": payload["splits"][split]["reversed_SF"],
                "overlaps_reversed_SF": payload["splits"][split]["overlaps_reversed_SF"],
                "aligned_acc": payload["splits"][split]["aligned_acc"],
                "synonym_extract_fail": payload["splits"][split]["synonym_extract_fail"],
            }
            for split in SPLITS
        }
        brief["batch_size"] = run_info["batch_size"]
        brief["generation_skipped"] = run_info["skipped"]
        log(json.dumps(brief, ensure_ascii=False))
        log("B1 SYNONYM COMPLETE")
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
