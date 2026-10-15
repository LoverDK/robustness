"""Synonym re-score for the saved E1 pilot B0 and B2 models.

No training. The test pools are e1.build_pools(seed=0) test_seen and test_unseen.
Only proof step 0 is replaced by wording.render_condition(..., "synonym").
Aligned right/wrong is read from the existing pilot logic.jsonl files, paired by
graph hash. Duplicate uids keep the last line, via e1.read_jsonl.

B0 and B2 are loaded with e1.load_for_eval, the same function that produced the
aligned generations. Pilot metrics record B2 quant as nf4 because training used
nf4; load_for_eval loads the base model in bfloat16 and merges the adapter.
"""

from __future__ import annotations

import json
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import logic
from . import wording
from .e1 import (
    PROJECT,
    build_pools,
    graph_hash,
    read_jsonl,
)
from .prompts import extract_yes_no, logic_prompt
from .stats import conditional_flip_ci, mean_ci

N_BOOT = 2000
BOOT_SEED = 0
N_TEST = 200
MAX_NEW_TOKENS = 512
BATCH_SIZES = (32, 16, 8, 4)
CONDITIONS = ("b0", "b2")
SPLITS = ("seen", "unseen")
POOL_NAME = {"seen": "test_seen", "unseen": "test_unseen"}
RESCORE_PATH = PROJECT / "experiments" / "e1" / "rescore" / "metrics.json"
PILOT_LOGIC = {
    "b0": PROJECT / "build" / "e1" / "pilot" / "b0" / "logic.jsonl",
    "b2": PROJECT / "build" / "e1" / "pilot" / "b2" / "logic.jsonl",
}
ADAPTER = {
    "b0": None,
    "b2": PROJECT / "build" / "e1" / "pilot" / "b2" / "adapter",
}
OUT_ROOT = PROJECT / "build" / "e1" / "synonym"
METRICS_PATH = PROJECT / "experiments" / "e1" / "synonym" / "metrics.json"
LOG_PATH = PROJECT / "build" / "logs" / "e1_synonym.log"
EXPECTED_LINES = {"b0": (896, 800), "b2": (800, 800)}
LOAD_NOTE = (
    "Pilot metrics record B2 quant as nf4 because training used nf4. "
    "Aligned generations were produced by e1.run_condition, which calls "
    "e1.load_for_eval. That loads the base model in bfloat16 and, when an "
    "adapter is set, merges it with PeftModel.merge_and_unload. Synonym "
    "generation uses the same function."
)

_SYN_BINARY = re.compile(
    r"^(.+) is always (\S+) whenever \1 is (\S+) and \1 is (\S+)\.$"
)
_SYN_UNARY = re.compile(r"^(.+) is always (\S+) whenever \1 is (\S+)\.$")


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S") + " " + msg, flush=True)


def guard_output(path: Path) -> Path:
    """Allow writes only under the synonym outputs and the synonym log."""
    resolved = path if path.is_absolute() else PROJECT / path
    resolved = resolved.resolve()
    allowed_roots = (
        (PROJECT / "build" / "e1" / "synonym").resolve(),
        (PROJECT / "experiments" / "e1" / "synonym").resolve(),
    )
    log_path = (PROJECT / "build" / "logs" / "e1_synonym.log").resolve()
    if resolved == log_path:
        return resolved
    if any(resolved == root or root in resolved.parents for root in allowed_roots):
        return resolved
    raise SystemExit(f"refusing to write {resolved}")


def parse_synonym_sentence(sentence: str) -> logic.Rule:
    binary = _SYN_BINARY.match(sentence)
    if binary:
        _entity, conclusion, left, right = binary.groups()
        return ((left, right), conclusion)
    unary = _SYN_UNARY.match(sentence)
    if unary:
        _entity, conclusion, premise = unary.groups()
        return ((premise,), conclusion)
    raise ValueError(sentence)


def _rule_lines(text: str) -> list[str]:
    rules = text.split("\n\nRules:\n", 1)[1].split("\n\nQuestion:", 1)[0]
    return rules.splitlines()


def assert_synonym_preserves_solver(lat: logic.Latent) -> None:
    """The synonym sentence denotes the same rule, so the solver answer is unchanged."""
    snapshot = (lat.core_rules, lat.other_rules, lat.facts, lat.query, lat.label, lat.entity)
    before = logic.solve(set(lat.facts), lat.all_rules(False), lat.query)
    aligned = wording.render_condition(lat, "aligned")
    synonym = wording.render_condition(lat, "synonym")
    after_fields = (lat.core_rules, lat.other_rules, lat.facts, lat.query, lat.label, lat.entity)
    if after_fields != snapshot or lat.core_rules is not snapshot[0]:
        raise AssertionError(f"{lat.task_id} render mutated the latent task")
    after = logic.solve(set(lat.facts), lat.all_rules(False), lat.query)
    if before is not after or before is not (lat.label == "yes"):
        raise AssertionError(f"{lat.task_id} solver answer changed")
    if synonym["presentation"] != "synonym":
        raise AssertionError(synonym["presentation"])
    aligned_lines = _rule_lines(aligned["text"])
    synonym_lines = _rule_lines(synonym["text"])
    if aligned_lines != [
        logic.rule_text(lat.entity, {"core": lat.core_rules, "other": lat.other_rules}[kind][index])
        for kind, index in aligned["rule_seq"]
    ]:
        raise AssertionError(f"{lat.task_id} aligned lines are not the generator sentences")
    if len(synonym_lines) != len(aligned_lines):
        raise AssertionError(f"{lat.task_id} rule count changed")
    step0 = None
    for pos, (kind, index) in enumerate(aligned["rule_seq"]):
        if kind == "core" and index == 0:
            step0 = pos
            break
    if step0 is None:
        raise AssertionError(f"{lat.task_id} has no proof step 0")
    lookup = {"core": lat.core_rules, "other": lat.other_rules}
    surface_rules = []
    for pos, (kind, index) in enumerate(aligned["rule_seq"]):
        rule = lookup[kind][index]
        if pos == step0:
            expected = wording.synonym_sentence(lat.entity, rule)
            if synonym_lines[pos] != expected:
                raise AssertionError(f"{lat.task_id} step 0 is not the synonym sentence")
            parsed = parse_synonym_sentence(synonym_lines[pos])
            if parsed != rule:
                raise AssertionError(f"{lat.task_id} synonym changed premises or conclusion: {parsed} != {rule}")
            surface_rules.append(parsed)
        else:
            if synonym_lines[pos] != aligned_lines[pos]:
                raise AssertionError(f"{lat.task_id} changed a sentence other than proof step 0")
            surface_rules.append(rule)
    if logic.solve(set(lat.facts), surface_rules, lat.query) is not before:
        raise AssertionError(f"{lat.task_id} surface rules change the solver answer")
    head, _rest = aligned["text"].split("\n\nRules:\n", 1)
    syn_head, _syn_rest = synonym["text"].split("\n\nRules:\n", 1)
    if head != syn_head:
        raise AssertionError(f"{lat.task_id} facts changed")
    if aligned["text"].split("\n\nQuestion:", 1)[1] != synonym["text"].split("\n\nQuestion:", 1)[1]:
        raise AssertionError(f"{lat.task_id} question changed")


def check_pools(pools: dict) -> None:
    for split in SPLITS:
        pool = pools[POOL_NAME[split]]
        if len(pool) != N_TEST:
            raise AssertionError(f"{split} has {len(pool)} tasks")
        for lat in pool:
            assert_synonym_preserves_solver(lat)


def build_synonym_items(pools: dict) -> tuple[list[dict], dict]:
    items, meta = [], {}
    for split in SPLITS:
        for lat in pools[POOL_NAME[split]]:
            digest = graph_hash(lat)
            rendered = wording.render_condition(lat, "synonym")
            uid = f"{split}_synonym|{digest}"
            items.append({"uid": uid, "prompt": logic_prompt(rendered["text"])})
            meta[uid] = {
                "uid": uid,
                "split": split,
                "graph_hash": digest,
                "label": lat.label,
                "family": lat.family,
                "task_id": lat.task_id,
                "presentation": "synonym",
            }
    return items, meta


def _by_hash(pool) -> dict:
    found = {}
    for lat in pool:
        digest = graph_hash(lat)
        if digest in found:
            raise RuntimeError(f"duplicate graph hash {digest}")
        found[digest] = lat
    return found


def correctness_vectors(pool, split: str, pilot_gens: dict, synonym_gens: dict):
    """Sorted by graph hash, matching rescore._vectors."""
    by_hash = _by_hash(pool)
    hashes = sorted(by_hash)
    aligned, reversed_, synonym = [], [], []
    rows = []
    missing = []
    for digest in hashes:
        lat = by_hash[digest]
        a_uid = f"{split}_aligned|{digest}"
        r_uid = f"{split}_shifted|{digest}"
        s_uid = f"{split}_synonym|{digest}"
        if a_uid not in pilot_gens or r_uid not in pilot_gens:
            missing.append(a_uid if a_uid not in pilot_gens else r_uid)
            continue
        a_pred = extract_yes_no(pilot_gens[a_uid]["generation"])
        r_pred = extract_yes_no(pilot_gens[r_uid]["generation"])
        a_ok = int(a_pred == lat.label)
        r_ok = int(r_pred == lat.label)
        row = {
            "split": split,
            "graph_hash": digest,
            "family": lat.family,
            "label": lat.label,
            "aligned_pred": a_pred,
            "aligned_correct": a_ok,
            "reversed_pred": r_pred,
            "reversed_correct": r_ok,
        }
        if s_uid in synonym_gens:
            s_pred = extract_yes_no(synonym_gens[s_uid]["generation"])
            s_ok = int(s_pred == lat.label)
            synonym.append(s_ok)
            row.update({
                "synonym_pred": s_pred,
                "synonym_correct": s_ok,
                "synonym_extract_fail": int(s_pred == ""),
            })
        aligned.append(a_ok)
        reversed_.append(r_ok)
        rows.append(row)
    if missing:
        raise RuntimeError(f"{split} missing {len(missing)} pilot rows, e.g. {missing[:3]}")
    return {
        "hashes": hashes,
        "aligned": np.asarray(aligned, dtype=float),
        "reversed": np.asarray(reversed_, dtype=float),
        "synonym": np.asarray(synonym, dtype=float) if len(synonym) == len(hashes) else None,
        "rows": rows,
    }


def sf_interval(canon, other) -> list[float]:
    point, low, high = conditional_flip_ci(canon, other, n_boot=N_BOOT, seed=BOOT_SEED)
    if point is None:
        raise RuntimeError("SF denominator is empty")
    return [float(point), float(low), float(high)]


def acc_interval(values) -> list[float]:
    point, low, high = mean_ci(values, n_boot=N_BOOT, seed=BOOT_SEED)
    return [float(point), float(low), float(high)]


def interval_relation(left: list[float], right: list[float]) -> str:
    """Same endpoint rule as rescore._relation. Touching endpoints count as overlap."""
    if left[2] < right[1] or right[2] < left[1]:
        return "不重叠"
    return "重叠"


def _close(got, expected, tol: float = 1e-9) -> bool:
    return abs(float(got) - float(expected)) <= tol


def line_stats(path: Path) -> dict:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return {"n_lines": len(rows), "n_uid": len({row["uid"] for row in rows})}


def load_rescore() -> dict:
    return json.loads(RESCORE_PATH.read_text(encoding="utf-8"))


def require_reversed_matches_rescore(pools: dict, rescore: dict | None = None) -> dict:
    """Stop if rebuilt aligned/reversed SF does not match the saved rescore file."""
    if rescore is None:
        rescore = load_rescore()
    pilot = rescore["rounds"]["pilot"]["conditions"]
    checked = {}
    for cond in CONDITIONS:
        path = PILOT_LOGIC[cond]
        stats = line_stats(path)
        expect_lines, expect_uid = EXPECTED_LINES[cond]
        if stats["n_lines"] != expect_lines or stats["n_uid"] != expect_uid:
            raise RuntimeError(f"{cond} logic.jsonl counts {stats} != {(expect_lines, expect_uid)}")
        gens = read_jsonl(path)
        if len(gens) != expect_uid:
            raise RuntimeError(f"{cond} read_jsonl returned {len(gens)} uids")
        block = pilot[cond]
        checked[cond] = {"logic_lines": stats["n_lines"], "logic_uids": stats["n_uid"], "splits": {}}
        for split in SPLITS:
            vectors = correctness_vectors(pools[POOL_NAME[split]], split, gens, {})
            got_sf = sf_interval(vectors["aligned"], vectors["reversed"])
            got_acc = {
                "aligned": acc_interval(vectors["aligned"]),
                "reversed": acc_interval(vectors["reversed"]),
            }
            exp_sf = block["SF"][split]
            exp_acc = block["acc"][split]
            for name, got, exp in (
                ("SF", got_sf, exp_sf),
                ("aligned_acc", got_acc["aligned"], exp_acc["aligned"]),
                ("reversed_acc", got_acc["reversed"], exp_acc["reversed"]),
            ):
                if len(got) != 3 or len(exp) != 3 or any(not _close(a, b) for a, b in zip(got, exp)):
                    raise RuntimeError(f"{cond} {split} {name} {got} != rescore {exp}")
            n_aligned_correct = int((vectors["aligned"] == 1).sum())
            n_flip = int(((vectors["aligned"] == 1) & (vectors["reversed"] == 0)).sum())
            exp_counts = block["counts"][split]
            if n_aligned_correct != exp_counts["n_aligned_correct"] or n_flip != exp_counts["n_flip"]:
                raise RuntimeError(f"{cond} {split} flip counts do not match rescore")
            checked[cond]["splits"][split] = {
                "reversed_SF": got_sf,
                "aligned": vectors["aligned"],
                "reversed": vectors["reversed"],
                "hashes": vectors["hashes"],
            }
    return checked


def _install_log() -> None:
    path = guard_output(LOG_PATH)
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


def _pending(items: list[dict], path: Path) -> list[dict]:
    done = read_jsonl(path) if path.exists() else {}
    return [item for item in items if item["uid"] not in done]


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


def run_generate(tok, model, items: list[dict], path: Path) -> int:
    import torch
    from .infer import generate

    path = guard_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for batch_size in BATCH_SIZES:
        try:
            generate(
                tok,
                model,
                items,
                path,
                max_new_tokens=MAX_NEW_TOKENS,
                batch_size=batch_size,
                shuffle_seed=BOOT_SEED,
            )
            return batch_size
        except RuntimeError as exc:
            last = exc
            if not _is_oom(exc) or batch_size == BATCH_SIZES[-1]:
                raise
            log(f"OOM at batch {batch_size}: {exc}")
            torch.cuda.empty_cache()
    raise last


def generate_condition(cond: str, items: list[dict]) -> dict:
    from .e1 import load_for_eval

    out_dir = guard_output(OUT_ROOT / cond)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_path = guard_output(out_dir / "generations.jsonl")
    run_path = guard_output(out_dir / "run.json")
    pending = _pending(items, gen_path)
    log(f"{cond} pending {len(pending)} / {len(items)}")
    if not pending:
        if run_path.exists():
            return json.loads(run_path.read_text(encoding="utf-8"))
        info = {"condition": cond, "skipped": True, "batch_size": None, "reason": "generations already complete"}
        _write_json(run_path, info)
        return info
    adapter = ADAPTER[cond]
    adapter_arg = None if adapter is None else str(adapter)
    if adapter is not None and not adapter.exists():
        raise RuntimeError(f"missing adapter {adapter}")
    log(f"{cond} load_for_eval adapter={adapter_arg}")
    tok, model = load_for_eval(adapter_arg)
    try:
        batch_size = run_generate(tok, model, items, gen_path)
    finally:
        _free(model)
    info = {
        "condition": cond,
        "skipped": False,
        "batch_size": batch_size,
        "max_new_tokens": MAX_NEW_TOKENS,
        "decoding": "greedy",
        "cot": True,
        "adapter": None if adapter is None else "build/e1/pilot/b2/adapter",
        "load": "e1.load_for_eval",
        "base_dtype": "bfloat16",
    }
    _write_json(run_path, info)
    log(f"{cond} generation finished batch_size={batch_size}")
    return info


def _write_json(path: Path, obj) -> None:
    path = guard_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _flip_counts(aligned: np.ndarray, other: np.ndarray) -> dict:
    chosen = aligned == 1
    flipped = chosen & (other == 0)
    return {
        "n": int(aligned.size),
        "n_aligned_correct": int(chosen.sum()),
        "n_flip": int(flipped.sum()),
        "n_stay": int((chosen & (other == 1)).sum()),
    }


def analyze(pools: dict, rescore: dict, run_info: dict) -> dict:
    pilot = rescore["rounds"]["pilot"]["conditions"]
    conditions = {}
    for cond in CONDITIONS:
        gen_path = OUT_ROOT / cond / "generations.jsonl"
        synonym_gens = read_jsonl(gen_path)
        pilot_gens = read_jsonl(PILOT_LOGIC[cond])
        per_split = {}
        rows = []
        for split in SPLITS:
            vectors = correctness_vectors(
                pools[POOL_NAME[split]], split, pilot_gens, synonym_gens,
            )
            if vectors["synonym"] is None or vectors["synonym"].size != N_TEST:
                raise RuntimeError(f"{cond} {split} synonym generations are incomplete")
            synonym_sf = sf_interval(vectors["aligned"], vectors["synonym"])
            reversed_sf = pilot[cond]["SF"][split]
            if any(not _close(a, b) for a, b in zip(sf_interval(vectors["aligned"], vectors["reversed"]), reversed_sf)):
                raise RuntimeError(f"{cond} {split} reversed SF drifted from rescore")
            counts = _flip_counts(vectors["aligned"], vectors["synonym"])
            extract_fail = int(sum(row.get("synonym_extract_fail", 0) for row in vectors["rows"]))
            per_split[split] = {
                "synonym_SF": synonym_sf,
                "reversed_SF": [float(x) for x in reversed_sf],
                "interval_relation": interval_relation(synonym_sf, [float(x) for x in reversed_sf]),
                "counts": counts,
                "synonym_acc": acc_interval(vectors["synonym"]),
                "aligned_acc": acc_interval(vectors["aligned"]),
                "synonym_extract_fail": extract_fail,
                "synonym_extract_fail_rate": extract_fail / N_TEST,
            }
            rows.extend(vectors["rows"])
        rows_path = guard_output(OUT_ROOT / cond / "rows.jsonl")
        with rows_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        conditions[cond] = {
            "splits": per_split,
            "run": run_info.get(cond),
            "generations": str(gen_path.relative_to(PROJECT)).replace("\\", "/"),
            "aligned_source": str(PILOT_LOGIC[cond].relative_to(PROJECT)).replace("\\", "/"),
        }
    payload = {
        "training": False,
        "round": "pilot",
        "conditions_run": list(CONDITIONS),
        "SF_definition": "P(synonym wrong | aligned right)",
        "interval": "[point, low, high]",
        "n_boot": N_BOOT,
        "bootstrap_seed": BOOT_SEED,
        "paired_unit": "latent_task",
        "hash_order": "sorted graph_hash",
        "pool_seed": 0,
        "decoding": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS, "cot": True},
        "prompt": "logic_prompt",
        "presentation": "synonym",
        "ambiguous_included": False,
        "eval_load": {
            "function": "e1.load_for_eval",
            "b0": {"adapter": None, "base_dtype": "bfloat16", "source_metrics_quant": "bf16-eval"},
            "b2": {
                "adapter": "build/e1/pilot/b2/adapter",
                "base_dtype": "bfloat16",
                "adapter_application": "merge_and_unload",
                "source_metrics_quant": "nf4",
            },
            "note": LOAD_NOTE,
        },
        "reversed_SF_source": "experiments/e1/rescore/metrics.json",
        "conditions": conditions,
    }
    _write_json(METRICS_PATH, payload)
    log("wrote " + str(METRICS_PATH.relative_to(PROJECT)))
    return payload


def main() -> None:
    os_home = PROJECT / ".cache" / "huggingface"
    import os

    os.environ["HF_HOME"] = str(os_home)
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _install_log()
    try:
        log("synonym eval start; training false")
        pools, summary = build_pools(seed=0)
        if not summary.get("ok"):
            raise RuntimeError("pool isolation failed")
        log("checking synonym sentences against the solver")
        check_pools(pools)
        log("solver check passed")
        rescore = load_rescore()
        require_reversed_matches_rescore(pools, rescore)
        log("reversed SF matches experiments/e1/rescore/metrics.json")
        items, _meta = build_synonym_items(pools)
        if len(items) != N_TEST * len(SPLITS):
            raise RuntimeError(f"expected {N_TEST * len(SPLITS)} synonym items, got {len(items)}")
        run_info = {}
        for cond in CONDITIONS:
            run_info[cond] = generate_condition(cond, items)
        payload = analyze(pools, rescore, run_info)
        log(json.dumps({
            cond: {
                split: {
                    "synonym_SF": payload["conditions"][cond]["splits"][split]["synonym_SF"],
                    "reversed_SF": payload["conditions"][cond]["splits"][split]["reversed_SF"],
                    "interval_relation": payload["conditions"][cond]["splits"][split]["interval_relation"],
                }
                for split in SPLITS
            }
            for cond in CONDITIONS
        }, ensure_ascii=False))
        log("SYNONYM COMPLETE")
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
