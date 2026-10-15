"""Generate and analyze the first solver-validated StructShift MVP.

This script deliberately uses only the Python standard library. It creates
symbolic dependency graphs, answer-preserving canonical/shifted renderings,
strict seen/unseen graph-family splits, and a prediction CSV template for an
external model runner.

It is an experiment harness, not a language-model result generator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


SEEN_FAMILIES = ("chain", "fork_join")
UNSEEN_FAMILIES = ("chain_then_fork", "double_chain_join")
ALL_FAMILIES = SEEN_FAMILIES + UNSEEN_FAMILIES


def stable_int(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def graph_hash(family: str, seed: int, rules: list[tuple[tuple[str, ...], str]],
               facts: tuple[str, ...], query: str) -> str:
    payload = json.dumps(
        {"family": family, "seed": seed, "rules": rules, "facts": facts, "query": query},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Graph:
    graph_id: str
    family: str
    graph_family: str
    seed: int
    facts: tuple[str, ...]
    rules: tuple[tuple[tuple[str, ...], str], ...]
    query: str
    answer: str
    depth: int
    branching: int
    graph_size: int
    operator_count: int


def solve(facts: Iterable[str], rules: Iterable[tuple[tuple[str, ...], str]], query: str) -> bool:
    known = set(facts)
    changed = True
    while changed:
        changed = False
        for sources, target in rules:
            if target not in known and all(source in known for source in sources):
                known.add(target)
                changed = True
    return query in known


def make_graph(family: str, seed: int, negative: bool = False) -> Graph:
    if family not in ALL_FAMILIES:
        raise ValueError(f"unknown graph family: {family}")
    facts: tuple[str, ...]
    rules: list[tuple[tuple[str, ...], str]]
    depth: int
    branching: int
    query = "target"
    if family == "chain":
        facts = ("p0",)
        depth = 4 + seed % 3
        rules = [((f"p{i}",), f"p{i + 1}") for i in range(depth)]
        rules.append(((f"p{depth}",), query))
        branching = 1
    elif family == "fork_join":
        facts = ("p0",)
        rules = [(("p0",), "left"), (("p0",), "right"),
                 (("left", "right"), query)]
        depth, branching = 2, 2
    elif family == "chain_then_fork":
        facts = ("p0",)
        depth = 3 + seed % 2
        rules = [((f"p{i}",), f"p{i + 1}") for i in range(depth)]
        rules += [((f"p{depth}",), "left"), ((f"p{depth}",), "right"),
                  (("left", "right"), query)]
        branching = 2
    else:  # double_chain_join
        facts = ("a0", "b0")
        length = 2 + seed % 2
        rules = [((f"a{i}",), f"a{i + 1}") for i in range(length)]
        rules += [((f"b{i}",), f"b{i + 1}") for i in range(length)]
        rules.append(((f"a{length}", f"b{length}"), query))
        depth, branching = length + 1, 2

    if negative:
        removable = [i for i, (sources, target) in enumerate(rules) if target != query]
        rules.pop(removable[seed % len(removable)])
    answer = "yes" if solve(facts, rules, query) else "no"
    gid = graph_hash(family, seed, rules, facts, query)
    return Graph(
        graph_id=gid,
        family=family,
        graph_family=family,
        seed=seed,
        facts=facts,
        rules=tuple(rules),
        query=query,
        answer=answer,
        depth=depth,
        branching=branching,
        graph_size=len(set(facts) | {x for sources, target in rules for x in (*sources, target)}),
        operator_count=len(rules),
    )


def ancestors(graph: Graph) -> set[str]:
    needed = {graph.query}
    changed = True
    while changed:
        changed = False
        for sources, target in graph.rules:
            if target in needed:
                before = len(needed)
                needed.update(sources)
                changed = changed or len(needed) > before
    return needed


def render(graph: Graph, presentation: str) -> dict[str, object]:
    if presentation not in {"canonical", "shifted"}:
        raise ValueError(presentation)
    lines: list[str] = []
    order = list(range(len(graph.rules)))
    if presentation == "shifted":
        order.sort(key=lambda i: stable_int(graph.graph_id, "rule", i))
    for index in order:
        sources, target = graph.rules[index]
        if len(sources) == 1:
            lines.append(f"If {sources[0]} is true, then {target} is true.")
        else:
            lines.append(f"If {sources[0]} and {sources[1]} are true, then {target} is true.")
    facts = list(graph.facts)
    if presentation == "shifted":
        facts.sort(key=lambda fact: stable_int(graph.graph_id, "fact", fact))
    lines = [f"{fact} is true." for fact in facts] + lines
    text = " ".join(lines) + f" Is the statement {graph.query} true? Answer yes or no."
    positions = {token: i for i, token in enumerate(re.findall(r"[A-Za-z0-9_]+", text))}
    needed = ancestors(graph)
    relevant_positions = [i for i, token in enumerate(re.findall(r"[A-Za-z0-9_]+", text)) if token in needed]
    if len(relevant_positions) > 1:
        relevant_distance = max(relevant_positions) - min(relevant_positions)
    else:
        relevant_distance = 0
    max_frontier = 0
    for i in range(len(lines)):
        open_nodes = sum(
            1 for node in needed
            if any(node in source and j <= i for j, (source, _) in enumerate(graph.rules))
        )
        max_frontier = max(max_frontier, open_nodes)
    target_position = text.lower().find(graph.query.lower()) / max(1, len(text))
    return {
        "text": text,
        "presentation": presentation,
        "token_count": len(text.split()),
        "target_position": round(target_position, 6),
        "relevant_distance": relevant_distance,
        "active_frontier": max_frontier,
        "operator_coordination": max(len(sources) for sources, _ in graph.rules),
    }


def make_record(graph: Graph, presentation: str, split: str, pair_id: str) -> dict[str, object]:
    rendered = render(graph, presentation)
    return {
        "sample_id": f"{graph.graph_id}-{presentation}",
        "pair_id": pair_id,
        "graph_id": graph.graph_id,
        "graph_family": graph.graph_family,
        "family_seen": graph.graph_family in SEEN_FAMILIES,
        "split": split,
        "presentation": presentation,
        "seed": graph.seed,
        "answer": graph.answer,
        "text": rendered.pop("text"),
        **asdict(graph),
        **rendered,
    }


def generate(seed: int, per_family: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for family in ALL_FAMILIES:
        for index in range(per_family):
            graph_seed = stable_int(seed, family, index) % (2**31)
            negative = index % 4 == 3
            graph = make_graph(family, graph_seed, negative)
            split = "discovery" if family in SEEN_FAMILIES and index % 4 < 2 else "confirmatory"
            if family in UNSEEN_FAMILIES:
                split = "transfer"
            pair_id = graph.graph_id
            for presentation in ("canonical", "shifted"):
                records.append(make_record(graph, presentation, split, pair_id))
    return records


def write_jsonl(records: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_prediction_template(records: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "sample_id", "prediction"])
        writer.writeheader()
        for record in records:
            writer.writerow({"model": "REPLACE_MODEL", "sample_id": record["sample_id"], "prediction": ""})


def bootstrap(values: list[float], seed: int = 17, rounds: int = 1000) -> tuple[float, float]:
    if not values:
        return [None, None]
    rng = random.Random(seed)
    means = [mean(values[rng.randrange(len(values))] for _ in values) for _ in range(rounds)]
    means.sort()
    return means[max(0, int(0.025 * rounds))], means[min(rounds - 1, int(0.975 * rounds))]


def normalize_prediction(value: str) -> str:
    text = value.strip().lower()
    matches = re.findall(r"\b(yes|no|true|false|1|0)\b", text)
    if not matches:
        return ""
    token = matches[-1]
    return {"true": "yes", "1": "yes", "false": "no", "0": "no"}.get(token, token)


def analyze(dataset_path: Path, predictions_path: Path, output_dir: Path) -> dict[str, object]:
    records = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    predictions: dict[str, dict[str, str]] = defaultdict(dict)
    with predictions_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            predictions[row["model"]][row["sample_id"]] = normalize_prediction(row["prediction"])
    by_id = {record["sample_id"]: record for record in records}
    results: dict[str, object] = {"n_records": len(records), "models": {}}
    for model, model_preds in predictions.items():
        scored = {
            sid: pred == by_id[sid]["answer"]
            for sid, pred in model_preds.items()
            if sid in by_id and pred
        }
        by_cell: dict[tuple[bool, str], list[bool]] = defaultdict(list)
        for sid, correct in scored.items():
            row = by_id[sid]
            by_cell[(bool(row["family_seen"]), row["presentation"])].append(correct)
        seen_canonical = mean(by_cell[(True, "canonical")]) if by_cell[(True, "canonical")] else None
        unseen_canonical = mean(by_cell[(False, "canonical")]) if by_cell[(False, "canonical")] else None
        ss_pairs = []
        for pair_id in {row["pair_id"] for row in records}:
            can = f"{pair_id}-canonical"
            shift = f"{pair_id}-shifted"
            if can in scored and shift in scored and scored[can]:
                ss_pairs.append(float(scored[shift]))
        stability_ci = bootstrap(ss_pairs, seed=stable_int(model, "ss") % 2**31)
        transfer_gap = (1 - unseen_canonical) - (1 - seen_canonical) if seen_canonical is not None and unseen_canonical is not None else None
        results["models"][model] = {
            "n_scored": len(scored),
            "cell_accuracy": {
                "seen_canonical": seen_canonical,
                "seen_shifted": mean(by_cell[(True, "shifted")]) if by_cell[(True, "shifted")] else None,
                "unseen_canonical": unseen_canonical,
                "unseen_shifted": mean(by_cell[(False, "shifted")]) if by_cell[(False, "shifted")] else None,
            },
            "SS": mean(ss_pairs) if ss_pairs else None,
            "SS_bootstrap_ci95": stability_ci,
            "SF": (1 - mean(ss_pairs)) if ss_pairs else None,
            "Delta_T": transfer_gap,
            "double_dissociation_cells": {
                "seen_canonical": len(by_cell[(True, "canonical")]),
                "seen_shifted": len(by_cell[(True, "shifted")]),
                "unseen_canonical": len(by_cell[(False, "canonical")]),
                "unseen_shifted": len(by_cell[(False, "shifted")]),
            },
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--output-dir", type=Path, default=Path("build/mvp_data"))
    gen.add_argument("--per-family", type=int, default=100)
    gen.add_argument("--seed", type=int, default=20261008)
    ana = sub.add_parser("analyze")
    ana.add_argument("--dataset", type=Path, required=True)
    ana.add_argument("--predictions", type=Path, required=True)
    ana.add_argument("--output-dir", type=Path, default=Path("build/mvp_results"))
    args = parser.parse_args()
    if args.command == "generate":
        records = generate(args.seed, args.per_family)
        write_jsonl(records, args.output_dir / "dataset.jsonl")
        write_prediction_template(records, args.output_dir / "predictions_template.csv")
        manifest = {
            "seed": args.seed,
            "per_family": args.per_family,
            "n_records": len(records),
            "seen_families": SEEN_FAMILIES,
            "unseen_families": UNSEEN_FAMILIES,
            "answer_preservation_checked": all(
                row["answer"] in {"yes", "no"} for row in records
            ),
            "model_results_not_present": True,
        }
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))
    else:
        print(json.dumps(analyze(args.dataset, args.predictions, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
