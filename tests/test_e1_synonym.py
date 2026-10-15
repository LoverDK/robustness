"""Synonym sentences on the E1 pilot pools do not change the solver answer."""

from __future__ import annotations

import json
import sys

import pytest

from structrobust import wording
from structrobust.e1 import build_pools, graph_hash, read_jsonl
from structrobust.e1_synonym import (
    ADAPTER,
    CONDITIONS,
    OUT_ROOT,
    PILOT_LOGIC,
    SPLITS,
    assert_synonym_preserves_solver,
    build_synonym_items,
    check_pools,
    correctness_vectors,
    guard_output,
    interval_relation,
    parse_synonym_sentence,
    require_reversed_matches_rescore,
)
from structrobust.prompts import extract_yes_no, logic_prompt


@pytest.fixture(scope="module")
def pools():
    assert "torch" not in sys.modules
    kept, summary = build_pools(seed=0)
    assert summary["ok"]
    assert "torch" not in sys.modules
    return kept


def test_synonym_keeps_solver_answer_on_pilot_pools(pools):
    assert "torch" not in sys.modules
    check_pools(pools)
    n = 0
    arities = set()
    for split in ("test_seen", "test_unseen"):
        assert len(pools[split]) == 200
        for lat in pools[split]:
            assert_synonym_preserves_solver(lat)
            arities.add(len(lat.core_rules[0][0]))
            n += 1
    assert n == 400
    assert arities <= {1, 2}
    assert 1 in arities
    assert "torch" not in sys.modules


def test_binary_synonym_parses_to_the_same_rule():
    rule = (("amber", "azure"), "coral")
    parsed = parse_synonym_sentence(wording.synonym_sentence("Kara", rule))
    assert parsed == rule
    unary = (("amber",), "coral")
    assert parse_synonym_sentence(wording.synonym_sentence("Kara", unary)) == unary


def test_items_are_synonym_only(pools):
    items, meta = build_synonym_items(pools)
    assert len(items) == 400
    assert len({item["uid"] for item in items}) == 400
    assert {row["presentation"] for row in meta.values()} == {"synonym"}
    assert {row["split"] for row in meta.values()} == {"seen", "unseen"}
    for item in items:
        assert item["prompt"].startswith(logic_prompt("")[:20])
        assert "likes" not in item["prompt"]
        assert " is always " in item["prompt"]
    for split, pool_name in (("seen", "test_seen"), ("unseen", "test_unseen")):
        hashes = {graph_hash(lat) for lat in pools[pool_name]}
        got = {row["graph_hash"] for row in meta.values() if row["split"] == split}
        assert got == hashes


def test_reversed_pairing_matches_rescore(pools):
    require_reversed_matches_rescore(pools)


def test_last_duplicate_line_is_kept(tmp_path, pools):
    lat = pools["test_seen"][0]
    digest = graph_hash(lat)
    uid = f"seen_aligned|{digest}"
    other = f"seen_shifted|{digest}"
    path = tmp_path / "logic.jsonl"
    first = "yes" if lat.label == "no" else "no"
    lines = [
        {"uid": uid, "generation": f"Answer: {first}"},
        {"uid": other, "generation": f"Answer: {lat.label}"},
        {"uid": uid, "generation": f"Answer: {lat.label}"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in lines), encoding="utf-8")
    gens = read_jsonl(path)
    assert extract_yes_no(gens[uid]["generation"]) == lat.label
    vectors = correctness_vectors([lat], "seen", gens, {})
    assert vectors["aligned"].tolist() == [1.0]
    assert vectors["reversed"].tolist() == [1.0]


def test_interval_relation_matches_endpoint_rule():
    assert interval_relation([0.2, 0.1, 0.3], [0.25, 0.2, 0.4]) == "重叠"
    assert interval_relation([0.2, 0.1, 0.3], [0.4, 0.31, 0.5]) == "不重叠"
    assert interval_relation([0.2, 0.1, 0.3], [0.4, 0.3, 0.5]) == "重叠"


def test_refuses_banned_output_paths():
    banned = [
        "build/e0/synonym",
        "build/e1/pilot/b0/logic.jsonl",
        "build/e1/pilot/b2/logic.jsonl",
        "experiments/e1/pilot/b0/metrics.json",
        "experiments/e1/pilot/b2/metrics.json",
        "experiments/e1/rescore/metrics.json",
        "build/e0/wording/metrics.json",
    ]
    for rel in banned:
        with pytest.raises(SystemExit):
            guard_output(__import__("pathlib").Path(rel))
    guard_output(OUT_ROOT / "b0" / "generations.jsonl")
    guard_output(__import__("pathlib").Path("experiments/e1/synonym/metrics.json"))
    guard_output(__import__("pathlib").Path("build/logs/e1_synonym.log"))


def test_sources_are_the_saved_pilot_files():
    assert ADAPTER["b0"] is None
    assert ADAPTER["b2"].name == "adapter"
    assert PILOT_LOGIC["b0"].exists()
    assert PILOT_LOGIC["b2"].exists()
    assert ADAPTER["b2"].exists()
    assert CONDITIONS == ("b0", "b2")
    assert SPLITS == ("seen", "unseen")
