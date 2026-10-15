"""E1 pilot data budget, isolation, and mining rank. No GPU."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from structrobust import e1, logic
from structrobust.prompts import logic_prompt


def _rule_lines(text: str) -> list[str]:
    rules = text.split("Rules:\n", 1)[1].split("\n\nQuestion:", 1)[0]
    return [line for line in rules.splitlines() if line]


@pytest.fixture(scope="module")
def built():
    pools, summary = e1.build_pools()
    return pools, summary


def test_pools_are_isolated_and_balanced(built):
    pools, summary = built
    assert summary["ok"]
    assert all(v == 0 for v in summary["overlaps"].values())
    assert summary["pools"]["train"]["n"] == 500
    assert summary["pools"]["val"]["n"] == 200
    assert summary["pools"]["test_seen"]["n"] == 200
    assert summary["pools"]["test_unseen"]["n"] == 200
    assert summary["pools"]["train"]["n_yes"] == summary["pools"]["train"]["n_no"] == 250
    assert summary["tags"]["test_seen"] == "e1test"
    assert summary["tags"]["test_unseen"] == "e1test_unseen"
    assert summary["tags"]["excluded_e0_tag"] == "e0"
    assert summary["shifted_presentation"] == "reversed"
    assert summary["slot_random_unique_max"] < e1.K_MINE
    for lat in pools["train"] + pools["val"] + pools["test_seen"] + pools["test_unseen"]:
        assert lat.depth == 2


def test_step_budget_uses_every_sequence_once():
    sizes, remine_after = e1.optimizer_step_plan()
    assert sum(sizes) == 1000
    assert sizes[:-1] == [16] * 62
    assert sizes[-1] == 8
    assert len(sizes) == 63
    assert remine_after == 31
    assert sum(sizes[:remine_after]) == 496
    assert sum(sizes[remine_after:]) == 504
    assert e1.warmup_steps(len(sizes)) == 2


def test_learning_rate_warms_up_then_decays():
    total = 63
    warmup = e1.warmup_steps(total)
    assert e1.lr_at(0, total, warmup) == pytest.approx(e1.LR / 2)
    assert e1.lr_at(warmup - 1, total, warmup) == pytest.approx(e1.LR)
    assert 0 <= e1.lr_at(total - 1, total, warmup) < e1.lr_at(warmup, total, warmup)


def test_training_renders_and_shared_b2_b3_orders(built):
    pools, _ = built
    for lat in pools["train"][:12]:
        aligned = logic.render(lat, "aligned")
        left, right = e1.training_renders(lat, "b1")
        assert left["text"] == right["text"] == aligned["text"]
        b2 = e1.training_renders(lat, "b2")
        b3 = e1.training_renders(lat, "b3")
        assert [item["text"] for item in b2] == [item["text"] for item in b3]
        assert b2[0]["text"] != b2[1]["text"]
        base_lines = _rule_lines(aligned["text"])
        for rendered in b2:
            assert sorted(_rule_lines(rendered["text"])) == sorted(base_lines)
        mined = e1.sample_renders(lat, e1.K_MINE, "bgm-k8")
        assert len({item["text"] for item in mined}) == e1.K_MINE
        for rendered in mined:
            assert sorted(_rule_lines(rendered["text"])) == sorted(base_lines)
        trace = logic.solver_trace(lat)
        assert trace.endswith(f"Answer: {lat.label}")


def test_eval_cells_use_reversed_as_the_shifted_view(built):
    pools, _ = built
    items, meta = e1.build_eval_items(pools["test_seen"], "seen")
    unseen_items, unseen_meta = e1.build_eval_items(pools["test_unseen"], "unseen")
    meta.update(unseen_meta)
    items.extend(unseen_items)
    counts = {}
    for row in meta.values():
        counts[row["cell"]] = counts.get(row["cell"], 0) + 1
        max_inv = row["n_core"] * (row["n_core"] - 1) / 2
        if row["presentation"] == "aligned":
            assert row["back_reads"] == 0
        else:
            assert row["presentation"] == "reversed"
            assert row["cell"].endswith("shifted")
            assert row["back_reads"] == max_inv
    assert counts == {
        "seen_aligned": 200,
        "seen_shifted": 200,
        "unseen_aligned": 200,
        "unseen_shifted": 200,
    }
    prompts = {item["uid"]: item["prompt"] for item in items}
    assert set(prompts) == set(meta)
    sample = next(iter(meta))
    assert prompts[sample].startswith(logic_prompt("")[:20])


def test_mining_rank_adds_standardized_nll_and_burden():
    scores, order = e1.rank_presentations([0.0, 0.0, 0.0, 3.0], [0.0, 5.0, 0.0, 5.0])
    assert order[0] == 3
    assert order[1] == 1
    assert scores[order[0]] > scores[order[1]] > scores[order[2]]
    _, tied = e1.rank_presentations([1, 1, 1, 1], [0, 2, 1, 0.5])
    assert tied[0] == 1
    flat_scores, flat_order = e1.rank_presentations([1, 1, 1], [4, 4, 4])
    assert flat_scores == [0.0, 0.0, 0.0]
    assert flat_order == [0, 1, 2]


def test_supervised_encoding_matches_the_eval_prompt():
    cache = Path(__file__).resolve().parents[1] / ".cache" / "huggingface"
    if not cache.exists():
        pytest.skip("local tokenizer cache is not present")
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(e1.MODEL_NAME, local_files_only=True)
    yes_id, no_id = e1.yes_no_token_ids(tok)
    for label in ("yes", "no"):
        lat = logic.make_latent("chain", 2, label, seed=logic.stable_seed("encode", label), vocab="train")
        rendered = e1.canonical_render(lat)
        example = e1.examples_from_renders(tok, lat, [rendered], yes_id, no_id)[0]
        from structrobust.infer import chat

        prompt_ids = e1._as_ids(tok(chat(tok, logic_prompt(rendered["text"]))))
        assert example["input_ids"][:len(prompt_ids)] == prompt_ids
        assert example["answer_pos"] >= len(prompt_ids)
        gold = yes_id if label == "yes" else no_id
        assert example["input_ids"][example["answer_pos"]] == gold
        assert example["labels"][example["answer_pos"]] == gold
        assert all(token == -100 for token in example["labels"][:len(prompt_ids)])
        assert any(token != -100 for token in example["labels"][len(prompt_ids):])
        assert len(example["input_ids"]) <= e1.MAX_LEN


def test_gsm_mix_keeps_every_synthetic_sequence():
    counts = e1.gsm_mix_counts()
    assert counts["n_synthetic_sequences"] == 1000
    assert counts["n_gsm_sequences"] == 176
    assert counts["n_gsm_questions"] == 88
    assert counts["n_total_sequences"] == 1176
    assert counts["gsm_fraction"] == pytest.approx(176 / 1176)
    assert abs(counts["gsm_fraction"] - 0.15) < 0.001
    sizes, remine_after = e1.optimizer_step_plan(counts["n_total_sequences"])
    assert sum(sizes) == 1176
    assert sizes[:-1] == [16] * 73
    assert sizes[-1] == 8
    assert len(sizes) == 74
    assert remine_after == 37
    assert all(n % e1.MICRO_BATCH == 0 for n in sizes)
    # The unmixed pilot budget stays on the default argument.
    plain, plain_remine = e1.optimizer_step_plan()
    assert sum(plain) == 1000
    assert plain_remine == 31


def test_gsm_trace_strips_calculator_markup_and_ends_on_answer():
    answer = (
        "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
        "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n"
        "#### 72"
    )
    assert e1.gsm_supervised_trace(answer) == (
        "Natalia sold 48/2 = 24 clips in May.\n"
        "Natalia sold 48+24 = 72 clips altogether in April and May.\n"
        "Answer: 72"
    )


def test_gsm_train_selection_blocks_held_out_text():
    questions = [f"question {i}" for i in range(200)]
    blocked = {"question 0", "question   2", "question 3"}
    chosen = e1.select_gsm_train_indices(questions, blocked, n_questions=10, seed=0)
    assert chosen == sorted(chosen)
    assert len(chosen) == 10
    assert set(chosen).isdisjoint({0, 2, 3})
    assert chosen == e1.select_gsm_train_indices(questions, blocked, n_questions=10, seed=0)
    other = e1.select_gsm_train_indices(questions, blocked, n_questions=10, seed=1)
    assert other != chosen
    evens = e1.select_gsm_train_indices(
        questions, set(), n_questions=4, seed=0, accept=lambda i: i % 2 == 0,
    )
    assert len(evens) == 4
    assert all(i % 2 == 0 for i in evens)
    with pytest.raises(RuntimeError):
        e1.select_gsm_train_indices(["only"], set(), n_questions=2, seed=0)


def test_gsm_encoding_matches_the_arith_prompt():
    cache = Path(__file__).resolve().parents[1] / ".cache" / "huggingface"
    if not cache.exists():
        pytest.skip("local tokenizer cache is not present")
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    from structrobust.infer import chat
    from structrobust.prompts import arith_prompt

    tok = AutoTokenizer.from_pretrained(e1.MODEL_NAME, local_files_only=True)
    question = "Natalia sold clips to 48 of her friends in April. How many clips did she sell?"
    answer = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n#### 24"
    example = e1.encode_gsm_example(tok, question, answer, "gsm8k-train|7")
    left, right = e1.gsm_training_pair(example)
    assert left["task_id"] == right["task_id"] == "gsm8k-train|7"
    assert left["input_ids"] == right["input_ids"]
    assert left["labels"] == right["labels"]
    prompt_ids = e1._as_ids(tok(chat(tok, arith_prompt(question))))
    assert example["input_ids"][:len(prompt_ids)] == prompt_ids
    assert all(token == -100 for token in example["labels"][:len(prompt_ids)])
    assert any(token != -100 for token in example["labels"][len(prompt_ids):])
    assert "Answer: 24" in tok.decode(example["input_ids"][len(prompt_ids):])
    assert len(example["input_ids"]) <= e1.MAX_LEN
