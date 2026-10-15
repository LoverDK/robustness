import csv
import json
import tempfile
import unittest
from pathlib import Path

from structshift_mvp import (
    analyze,
    generate,
    normalize_prediction,
    write_jsonl,
    write_prediction_template,
)


class StructShiftMvpTest(unittest.TestCase):
    def test_graph_families_are_split_before_rendering(self):
        records = generate(seed=11, per_family=8)
        self.assertEqual(len(records), 4 * 8 * 2)
        seen = {row["graph_family"] for row in records if row["family_seen"]}
        unseen = {row["graph_family"] for row in records if not row["family_seen"]}
        self.assertTrue(seen.isdisjoint(unseen))

    def test_same_graph_pair_preserves_answer(self):
        records = generate(seed=13, per_family=5)
        grouped = {}
        for row in records:
            grouped.setdefault(row["pair_id"], {})[row["presentation"]] = row
        self.assertTrue(all(pair["canonical"]["answer"] == pair["shifted"]["answer"] for pair in grouped.values()))
        self.assertTrue(all(pair["canonical"]["graph_id"] == pair["shifted"]["graph_id"] for pair in grouped.values()))

    def test_template_is_auditable(self):
        records = generate(seed=17, per_family=2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_jsonl(records, root / "dataset.jsonl")
            write_prediction_template(records, root / "predictions.csv")
            self.assertEqual(len((root / "dataset.jsonl").read_text(encoding="utf-8").splitlines()), len(records))
            with (root / "predictions.csv").open(encoding="utf-8") as handle:
                self.assertEqual(sum(1 for _ in csv.DictReader(handle)), len(records))
        self.assertEqual(normalize_prediction("The answer is YES."), "yes")
        self.assertEqual(normalize_prediction("no"), "no")

    def test_oracle_predictions_recover_metrics(self):
        records = generate(seed=19, per_family=4)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "dataset.jsonl"
            predictions = root / "predictions.csv"
            output = root / "results"
            write_jsonl(records, dataset)
            with predictions.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["model", "sample_id", "prediction"])
                writer.writeheader()
                for row in records:
                    writer.writerow({
                        "model": "oracle",
                        "sample_id": row["sample_id"],
                        "prediction": row["answer"],
                    })
            metrics = analyze(dataset, predictions, output)
        oracle = metrics["models"]["oracle"]
        self.assertEqual(oracle["n_scored"], len(records))
        self.assertEqual(oracle["SS"], 1.0)
        self.assertEqual(oracle["SF"], 0.0)
        self.assertEqual(oracle["Delta_T"], 0.0)


if __name__ == "__main__":
    unittest.main()
