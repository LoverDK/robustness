"""Run a small Hugging Face instruction model on the MVP dataset.

Optional dependencies:
    pip install torch transformers accelerate

The script writes raw generations and the compact prediction CSV consumed by
structshift_mvp.py. It does not train or alter the model.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path


def load_records(path: Path, limit: int | None) -> list[dict[str, object]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[:limit] if limit else rows


def prompt_for(text: str) -> str:
    return (
        "Solve the symbolic reasoning question. Return exactly one token: yes or no.\n"
        "Do not explain your answer.\n\n"
        f"{text}\n"
        "Final answer:"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("build/mvp_predictions"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing optional dependencies. Install torch, transformers, and accelerate first."
        ) from exc

    records = load_records(args.dataset, args.limit)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.csv"
    raw_path = args.output_dir / "raw_generations.jsonl"
    started = time.time()
    with predictions_path.open("w", newline="", encoding="utf-8") as pred_handle, raw_path.open(
        "w", encoding="utf-8"
    ) as raw_handle:
        writer = csv.DictWriter(pred_handle, fieldnames=["model", "sample_id", "prediction"])
        writer.writeheader()
        for index, record in enumerate(records, start=1):
            prompt = prompt_for(str(record["text"]))
            if hasattr(tokenizer, "apply_chat_template"):
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                rendered = prompt
            inputs = tokenizer(rendered, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
            continuation = output[0][inputs["input_ids"].shape[-1]:]
            raw = tokenizer.decode(continuation, skip_special_tokens=True)
            writer.writerow({
                "model": args.model,
                "sample_id": record["sample_id"],
                "prediction": raw,
            })
            raw_handle.write(json.dumps({
                "model": args.model,
                "sample_id": record["sample_id"],
                "prompt": rendered,
                "raw_generation": raw,
            }, ensure_ascii=False) + "\n")
            if index % 25 == 0:
                print(f"{index}/{len(records)} completed")
    metadata = {
        "model": args.model,
        "device": device,
        "n_records": len(records),
        "seconds": round(time.time() - started, 2),
        "deterministic": True,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
