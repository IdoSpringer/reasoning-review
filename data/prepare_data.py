#!/usr/bin/env python3
"""Download GGBench and build train/heldout JSONL for LoRA SFT.

Output records: {"id", "question", "target", "reference_code"}
  - question: GGBench problem text (model input)
  - target:   <think> + GGBench reference steps + final code (model output)
  - reference_code: last geogebra code block of the reference (for scoring)

CPU-safe, stdlib only. Run:  python3 data/prepare_data.py
"""
import argparse
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

DATA_URL = "https://huggingface.co/datasets/OpenRaiser/GGBench/resolve/main/GGBench_dataset.json"
RAW_PATH = Path(__file__).parent / "GGBench_dataset.json"
GEO_BLOCK = re.compile(r"```geogebra\s*\n(.*?)```", re.DOTALL)
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def download_raw(force: bool = False) -> Path:
    if RAW_PATH.exists() and not force:
        return RAW_PATH
    print(f"Downloading {DATA_URL} ...")
    req = urllib.request.Request(DATA_URL, headers={"User-Agent": "polygons-reasoning/0.1"})
    with urllib.request.urlopen(req) as resp, RAW_PATH.open("wb") as f:
        f.write(resp.read())
    return RAW_PATH


def last_geogebra_block(text: str) -> str:
    blocks = GEO_BLOCK.findall(text or "")
    return blocks[-1].strip() if blocks else ""


def build_target(text_answer: str) -> str:
    """Wrap GGBench reference answer in a minimal R1 reasoning envelope.

    R1-distill models are trained so that <think> opens the assistant turn.
    GGBench text_answer has no think tags, so we add exactly one pair around it.
    """
    body = text_answer.strip()
    # Normalize accidental think tags in the reference.
    body = body.replace(THINK_OPEN, "").replace(THINK_CLOSE, "")
    return f"{THINK_OPEN}\n{body}\n{THINK_CLOSE}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-target-chars", type=int, default=12000,
                    help="Drop examples whose target exceeds this many chars (~3k tokens).")
    ap.add_argument("--heldout", type=int, default=100, help="Held-out examples count.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = download_raw()
    with path.open("r", encoding="utf-8") as f:
        items = json.load(f)
    print(f"Loaded {len(items)} problems from {path.name}")

    records, dropped_long, dropped_nocode = [], 0, 0
    for it in items:
        question = (it.get("question") or "").strip()
        text_answer = (it.get("text_answer") or "").strip()
        if not question or not text_answer:
            continue
        ref_code = last_geogebra_block(text_answer)
        if not ref_code:
            dropped_nocode += 1
            continue
        target = build_target(text_answer)
        if len(target) > args.max_target_chars:
            dropped_long += 1
            continue
        records.append({
            "id": str(it.get("id")),
            "question": question,
            "target": target,
            "reference_code": ref_code,
        })

    random.Random(args.seed).shuffle(records)
    n_held = min(args.heldout, max(1, len(records) // 10))
    held, train = records[:n_held], records[n_held:]

    out_dir = Path(__file__).parent
    for name, subset in (("train.jsonl", train), ("heldout.jsonl", held)):
        with (out_dir / name).open("w", encoding="utf-8") as f:
            for r in subset:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"train={len(train)} heldout={len(held)} "
          f"dropped_long={dropped_long} dropped_nocode={dropped_nocode}")
    # Show one decoded example so template handling can be eyeballed.
    print("\n--- sample target (first 600 chars) ---")
    print(train[0]["target"][:600] if train else "(no records)")


if __name__ == "__main__":
    sys.exit(main())
