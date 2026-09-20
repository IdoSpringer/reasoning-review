#!/usr/bin/env python3
"""Greedy inference over a JSONL/JSON problem file, with or without a LoRA adapter.

Runs ONLY on a CUDA GPU. Input records need {"id", "question"} (reference_code optional).
Output: JSONL with {"id", "question", "output", "generated_code", "reference_code"?}.
Local CPU: `python3 eval/run_inference.py --help` works for a dry check.
"""
import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
GEO_BLOCK = re.compile(r"```geogebra\s*\n(.*?)```", re.DOTALL)


def load_records(path: Path):
    text = path.read_text(encoding="utf-8")
    data = json.loads(text) if path.suffix == ".json" else [
        json.loads(line) for line in text.splitlines() if line.strip()]
    out = []
    for r in data:
        if "question" not in r:
            raise SystemExit(f"{path}: record missing 'question': {r}")
        out.append({"id": str(r.get("id", len(out))),
                    "question": r["question"],
                    "reference_code": r.get("reference_code")})
    return out


def extract_code(text: str) -> str:
    blocks = GEO_BLOCK.findall(text or "")
    return blocks[-1].strip() if blocks else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="JSONL/JSON file with 'question' fields")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF base model id")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (omit = base model)")
    ap.add_argument("--output", default=None, help="output JSONL (default eval/outputs/<input>.jsonl)")
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--seq-len", type=int, default=3072)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    records = load_records(Path(args.input))
    if args.limit:
        records = records[: args.limit]

    out_path = Path(args.output) if args.output else (
        Path(__file__).parent / "outputs" / (Path(args.input).stem + ".jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import torch
    from unsloth import FastLanguageModel

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU available. Run this on a GPU provider box.")

    load_from = args.adapter or args.model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=load_from,
        max_seq_length=args.seq_len,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    results = []
    for i, r in enumerate(records):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": r["question"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        gen_kwargs = dict(max_new_tokens=args.max_new, pad_token_id=tokenizer.eos_token_id)
        if args.temperature and args.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=0.95)
        with torch.no_grad():
            out_ids = model.generate(**inputs, **gen_kwargs)
        # Strip the prompt tokens.
        new_ids = out_ids[:, inputs["input_ids"].shape[1]:]
        output = tokenizer.decode(new_ids[0], skip_special_tokens=True)
        rec = {"id": r["id"], "question": r["question"], "output": output,
               "generated_code": extract_code(output)}
        if r.get("reference_code"):
            rec["reference_code"] = r["reference_code"]
        results.append(rec)
        print(f"[{i + 1}/{len(records)}] id={r['id']} "
              f"code={'yes' if rec['generated_code'] else 'NO'} len={len(output)}")
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(results)} generations to {out_path}")


if __name__ == "__main__":
    main()
