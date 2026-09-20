#!/usr/bin/env python3
"""Unsloth 4-bit QLoRA SFT on GGBench.

Runs ONLY on a CUDA GPU (rented provider, >=16GB recommended for 7B).
Local CPU: `python3 train/train_lora.py --help` works for a dry check.
"""
import argparse
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
OUT_DIR = REPO / "out"

DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# DeepSeek-R1-Distill chat template markers
INSTRUCTION_PART = "<\uff5cUser\uff5c>"
RESPONSE_PART = "<\uff5cAssistant\uff5c>"


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_dataset(records, tokenizer, max_seq_len):
    """Apply the model's chat template: user=question, assistant=target(+eos)."""
    texts = []
    for r in records:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": r["question"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        target = r["target"].rstrip() + tokenizer.eos_token
        texts.append(prompt + target)
    dataset = [{"text": t} for t in texts]
    n_over = sum(1 for d in dataset if len(d["text"]) > max_seq_len * 3)
    if n_over:
        print(f"note: {n_over} examples exceed ~{max_seq_len} tokens (will be truncated)")
    return dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF base model id")
    ap.add_argument("--train-file", default=str(DATA_DIR / "train.jsonl"))
    ap.add_argument("--output", default=str(OUT_DIR / "adapter"))
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all (smoke test: 20)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--seq-len", type=int, default=3072)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch  # noqa: F401  (GPU only)
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    random.seed(args.seed)
    records = load_jsonl(Path(args.train_file))
    if args.max_samples:
        records = records[: args.max_samples]
    print(f"{len(records)} training examples")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.seq_len,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    dataset = Dataset.from_list(build_dataset(records, tokenizer, args.seq_len))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.seq_len,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=10,
            logging_steps=5,
            save_strategy="no",
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            optim="paged_adamw_8bit",
            output_dir=str(OUT_DIR / "trainer_tmp"),
            report_to="none",
            seed=args.seed,
        ),
        packing=False,
    )
    # Loss on assistant tokens only.
    trainer = train_on_responses_only(
        trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART
    )

    gpu_ok = torch.cuda.is_available()
    if not gpu_ok:
        raise SystemExit("No CUDA GPU available. Run this on a GPU provider box.")

    trainer.train()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Adapter saved to {args.output}")


if __name__ == "__main__":
    main()
