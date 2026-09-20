# polygons-reasoning

LoRA-train a small reasoning model (DeepSeek-R1-Distill-Qwen-7B) on GGBench
(GeoGebra construction problems), then test out-of-distribution transfer on
your own polygon problems.

Pipeline: `data/prepare_data.py` → `train/train_lora.py` → `eval/run_inference.py` → `eval/score.py`

## Local CPU (no GPU needed)

```bash
pip install huggingface_hub

python3 data/prepare_data.py        # downloads GGBench → data/train.jsonl (1308) + data/heldout.jsonl (100)
python3 tests/test_score.py         # scorer unit tests (7 tests)
python3 train/train_lora.py --help  # dry check, no GPU touched
python3 eval/run_inference.py --help
python3 eval/score.py --help
```

## GPU provider (≥24GB VRAM, e.g. A10 / 3090 / A100)

Upload this folder (without `external/`, `data/GGBench_dataset.json`, `out/`),
then:

```bash
pip install unsloth trl transformers datasets accelerate peft torch openai

# 1. smoke train (~minutes)
python train/train_lora.py --max-samples 20

# 2. full train (1 epoch, ~1308 examples) → out/adapter
python train/train_lora.py

# 3. baseline: untuned base model on heldout
python eval/run_inference.py --input data/heldout.jsonl \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --output eval/outputs/heldout_base.jsonl

# 4. tuned model on heldout
python eval/run_inference.py --input data/heldout.jsonl --adapter out/adapter \
    --output eval/outputs/heldout_tuned.jsonl

# 5. tuned model on your problems (edit my_problems.json first!)
python eval/run_inference.py --input my_problems.json --adapter out/adapter \
    --output eval/outputs/my_problems.jsonl

# 6. score everything
python eval/score.py --input eval/outputs/heldout_base.jsonl  --output eval/outputs/heldout_base_report.md
python eval/score.py --input eval/outputs/heldout_tuned.jsonl --output eval/outputs/heldout_tuned_report.md
python eval/score.py --input eval/outputs/my_problems.jsonl   --output eval/outputs/my_problems_report.md
```

Optional VLM-T judge (GGBench paper's text rubric, 1–5, via OpenRouter):

```bash
export OPENROUTER_API_KEY=sk-or-...
python eval/score.py --input eval/outputs/my_problems.jsonl --vlm-judge
```

## Reading the scores

| Metric | Meaning |
|---|---|
| `has_code` / `valid` | emitted a ```geogebra block / every line parses as a known command |
| `seq_match` | exact normalized command sequence vs reference (strict) |
| `obj_sim` | Dice overlap of (object, args) pairs vs reference |
| `cmd_sim` | Dice overlap of command names vs reference |
| `mean_vlm_t` | optional LLM-judge rubric score 1–5 |

- **SFT worked** if tuned `valid`/`seq_match`/`obj_sim` ≫ base on heldout.
- **OOD transfer** = `my_problems` scores minus heldout scores. A drop measures
  how far construction reasoning generalizes outside GGBench's distribution.

`my_problems.json` is a template with 3 example problems (rotation, translation,
reflection) — replace with your own. `reference_code` is optional; without it
only `has_code`/`valid` are reported.

## Viewing generated constructions

Serve the repo and open the browser viewer (GeoGebra applet runs the code live):

```bash
python3 -m http.server 8000
# then open http://127.0.0.1:8000/viewer.html?file=my_problems_report.md
# or:        http://127.0.0.1:8000/viewer.html?file=eval/outputs/my_problems.jsonl
```

Rejected lines (syntax GeoGebra refuses) are listed under the code box.

## Layout

```
data/prepare_data.py     dataset → train/heldout jsonl (CPU)
train/train_lora.py      Unsloth 4-bit QLoRA SFT (GPU only)
eval/run_inference.py    greedy generation, base or adapter (GPU only)
eval/score.py            deterministic scorer + optional VLM judge (CPU)
tests/test_score.py      scorer unit tests (CPU)
my_problems.json         your OOD test problems
external/GGBench/        upstream eval repo (VLM-T prompt template imported from here)
```

If 7B OOMs at seq len 3072 on the provider: retry `--seq-len 2048`; if still
OOM, switch base: `--model Qwen/Qwen3-4B-Thinking-2507` (same flags).
