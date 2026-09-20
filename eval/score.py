#!/usr/bin/env python3
"""Score model generations on GeoGebra construction problems.

Deterministic scoring (always):
  - has_code:        a ```geogebra block was extracted
  - valid:           every line parses as a GeoGebra command/assignment
  - structural match vs reference_code (when present):
      seq_match      exact normalized command sequence
      obj_sim        Dice similarity of (object, args) pairs [0,1]
      cmd_sim        Dice similarity of command names [0,1]

Optional VLM-T judge (GGBench paper's text rubric, 1-5):
  --vlm-judge with an OpenAI-compatible endpoint (e.g. OpenRouter):
      env OPENROUTER_API_KEY, or --api-key/--base-url/--judge-model.
  Prompt template imported verbatim from external/GGBench/eval_prompts.py.

Usage:
  python3 eval/score.py --input eval/outputs/heldout.jsonl
  python3 eval/score.py --input eval/outputs/heldout.jsonl --vlm-judge
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GGBENCH_PROMPTS = REPO / "external" / "GGBench" / "eval_prompts.py"

GEO_BLOCK = re.compile(r"```(?:geogebra|ggb)?\s*\n(.*?)```", re.DOTALL)
# GeoGebra command style: Name(args) possibly assigned: x = Command(...)
CMD_LINE = re.compile(
    r"^\s*(?:([A-Za-z][A-Za-z0-9_]*)\s*=\s*)?"     # optional assignment (lhs)
    r"([A-Za-z][A-Za-z0-9_]*)\s*\((.*)\)\s*$"      # Command(args)
)

# Commands taking no constructor args (canvas/state settings are fine too).
# Settings commands are style/state, not part of the geometric construction:
# they are excluded from object-level similarity (still checked for validity).
SET_COMMANDS = {
    "SetCaption", "SetColor", "SetPointStyle", "SetPointSize", "SetLineThickness",
    "SetLineStyle", "ShowLabel", "ShowAxes", "ShowGrid", "SetVisibleInView",
    "SetFilling", "SetLabelStyle", "SetConditionToShowObject", "SetText", "SetSeed",
    "ZoomIn", "ZoomOut", "Pan", "SetViewDirection", "CenterView",
}
KNOWN_COMMANDS = {
    "Point", "Segment", "Line", "Ray", "Circle", "Arc", "Intersect",
    "Midpoint", "PerpendicularLine", "PerpendicularBisector", "AngleBisector",
    "ParallelLine", "Rotate", "Translate", "Reflect", "Dilate", "Vector",
    "Polygon", "RegularPolygon", "Angle", "Distance", "Length", "Slope",
    "Tangent", "Ellipse", "Hyperbola", "Parabola", "PointIn", "Midpoint",
    "SetCaption", "SetColor", "SetPointStyle", "SetPointSize", "SetLineThickness",
    "SetLineStyle", "ShowLabel", "ShowAxes", "ShowGrid", "SetVisibleInView",
    "SetFilling", "SetLabelStyle", "SetConditionToShowObject", "SetText", "SetSeed",
    "FillCells", "ZoomIn", "ZoomOut", "Pan", "SetViewDirection", "CenterView",
    "If", "Sequence", "Zip", "Iteration", "IterationList", "First", "Last",
    "Element", "Length_", "Sum", "Max", "Min", "Abs", "Round", "Floor", "Ceil",
    "Sqrt", "Sin", "Cos", "Tan", "Asin", "Acos", "Atan", "Mod", "GCD", "LCM",
    "Text", "UnicodeToLetter", "LetterToUnicode",
}
# GeoGebra's lowercase builtin functions (math fns + coordinate extraction,
# e.g. "h = sqrt(L^2 - r^2)", "s = y(I)"). Used in ~0.3% of GGBench code.
LOWERCASE_FUNCTIONS = {
    "x", "y", "z", "sqrt", "cbrt", "abs", "exp", "log", "ln", "sin", "cos",
    "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "floor", "ceil",
    "round", "min", "max", "gcd", "lcm", "sign", "nroot",
}


def extract_code(text: str) -> str:
    blocks = GEO_BLOCK.findall(text or "")
    return blocks[-1].strip() if blocks else ""


def normalize_coord(token: str) -> str:
    """Normalize a numeric literal so 2.0 == 2 == 2.00."""
    try:
        v = float(token)
        if v == int(v):
            return str(int(v))
        return f"{v:.6g}"
    except ValueError:
        return token


def normalize_args(args: str) -> str:
    """Normalize numeric literals and whitespace inside an argument string."""
    args = re.sub(r"-?\d+\.?\d*(?:e-?\d+)?", lambda m: normalize_coord(m.group()), args)
    return re.sub(r"\s*,\s*", ", ", args.strip())


def strip_comment(line: str) -> str:
    """Strip a trailing '#' comment, but not inside double-quoted strings.

    GGBench code uses SetColor(obj, "#RRGGBB") — hex colors contain '#', and
    the benchmark never uses '#' comments.
    """
    in_str = False
    for i, ch in enumerate(line):
        if ch == '"':
            in_str = not in_str
        elif ch == "#" and not in_str:
            return line[:i]
    return line


EXPR_LINE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.+)$")


def parse_code(code: str):
    """Return (commands, invalid_lines).

    commands: list of (cmd_name, obj_name, normalized_args) per parseable line.
    obj_name is the assigned variable when present, else the command name.

    Besides Command(...) lines this accepts plain assignments like
    "scaleFactor = 0.5" or "E = A + 3 * UnitVector(bisector)" — GGBench
    reference code uses these (~10% of blocks); cmd_name is "<expr>" for them.
    """
    commands, invalid = [], []
    for raw in code.splitlines():
        line = strip_comment(raw).strip()
        if not line:
            continue
        m = CMD_LINE.match(line)
        if m:
            lhs, name, args = m.group(1), m.group(2), normalize_args(m.group(3))
            commands.append((name, lhs or name, args))
            continue
        m = EXPR_LINE.match(line)
        if m and all(tok not in m.group(2) for tok in "=≠<>≤≥∈||") \
                and not re.search(r"!\s*[A-Za-z(]", m.group(2)):
            # simple assignment of an arithmetic/geometric expression
            commands.append(("<expr>", m.group(1), normalize_args(m.group(2))))
            continue
        invalid.append(raw.strip())
    return commands, invalid


def code_stats(code: str):
    commands, invalid = parse_code(code)
    names = [c[0] for c in commands]
    heuristic_ok = (
        bool(commands)
        and not invalid
        and all(n == "<expr>" or n in KNOWN_COMMANDS or n in LOWERCASE_FUNCTIONS
                or n[0].isupper() for n in names)
    )
    return {
        "n_commands": len(commands),
        "n_invalid_lines": len(invalid),
        "heuristic_valid": heuristic_ok,
    }


def structural_metrics(gen_code: str, ref_code: str):
    gen, _ = parse_code(gen_code)
    ref, _ = parse_code(ref_code)
    # Settings commands are style, not geometry: drop from object comparison
    # (they can legitimately differ between two correct constructions).
    gen_objs = [c for c in gen if c[1] not in SET_COMMANDS]
    ref_objs = [c for c in ref if c[1] not in SET_COMMANDS]
    seq_match = gen_objs == ref_objs
    # Dice similarity on (object name, args) pairs, then on command names only.
    full_gen, full_ref = Counter((o, a) for _, o, a in gen_objs), Counter((o, a) for _, o, a in ref_objs)
    denom = len(gen_objs) + len(ref_objs) or 1
    obj_sim = 2 * sum((full_gen & full_ref).values()) / denom
    gen_names = Counter(n for n, _, _ in gen_objs)
    ref_names = Counter(n for n, _, _ in ref_objs)
    name_denom = len(gen_objs) + len(ref_objs) or 1
    cmd_sim = 2 * sum((gen_names & ref_names).values()) / name_denom
    return {"seq_match": seq_match, "obj_sim": round(obj_sim, 3), "cmd_sim": round(cmd_sim, 3)}


# ---------------------------------------------------------------- VLM-T judge

def judge_client(args):
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("No API key: set OPENROUTER_API_KEY or pass --api-key.")
    from openai import OpenAI
    base_url = args.base_url or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    return OpenAI(base_url=base_url, api_key=api_key), args.judge_model


def load_text_prompt_template() -> str:
    """Import TEXT_STEP_PROMPT_TEMPLATE verbatim from the GGBench repo."""
    if not GGBENCH_PROMPTS.exists():
        raise SystemExit(f"GGBench prompts not found at {GGBENCH_PROMPTS}. "
                         "Run: git clone https://github.com/OpenRaiser/GGBench external/GGBench")
    import importlib.util
    spec = importlib.util.spec_from_file_location("ggbench_eval_prompts", GGBENCH_PROMPTS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TEXT_STEP_PROMPT_TEMPLATE


def vlm_text_scores(client, model, records, template, max_workers=4):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def worker(rec):
        prompt = template.format(
            problem=rec["question"],
            reference_answer=rec.get("reference_answer") or rec.get("reference_code") or "(no reference)",
            model_answer=rec["output"],
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return rec["id"], resp.choices[0].message.content.strip()
        except Exception as exc:  # noqa: BLE001
            return rec["id"], f"ERROR: {exc}"

    scores = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, r) for r in records]
        for fut in as_completed(futures):
            rid, val = fut.result()
            scores[rid] = val
    return scores


# ---------------------------------------------------------------------- report

def aggregate(results):
    n = len(results) or 1
    with_ref = [r for r in results if r.get("reference_code")]
    return {
        "total": len(results),
        "has_code": sum(r["deterministic"]["has_code"] for r in results),
        "valid": sum(r["deterministic"]["stats"]["heuristic_valid"] for r in results),
        "with_reference": len(with_ref),
        "seq_match": sum(r["structural"]["seq_match"] for r in with_ref) if with_ref else None,
        "mean_obj_sim": round(sum(r["structural"]["obj_sim"] for r in with_ref) / len(with_ref), 3) if with_ref else None,
        "mean_cmd_sim": round(sum(r["structural"]["cmd_sim"] for r in with_ref) / len(with_ref), 3) if with_ref else None,
        "mean_vlm_t": (round(sum(float(r["vlm_t"]) for r in results if r.get("vlm_t", "").replace(".", "").isdigit()) /
                             max(1, sum(1 for r in results if r.get("vlm_t", "").replace(".", "").isdigit())), 2)
                       ) if any(r.get("vlm_t", "").replace(".", "").isdigit() for r in results) else None,
    }


def write_report(out_path: Path, results, agg, vlm_used: bool):
    lines = ["# Evaluation report", ""]
    lines += ["## Headline", "", f"```json", json.dumps(agg, indent=2), "```", ""]
    lines += ["## Per-example", ""]
    for r in results:
        d, s = r["deterministic"], r.get("structural") or {}
        lines.append(f"### id={r['id']}  has_code={d['has_code']} valid={d['stats']['heuristic_valid']}")
        if s:
            lines.append(f"- seq_match={s['seq_match']} obj_sim={s['obj_sim']} cmd_sim={s['cmd_sim']}")
        if r.get("vlm_t"):
            lines.append(f"- VLM-T score: {r['vlm_t']}")
        lines.append("")
        lines.append("<details><summary>question</summary>\n")
        lines.append(f"```\n{r['question']}\n```")
        lines.append("</details>")
        lines.append("")
        lines.append("<details><summary>generated code</summary>\n")
        lines.append(f"```geogebra\n{r.get('generated_code') or '(none)'}\n```")
        lines.append("</details>")
        if r.get("reference_code"):
            lines.append("<details><summary>reference code</summary>\n")
            lines.append(f"```geogebra\n{r['reference_code']}\n```")
            lines.append("</details>")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="inference output JSONL")
    ap.add_argument("--output", default=None, help="report markdown (default: <input>_report.md)")
    ap.add_argument("--vlm-judge", action="store_true",
                    help="additionally run the GGBench VLM-T text judge (needs API key)")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--judge-model", default=None,
                    help="judge model id (default: env GGBENCH_JUDGE_MODEL or gpt-4o)")
    ap.add_argument("--max-workers", type=int, default=4)
    args = ap.parse_args()

    in_path = Path(args.input)
    records = [json.loads(l) for l in in_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    template = load_text_prompt_template() if args.vlm_judge else None
    client, jmodel = (None, None)
    if args.vlm_judge:
        client, jmodel = judge_client(args)
        jmodel = jmodel or os.environ.get("GGBENCH_JUDGE_MODEL", "gpt-4o")

    results = []
    for rec in records:
        gen_code = rec.get("generated_code") or extract_code(rec.get("output", ""))
        stats = code_stats(gen_code) if gen_code else {"n_commands": 0, "n_invalid_lines": 0, "heuristic_valid": False}
        r = {
            "id": rec.get("id"),
            "question": rec.get("question", ""),
            "output": rec.get("output", ""),
            "generated_code": gen_code,
            "deterministic": {"has_code": bool(gen_code), "stats": stats},
        }
        if rec.get("reference_code"):
            r["reference_code"] = rec["reference_code"]
            r["structural"] = structural_metrics(gen_code, rec["reference_code"]) if gen_code else \
                {"seq_match": False, "obj_sim": 0.0, "cmd_sim": 0.0}
        results.append(r)

    if args.vlm_judge:
        scores = vlm_text_scores(client, jmodel, results, template, args.max_workers)
        for r in results:
            r["vlm_t"] = scores.get(r["id"], "")

    agg = aggregate(results)
    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_report.md")
    write_report(out_path, results, agg, args.vlm_judge)
    print(json.dumps(agg, indent=2))
    print(f"Report: {out_path}")


if __name__ == "__main__":
    sys.exit(main())
