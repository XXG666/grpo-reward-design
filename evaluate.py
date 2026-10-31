"""Evaluate a model (optionally with a trained LoRA adapter) on the GSM8K test set.

Usage:
    python evaluate.py --limit 100 --out results/smoke.json      # quick check
    python evaluate.py --out results/baseline.json               # full test set (1319 problems)
    python evaluate.py --adapter runs/correct_s0 --out results/correct_s0.json
    python evaluate.py --prompt qwen --out results/qwen_prompt/baseline.json   # Qwen's standard prompt
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import load_gsm8k
from rewards import extract_answer, extract_boxed_answer, has_answer_tag

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--adapter", default=None, help="path to a LoRA adapter saved by training")
    parser.add_argument("--limit", type=int, default=None, help="only evaluate the first N problems")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--prompt", choices=["ours", "qwen"], default="ours",
                        help="ours: training prompt, <answer> tags; qwen: Qwen's standard math prompt, \\boxed{}")
    parser.add_argument("--out", default="results/baseline.json")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.eval()

    ds = load_gsm8k("test", args.limit, args.prompt)
    extract = extract_boxed_answer if args.prompt == "qwen" else extract_answer
    records = []
    start = time.time()
    for i in range(0, len(ds), args.batch_size):
        batch = ds[i : i + args.batch_size]
        texts = [tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in batch["prompt"]]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 temperature=None, top_p=None, top_k=None)
        generated = out[:, enc["input_ids"].shape[1]:]

        for prompt, gold, ids in zip(batch["prompt"], batch["answer"], generated):
            text = tok.decode(ids, skip_special_tokens=True)
            n_tokens = int((ids != tok.pad_token_id).sum())
            pred = extract(text)
            records.append({
                "question": prompt[-1]["content"],
                "gold": gold,
                "pred": pred,
                "correct": pred is not None and abs(pred - float(gold)) < 1e-6,
                "has_tag": has_answer_tag(text),
                "tokens": n_tokens,
                "truncated": n_tokens >= args.max_new_tokens,
                "completion": text,
            })
        acc = sum(r["correct"] for r in records) / len(records)
        print(f"{len(records)}/{len(ds)}  accuracy so far {acc:.3f}  ({time.time() - start:.0f}s)", flush=True)

    n = len(records)
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "prompt": args.prompt,
        "n": n,
        "accuracy": sum(r["correct"] for r in records) / n,
        "format_rate": sum(r["has_tag"] for r in records) / n,
        "truncation_rate": sum(r["truncated"] for r in records) / n,
        "mean_tokens": sum(r["tokens"] for r in records) / n,
        "seconds": round(time.time() - start),
    }
    print(json.dumps(summary, indent=2))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "samples": records}, indent=2))
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
