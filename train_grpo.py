"""GRPO fine-tuning of Qwen2.5-0.5B-Instruct with LoRA on GSM8K.

Usage:
    python train_grpo.py --variant correct --seed 0                  # full run (~300 steps)
    python train_grpo.py --variant partial --max-steps 10 --out runs/smoke   # timing test

Variants are defined in rewards.REWARD_VARIANTS. Correctness and format are always
logged (with weight 0 when they are not part of the reward), so every run reports
true accuracy and format rate during training.

Full checkpoints (weights + optimizer + RNG state) are saved every --save-steps steps.
If the output folder already has one, training resumes from the latest complete one.
"""
import argparse
import json
import re
from pathlib import Path

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from data import load_gsm8k
from rewards import REWARD_VARIANTS, correctness_reward, format_reward

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CHECKPOINT_FILES = ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json")


def latest_checkpoint(out_dir: str) -> str | None:
    """Newest checkpoint with every file needed to resume (skips one cut off mid-save)."""
    ckpts = [p for p in Path(out_dir).glob("checkpoint-*") if re.fullmatch(r"checkpoint-\d+", p.name)]
    for ckpt in sorted(ckpts, key=lambda p: int(p.name.split("-")[1]), reverse=True):
        if all((ckpt / f).exists() for f in CHECKPOINT_FILES):
            return str(ckpt)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=REWARD_VARIANTS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num-generations", type=int, default=8, help="completions per prompt (GRPO group size)")
    parser.add_argument("--prompts-per-step", type=int, default=8)
    parser.add_argument("--max-completion-length", type=int, default=384)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--out", default=None, help="default: runs/<variant>_s<seed>")
    args = parser.parse_args()
    out_dir = args.out or f"runs/{args.variant}_s{args.seed}"

    # The reward is the variant's functions; the monitors ride along with weight 0.
    reward_funcs = list(REWARD_VARIANTS[args.variant])
    weights = [1.0] * len(reward_funcs)
    for monitor in (correctness_reward, format_reward):
        if monitor not in reward_funcs:
            reward_funcs.append(monitor)
            weights.append(0.0)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    # Each optimizer step generates prompts_per_step * num_generations completions,
    # processed in micro-batches of 8 to fit in 8 GB of VRAM.
    micro_batch = 8
    generations_per_step = args.prompts_per_step * args.num_generations
    config = GRPOConfig(
        output_dir=out_dir,
        seed=args.seed,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=generations_per_step // micro_batch,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        reward_weights=weights,
        bf16=True,
        logging_steps=1,
        save_steps=args.save_steps,
        report_to="none",
    )

    train = load_gsm8k("train").shuffle(seed=args.seed)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tok,
        reward_funcs=reward_funcs,
        args=config,
        train_dataset=train,
        peft_config=lora,
    )
    resume_from = latest_checkpoint(out_dir)
    if resume_from:
        print(f"resuming from {resume_from}", flush=True)
    trainer.train(resume_from_checkpoint=resume_from)

    trainer.save_model(out_dir)
    Path(out_dir, "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    Path(out_dir, "run_args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"saved adapter and logs to {out_dir}")


if __name__ == "__main__":
    main()
