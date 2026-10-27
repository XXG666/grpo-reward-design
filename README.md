# Reward Design for RL Fine-Tuning of Small Language Models

How does the choice of reward change what a small language model learns during
reinforcement learning? This project fine-tunes **Qwen2.5-0.5B-Instruct** with
**GRPO** (with LoRA) on **GSM8K** math word problems, on a single laptop GPU
(RTX 4060, 8 GB), and compares three reward designs:

| Variant | Reward |
|---|---|
| `correct` | 1 if the final answer is correct, else 0 |
| `correct_format` | correctness + 0.2 bonus for giving the answer in `<answer></answer>` tags |
| `partial` | 1 if correct, up to 0.5 for answers within 50% of the true value |

Measured: test accuracy, format compliance, response length, and examples of
reward hacking.

**Status (Oct 2026): in progress.** The pipeline is complete and the first run
(`correct`, seed 0) is evaluated below. The full comparison (3 reward variants ×
3 seeds) is currently training.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

## Usage

```bash
# Baseline accuracy of the untrained model on the GSM8K test set
python evaluate.py --out results/baseline.json

# GRPO training (300 steps, ~3.2 h on an RTX 4060 Laptop GPU)
python train_grpo.py --variant correct --seed 0

# Evaluate the trained adapter
python evaluate.py --adapter runs/correct_s0 --out results/correct_s0.json

# Full experiment: 3 reward variants x 3 seeds, each trained then evaluated (~3.5 h per run).
# Finished runs are skipped and an interrupted run resumes from its latest checkpoint
# (saved every 50 steps), so it is safe to stop with Ctrl+C and re-run later.
python run_all.py
```

Training settings: LoRA r=16 on all attention and MLP projections, learning rate 1e-5,
8 prompts × 8 completions per step, completions capped at 384 tokens, no KL penalty.

## Results

Zero-shot, greedy decoding, full GSM8K test set (1319 problems), max 512 new tokens.

| Model | GSM8K accuracy | Format rate | Mean tokens |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct (baseline) | 43.3% | 7.4% | 291 |
| + GRPO, `correct`, seed 0 | 49.6% | 0.3% | 208 |

## License

MIT
