"""Compare the reward variants across seeds: test-set table, reward-hacking checks and figures.

Usage:
    python analyze.py

Reads results/<variant>_s<seed>.json (from evaluate.py) and the training logs. Each run's
log_history.json is copied from runs/ (not tracked by git) into results/train_logs/ in
compact form, so the figures can be rebuilt from the repository alone.
"""
import json
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARIANTS = ["correct", "correct_format", "partial"]
SEEDS = [0, 1, 2]
COLORS = {"correct": "#2a78d6", "correct_format": "#eb6834", "partial": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BOXED = re.compile(r"\\boxed\{")
LOG_KEYS = {
    "correct": "rewards/correctness_reward/mean",
    "format": "rewards/format_reward/mean",
    "partial": "rewards/partial_credit_reward/mean",
    "length": "completions/mean_length",
    "clipped": "completions/clipped_ratio",
    "no_signal": "frac_reward_zero_std",
}
SMOOTH = 25


def export_train_log(name: str) -> list[dict]:
    """Compact copy of runs/<name>/log_history.json kept under results/train_logs/."""
    out = Path(f"results/train_logs/{name}.json")
    src = Path(f"runs/{name}/log_history.json")
    if src.exists():
        rows = [
            {"step": e["step"], **{k: float(e[v]) for k, v in LOG_KEYS.items() if v in e}}
            for e in json.loads(src.read_text()) if "step_time" in e
        ]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows))
    return json.loads(out.read_text())


def near_miss_rate(samples: list[dict]) -> float:
    """Share of answers that are wrong but within 50% of the gold value (earn partial credit)."""
    n = 0
    for s in samples:
        pred, gold = s["pred"], float(s["gold"])
        if pred is not None and not s["correct"] and abs(pred - gold) / max(abs(gold), 1.0) < 0.5:
            n += 1
    return n / len(samples)


def test_metrics(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    m, samples = data["summary"], data["samples"]
    return {
        "accuracy": m["accuracy"],
        "tag_rate": m["format_rate"],
        "boxed_rate": sum(bool(BOXED.search(s["completion"])) for s in samples) / len(samples),
        "near_miss": near_miss_rate(samples),
        "truncation": m["truncation_rate"],
        "tokens": m["mean_tokens"],
    }


def smooth(xs: list[float]) -> list[float]:
    """Trailing moving average, starting once the window is full."""
    return [statistics.mean(xs[i - SMOOTH + 1: i + 1]) for i in range(SMOOTH - 1, len(xs))]


def mean_sd(xs: list[float]) -> tuple[float, float]:
    return statistics.mean(xs), statistics.stdev(xs)


def style(ax, title: str, ylabel: str):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def plot_test_accuracy(runs: dict, baseline: dict, path: Path):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for i, v in enumerate(VARIANTS):
        accs = [runs[v][s]["accuracy"] * 100 for s in SEEDS]
        ax.scatter([i] * len(accs), accs, s=36, color=COLORS[v], alpha=0.55, zorder=3,
                   edgecolors="white", linewidths=1)
        mean = statistics.mean(accs)
        ax.hlines(mean, i - 0.22, i + 0.22, color=COLORS[v], linewidth=2.5, zorder=4)
        ax.text(i + 0.27, mean, f"{mean:.1f}%", va="center", fontsize=9, color=INK)
    ax.axhline(baseline["accuracy"] * 100, color=MUTED, linestyle="--", linewidth=1.2)
    ax.text(len(VARIANTS) - 0.5, baseline["accuracy"] * 100 + 0.4,
            f"untrained baseline {baseline['accuracy']:.1%}", ha="right", fontsize=9, color=MUTED)
    ax.set_xticks(range(len(VARIANTS)), VARIANTS)
    ax.set_xlim(-0.5, len(VARIANTS) - 0.4)
    ax.set_ylim(40, 54)
    style(ax, "GSM8K test accuracy (dots: seeds, bar: mean of 3)", "accuracy (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)


def plot_training(logs: dict, path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    (acc, tag), (length, near) = axes
    for v in VARIANTS:
        for s in SEEDS:
            rows = logs[v][s]
            steps = [r["step"] for r in rows][SMOOTH - 1:]
            label = v if s == 0 else None
            acc.plot(steps, smooth([r["correct"] * 100 for r in rows]), color=COLORS[v], lw=1.5, label=label)
            tag.plot(steps, smooth([r["format"] / 0.2 * 100 for r in rows]), color=COLORS[v], lw=1.5)
            length.plot(steps, smooth([r["length"] for r in rows]), color=COLORS[v], lw=1.5)
            if v == "partial":
                near.plot(steps, smooth([r["partial"] - r["correct"] for r in rows]), color=COLORS[v], lw=1.5)
    style(acc, "Training accuracy (sampled answers)", "correct (%)")
    style(tag, "Answers in <answer> tags", "share (%)")
    style(length, "Response length", "mean tokens")
    style(near, "Partial-credit reward earned by wrong answers", "reward per answer")
    near.set_ylim(0, 0.1)
    for ax in (length, near):
        ax.set_xlabel("training step", color=MUTED)
    fig.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.005, f"Each line is one seed; {SMOOTH}-step moving average.", ha="center",
             fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    baseline = test_metrics(Path("results/baseline.json"))
    runs = {v: {s: test_metrics(Path(f"results/{v}_s{s}.json")) for s in SEEDS} for v in VARIANTS}
    logs = {v: {s: export_train_log(f"{v}_s{s}") for s in SEEDS} for v in VARIANTS}

    cols = ["accuracy", "tag_rate", "boxed_rate", "near_miss", "truncation", "tokens"]
    print("| Model | Accuracy | `<answer>` tags | `\\boxed{}` | Near misses | Truncated | Mean tokens |")
    print("|---|---|---|---|---|---|---|")
    fmt = lambda c, x: f"{x:.0f}" if c == "tokens" else f"{x:.1%}"
    print("| Baseline | " + " | ".join(fmt(c, baseline[c]) for c in cols) + " |")
    summary = {"baseline": baseline}
    for v in VARIANTS:
        cells, summary[v] = [], {}
        for c in cols:
            mean, sd = mean_sd([runs[v][s][c] for s in SEEDS])
            summary[v][c] = {"mean": mean, "sd": sd, "seeds": [runs[v][s][c] for s in SEEDS]}
            cells.append(f"{mean:.0f} ± {sd:.0f}" if c == "tokens" else f"{mean:.1%} ± {sd:.1%}")
        print(f"| `{v}` | " + " | ".join(cells) + " |")

    print("\nPer-seed accuracy / tag rate:")
    for v in VARIANTS:
        print(f"  {v:15s}", "  ".join(f"s{s}: {runs[v][s]['accuracy']:.1%} / {runs[v][s]['tag_rate']:.1%}" for s in SEEDS))

    print(f"\nPartial-credit reward earned by wrong answers (training, {SMOOTH}-step windows):")
    for s in SEEDS:
        gap = [r["partial"] - r["correct"] for r in logs["partial"][s]]
        print(f"  partial_s{s}: first {statistics.mean(gap[:SMOOTH]):.3f}, last {statistics.mean(gap[-SMOOTH:]):.3f}")

    Path("figures").mkdir(exist_ok=True)
    plot_test_accuracy(runs, baseline, Path("figures/test_accuracy.png"))
    plot_training(logs, Path("figures/training_curves.png"))
    Path("results/summary.json").write_text(json.dumps(summary, indent=2))
    print("\nsaved figures/test_accuracy.png, figures/training_curves.png, results/summary.json")


if __name__ == "__main__":
    main()
