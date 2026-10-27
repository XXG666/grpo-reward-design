"""Train every reward variant x seed, each followed by a test-set evaluation.

Finished runs (those with results/<variant>_s<seed>.json) are skipped, and a run
stopped mid-training resumes from its latest checkpoint (saved every 50 steps), so
after an interruption just start the script again and it picks up where it left off.

Usage:
    python run_all.py              # run everything that is missing
    python run_all.py --dry-run    # only list what would run
"""
import argparse
import ctypes
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

VARIANTS = ["correct", "correct_format", "partial"]
SEEDS = [0, 1, 2]


def check_environment():
    """Stop early if this isn't the project's Python (training libraries + GPU PyTorch)."""
    missing = [m for m in ("torch", "transformers", "trl", "peft", "datasets") if importlib.util.find_spec(m) is None]
    problem = f"missing packages: {', '.join(missing)}" if missing else None
    if not problem:
        import torch
        if not torch.cuda.is_available():
            problem = f"PyTorch {torch.__version__} cannot see the GPU"
    if problem:
        sys.exit(
            f"Wrong Python environment ({sys.executable}): {problem}.\n"
            "Start the queue with the project's Python instead:\n"
            r"    C:\Users\bowen\.venvs\grpo\Scripts\python.exe run_all.py"
        )


def keep_awake():
    """Ask Windows to keep the system and display on while this script runs.

    Keeping the display on also keeps the laptop out of Modern Standby, where an
    ASUS service forced it to sleep during the first run.
    """
    if sys.platform == "win32":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x1, 0x2
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)


def run(cmd: list[str], log_path: Path) -> bool:
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
    # Append so a resumed run keeps the log from before the interruption.
    with open(log_path, "a", encoding="utf-8") as log:
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode == 0


def log(msg: str):
    print(f"[{time.strftime('%m-%d %H:%M')}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    check_environment()

    # Seed-major order: if the queue is cut short, every variant has the same number of seeds.
    jobs = [(v, s) for s in SEEDS for v in VARIANTS]
    todo = [(v, s) for v, s in jobs if not Path(f"results/{v}_s{s}.json").exists()]
    log(f"{len(jobs) - len(todo)} of {len(jobs)} runs already done; {len(todo)} to go (~{len(todo) * 3.5:.0f} h)")
    if args.dry_run:
        for v, s in todo:
            print(f"  would run: {v}_s{s}")
        return

    keep_awake()
    Path("runs").mkdir(exist_ok=True)
    failed = []
    for v, s in todo:
        name = f"{v}_s{s}"
        if Path(f"runs/{name}/adapter_model.safetensors").exists():
            log(f"{name}: already trained, skipping to evaluation")
        else:
            log(f"{name}: training (log: runs/{name}.log)")
            if not run([sys.executable, "train_grpo.py", "--variant", v, "--seed", str(s)], Path(f"runs/{name}.log")):
                log(f"{name}: TRAINING FAILED, see runs/{name}.log")
                failed.append(name)
                continue
        log(f"{name}: evaluating")
        if not run([sys.executable, "evaluate.py", "--adapter", f"runs/{name}", "--out", f"results/{name}.json"],
                   Path(f"runs/{name}_eval.log")):
            log(f"{name}: EVALUATION FAILED, see runs/{name}_eval.log")
            failed.append(name)
            continue
        log(f"{name}: done")

    log("all runs finished" + (f"; failed: {', '.join(failed)}" if failed else ""))


if __name__ == "__main__":
    main()
