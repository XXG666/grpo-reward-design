"""Answer parsing and the three reward designs compared in this project.

Each reward function follows TRL's GRPOTrainer signature: it receives the batch
of completions plus dataset columns (here `answer`) and returns one float per
completion. When several functions are listed for a variant, TRL sums them.
"""
import re

ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def completion_text(completion) -> str:
    # Conversational completions arrive as [{"role": "assistant", "content": ...}].
    if isinstance(completion, list):
        return completion[-1]["content"]
    return completion


def has_answer_tag(text: str) -> bool:
    return ANSWER_TAG.search(text) is not None


def last_number(segment: str) -> float | None:
    # Drop dollar signs (plain or LaTeX-escaped) so "-\$10" parses as -10.
    numbers = NUMBER.findall(segment.replace("\\$", "").replace("$", ""))
    if not numbers:
        return None
    try:
        return float(numbers[-1].replace(",", ""))
    except ValueError:
        return None


def extract_answer(text: str) -> float | None:
    """Last number inside the last <answer> tag, or the last number in the text if there is no tag."""
    tags = ANSWER_TAG.findall(text)
    return last_number(tags[-1] if tags else text)


def extract_boxed_answer(text: str) -> float | None:
    """Last number inside the last \\boxed{}, or the last number in the text if there is none."""
    boxes = BOXED.findall(text)
    return last_number(boxes[-1] if boxes else text)


def is_correct(text: str, gold: str) -> bool:
    pred = extract_answer(text)
    return pred is not None and abs(pred - float(gold)) < 1e-6


def correctness_reward(completions, answer, **kwargs) -> list[float]:
    """1 if the final answer is right, else 0."""
    return [1.0 if is_correct(completion_text(c), a) else 0.0 for c, a in zip(completions, answer)]


def format_reward(completions, **kwargs) -> list[float]:
    """Small bonus for putting the final answer in <answer> tags."""
    return [0.2 if has_answer_tag(completion_text(c)) else 0.0 for c in completions]


def partial_credit_reward(completions, answer, **kwargs) -> list[float]:
    """1 for an exact answer; up to 0.5 for answers within 50% of the gold value."""
    rewards = []
    for c, a in zip(completions, answer):
        pred, gold = extract_answer(completion_text(c)), float(a)
        if pred is None:
            rewards.append(0.0)
        elif abs(pred - gold) < 1e-6:
            rewards.append(1.0)
        else:
            rel_err = abs(pred - gold) / max(abs(gold), 1.0)
            rewards.append(max(0.0, 0.5 * (1 - 2 * rel_err)))
    return rewards


REWARD_VARIANTS = {
    "correct": [correctness_reward],
    "correct_format": [correctness_reward, format_reward],
    "partial": [partial_credit_reward],
}
