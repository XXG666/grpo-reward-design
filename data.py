"""GSM8K loading and prompt formatting."""
from datasets import load_dataset

SYSTEM_PROMPT = (
    "You are a helpful assistant that solves math word problems. "
    "Think step by step, then give the final numeric answer inside "
    "<answer></answer> tags, for example <answer>42</answer>."
)
# Qwen's standard math prompt, used only to check results under a second evaluation
# setup. With no system message, the chat template adds Qwen's default one.
QWEN_MATH_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def extract_gold(solution: str) -> str:
    """GSM8K solutions end with '#### <answer>'."""
    return solution.split("####")[-1].strip().replace(",", "")


def build_prompt(question: str, style: str = "ours") -> list[dict]:
    if style == "qwen":
        return [{"role": "user", "content": question + QWEN_MATH_SUFFIX}]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def load_gsm8k(split: str, limit: int | None = None, prompt_style: str = "ours"):
    """Returns a dataset with a chat-format `prompt` and the gold `answer` string."""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds.map(
        lambda ex: {"prompt": build_prompt(ex["question"], prompt_style), "answer": extract_gold(ex["answer"])},
        remove_columns=ds.column_names,
    )
