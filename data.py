"""GSM8K loading and prompt formatting."""
from datasets import load_dataset

SYSTEM_PROMPT = (
    "You are a helpful assistant that solves math word problems. "
    "Think step by step, then give the final numeric answer inside "
    "<answer></answer> tags, for example <answer>42</answer>."
)


def extract_gold(solution: str) -> str:
    """GSM8K solutions end with '#### <answer>'."""
    return solution.split("####")[-1].strip().replace(",", "")


def build_prompt(question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def load_gsm8k(split: str, limit: int | None = None):
    """Returns a dataset with a chat-format `prompt` and the gold `answer` string."""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds.map(
        lambda ex: {"prompt": build_prompt(ex["question"]), "answer": extract_gold(ex["answer"])},
        remove_columns=ds.column_names,
    )
