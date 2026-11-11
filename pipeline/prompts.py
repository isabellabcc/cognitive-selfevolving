"""Plain-text GSM8K prompts for causal language-model completion."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from .common import PipelineError, read_jsonl


PROMPT_VERSION = "gsm8k_star_v1"


def load_few_shot(path: str | Path) -> list[dict[str, Any]]:
    examples = list(read_jsonl(path))
    if not examples:
        raise PipelineError("The GSM8K few-shot prompt set is empty")
    for example in examples:
        if not str(example.get("question") or "").strip() or "####" not in str(example.get("answer") or ""):
            raise PipelineError("Invalid GSM8K few-shot example")
    return examples


def build_prompt(
    question: str, examples: Iterable[Mapping[str, Any]], *, gold_hint: str | None = None
) -> str:
    blocks = [f"Q: {row['question']}\nA: {str(row['answer']).strip()}" for row in examples]
    if gold_hint is None:
        target = f"Q: {question}\nA:"
    else:
        target = (
            f"Q: {question}\n"
            f"The correct answer is {gold_hint}. Explain step by step why this answer is correct.\nA:"
        )
    return "\n\n".join([*blocks, target])
