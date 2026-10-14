"""Lossless parsing of Qwen3 thinking completions."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ParsedOutput:
    raw_completion: str
    reasoning_text: str
    final_text: str
    format_status: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def parse_completion(raw_completion: str) -> ParsedOutput:
    """Split a completion without changing or discarding its original text."""
    if not raw_completion or not raw_completion.strip():
        return ParsedOutput(raw_completion, "", "", "empty_output")

    open_pos = raw_completion.find("<think>")
    close_pos = raw_completion.find("</think>", open_pos + len("<think>") if open_pos >= 0 else 0)
    if open_pos >= 0 and close_pos < 0:
        reasoning = raw_completion[open_pos + len("<think>") :].strip()
        return ParsedOutput(raw_completion, reasoning, "", "missing_think_end")
    if close_pos >= 0:
        reasoning_start = open_pos + len("<think>") if open_pos >= 0 else 0
        reasoning = raw_completion[reasoning_start:close_pos].strip()
        final = raw_completion[close_pos + len("</think>") :].strip()
        if not final:
            return ParsedOutput(raw_completion, reasoning, "", "missing_final")
        return ParsedOutput(raw_completion, reasoning, final, "ok")

    # Qwen's chat template can place the opening <think> marker in the input,
    # so the newly decoded tokens legitimately contain only the closing marker.
    # A completion with neither marker is a non-thinking, final-only response.
    return ParsedOutput(raw_completion, "", raw_completion.strip(), "malformed")


def finish_reason(*, generated_token_count: int, max_new_tokens: int, eos_reached: bool) -> str:
    if eos_reached:
        return "eos"
    if generated_token_count >= max_new_tokens:
        return "length"
    return "unknown"
