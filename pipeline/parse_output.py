"""Parse GSM8K completions without scanning arbitrary trailing numbers."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


FINAL_PATTERN = re.compile(r"####\s*([^\n]+)")


@dataclass(frozen=True)
class ParsedOutput:
    raw_completion: str
    reasoning_text: str
    final_text: str
    format_status: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def parse_completion(raw_completion: str) -> ParsedOutput:
    if not raw_completion or not raw_completion.strip():
        return ParsedOutput(raw_completion, "", "", "empty_output")
    matches = list(FINAL_PATTERN.finditer(raw_completion))
    if not matches:
        return ParsedOutput(raw_completion, raw_completion.strip(), "", "missing_final")
    answers = [match.group(1).strip() for match in matches]
    if any(not answer for answer in answers):
        return ParsedOutput(raw_completion, raw_completion[:matches[0].start()].strip(), "", "malformed_final")
    status = "ok" if len(set(answers)) == 1 else "ambiguous_final"
    return ParsedOutput(
        raw_completion,
        raw_completion[:matches[0].start()].strip(),
        f"#### {answers[-1]}",
        status,
    )


def finish_reason(*, generated_token_count: int, max_new_tokens: int, eos_reached: bool) -> str:
    if eos_reached:
        return "eos"
    if generated_token_count >= max_new_tokens:
        return "length"
    return "unknown"
