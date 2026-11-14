"""Shared parsing for model replies."""

from __future__ import annotations

import json
import re

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_BRACES = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_object(text: str) -> dict | None:
    """Pull one JSON object out of a model reply.

    Tries, in order: the contents of a markdown code fence, the whole reply,
    and the outermost {...} span. Returns None when none of them is a JSON
    object.
    """
    raw = text.strip()
    candidates: list[str] = []

    fence = _FENCE.search(raw)
    if fence:
        candidates.append(fence.group(1).strip())
    candidates.append(raw)
    braces = _BRACES.search(raw)
    if braces:
        candidates.append(braces.group())

    for candidate in candidates:
        try:
            data = json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None
