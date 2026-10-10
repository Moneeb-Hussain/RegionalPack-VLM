"""Inspect a decoded string without judging whether the OCR is correct."""

from __future__ import annotations


def inspect_response(text: str | None) -> str:
    """Classify a response for technical faults.

    ``ok`` means the string is non-empty and not a single-character dump.
    It does not mean the transcription or the brand name is right.
    """
    if text is None:
        return "empty"
    stripped = text.strip()
    if stripped == "":
        return "empty"
    letters = [char for char in stripped if not char.isspace()]
    if len(letters) >= 20 and letters.count(letters[0]) / len(letters) >= 0.9:
        return "repeated_character"
    return "ok"
