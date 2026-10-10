"""Compare dotted version strings without importing packaging."""

from __future__ import annotations

MINIMUM_TRANSFORMERS = "4.57.0"


def version_tuple(text: str) -> tuple[int, ...]:
    """Return the leading numeric components of a version string."""
    core = text.strip().split("+", 1)[0]
    parts: list[int] = []
    for piece in core.split("."):
        digits = []
        for char in piece:
            if char.isdigit():
                digits.append(char)
            else:
                break
        if not digits:
            break
        parts.append(int("".join(digits)))
    if not parts:
        raise ValueError(f"Could not parse a version from {text!r}.")
    return tuple(parts)


def version_at_least(found: str, minimum: str) -> bool:
    left = version_tuple(found)
    right = version_tuple(minimum)
    width = max(len(left), len(right))
    left = left + (0,) * (width - len(left))
    right = right + (0,) * (width - len(right))
    return left >= right
