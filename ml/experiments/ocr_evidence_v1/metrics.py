from __future__ import annotations

from difflib import SequenceMatcher

from app.ocr_evidence import normalize_ocr_text, search_text


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def best_match(reference: str, lines: list[str]) -> tuple[str, float]:
    expected = search_text(reference)
    candidates = [
        " ".join(lines[start : start + width])
        for start in range(len(lines))
        for width in range(1, min(8, len(lines) - start) + 1)
    ]
    if not expected or not candidates:
        return "", 0.0
    best = max(candidates, key=lambda value: SequenceMatcher(None, expected, search_text(value)).ratio())
    return best, SequenceMatcher(None, expected, search_text(best)).ratio()


def character_accuracy(pairs: list[tuple[str, str]]) -> float:
    denominator = sum(len(normalize_ocr_text(reference)) for reference, _ in pairs)
    errors = sum(
        edit_distance(normalize_ocr_text(reference), normalize_ocr_text(prediction))
        for reference, prediction in pairs
    )
    return max(0.0, 1 - errors / max(1, denominator))


def timestamp_error(timestamp: float, intervals: list[list[float]]) -> float:
    return min(0.0 if start <= timestamp <= end else min(abs(timestamp - start), abs(timestamp - end)) for start, end in intervals)
