from __future__ import annotations

import re
from typing import List, Optional


def split_possible_answers(answer: str) -> List[str]:
    # Very light heuristic: split on commas for multiple roots like "2, 3"
    parts = [p.strip() for p in answer.split(",")]
    return [p for p in parts if p]


def normalize_answer_str(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("$", "")
    return s


def answers_equivalent(pred: str, gold: str) -> bool:
    pred = pred.strip()
    gold = gold.strip()
    if not pred or not gold:
        return False

    # cheap exact match
    if normalize_answer_str(pred) == normalize_answer_str(gold):
        return True

    # try math-verify if available
    try:
        from math_verify import parse, verify

        return bool(verify(parse(gold), parse(pred)))
    except Exception:
        return False


def is_correct(pred: str, gold: str) -> bool:
    """
    Check correctness against a gold answer that may contain multiple options (comma-separated).
    """
    for g in split_possible_answers(gold):
        if answers_equivalent(pred, g):
            return True
    return False

