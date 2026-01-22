from __future__ import annotations

import re
from typing import Optional


def extract_tag_content(text: str, tag: str) -> Optional[str]:
    """Extract content between <tag>...</tag> (non-greedy)."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip()


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract the last \\boxed{...} content from a string.
    Handles nested braces.
    """
    prefix = r"\boxed{"
    i = 0
    last: Optional[str] = None
    while True:
        start = text.find(prefix, i)
        if start == -1:
            break
        j = start + len(prefix)
        depth = 1
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            last = text[start + len(prefix) : j - 1].strip()
        i = j
    return last


def strip_boxed_content(text: str) -> str:
    """
    Remove all \\boxed{...} from text to prevent answer leakage.
    Handles nested braces.
    """
    result = text
    prefix = r"\boxed{"
    
    while True:
        start = result.find(prefix)
        if start == -1:
            break
        j = start + len(prefix)
        depth = 1
        while j < len(result) and depth:
            if result[j] == "{":
                depth += 1
            elif result[j] == "}":
                depth -= 1
            j += 1
        # Remove the entire \boxed{...} including braces
        result = result[:start] + result[j:]
    
    return result.strip()


def has_boxed_content(text: str) -> bool:
    """Check if text contains any \\boxed{...}."""
    return r"\boxed{" in text

