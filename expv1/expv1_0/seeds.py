from __future__ import annotations

from typing import List

from .types import SeedExample
from .utils import read_jsonl


def load_seed_examples_from_jsonl(path: str, max_seeds: int) -> List[SeedExample]:
    rows = read_jsonl(path)
    seeds: List[SeedExample] = []
    for r in rows[:max_seeds]:
        seeds.append(
            SeedExample(
                problem_id=str(r.get("problem_id", "")),
                problem=str(r.get("problem", "")),
                answer=str(r.get("answer", "")),
                domain=r.get("domain"),
            )
        )
    return seeds

