from __future__ import annotations

from typing import List

from .types import SeedExample
from .text_parse import extract_boxed_answer
from .utils import read_jsonl


def load_seed_examples_from_jsonl(path: str, max_seeds: int) -> List[SeedExample]:
    rows = read_jsonl(path)
    seeds: List[SeedExample] = []
    for r in rows[:max_seeds]:
        seeds.append(
            SeedExample(
                problem_id=str(r.get("problem_id", "")),
                problem=str(r.get("problem", "")),
                solution=str(r.get("solution", "")),
                answer=str(r.get("answer", "")),
                domain=r.get("domain"),
            )
        )
    return seeds


def load_seed_examples_from_hendrycks_math(
    *,
    dataset_config: str,
    split: str,
    max_seeds: int,
    seed: int,
) -> List[SeedExample]:
    """
    Load seeds from EleutherAI/hendrycks_math (MATH dataset on HuggingFace).
    We keep (problem, solution) and extract the boxed final answer from solution.
    """
    from datasets import load_dataset

    ds = load_dataset("EleutherAI/hendrycks_math", dataset_config, split=split)
    ds = ds.shuffle(seed=seed)
    if max_seeds > 0:
        ds = ds.select(range(min(max_seeds, len(ds))))

    seeds: List[SeedExample] = []
    for i, ex in enumerate(ds):
        problem = str(ex.get("problem", ""))
        solution = str(ex.get("solution", ""))
        answer = extract_boxed_answer(solution) or ""
        seeds.append(
            SeedExample(
                problem_id=f"hendrycks_math/{dataset_config}/{split}/{i}",
                problem=problem,
                solution=solution,
                answer=answer,
                domain=dataset_config,
            )
        )
    return seeds

