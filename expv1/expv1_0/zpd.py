from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .grading import is_correct
from .llm import LLMClient
from .text_parse import extract_boxed_answer
from .types import ZPDResult
from .utils import read_text, render_template


@dataclass
class ZPDScorer:
    llm: LLMClient
    prompt_path: str
    rollouts: int
    max_tokens: int
    temperature: float
    top_p: float

    def __post_init__(self) -> None:
        self.prompt_template = read_text(self.prompt_path)

    def score(self, problem: str, gold_answer: str) -> ZPDResult:
        prompt = render_template(self.prompt_template, {"problem": problem})
        outs = self.llm.generate(prompt, max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p, n=self.rollouts)

        preds: List[str] = []
        correct = 0
        for out in outs:
            pred = extract_boxed_answer(out) or ""
            preds.append(pred)
            if is_correct(pred, gold_answer):
                correct += 1

        p_succ = correct / max(1, self.rollouts)
        return ZPDResult(
            p_succ=p_succ,
            rollouts=self.rollouts,
            num_correct=correct,
            details={"preds": preds},
        )

