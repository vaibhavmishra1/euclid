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
            if gold_answer and is_correct(pred, gold_answer):
                correct += 1

        if gold_answer:
            p_succ = correct / max(1, self.rollouts)
            modal_answer = ""
        else:
            # Self-consistency mode: use agreement on the modal non-empty answer
            non_empty = [p for p in preds if p]
            if non_empty:
                modal_answer = max(set(non_empty), key=non_empty.count)
                p_succ = non_empty.count(modal_answer) / max(1, len(preds))
            else:
                modal_answer = ""
                p_succ = 0.0
        
        # Store full outputs for COT extraction
        return ZPDResult(
            p_succ=p_succ,
            rollouts=self.rollouts,
            num_correct=correct,
            details={
                "preds": preds, 
                "modal_answer": modal_answer,
                "full_outputs": outs,  # Store full solver outputs for COT
            },
        )

