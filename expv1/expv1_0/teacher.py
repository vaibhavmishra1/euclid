from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from .llm import LLMClient
from .types import TeacherVerdict
from .utils import read_text, render_template


@dataclass
class LLMTeacherVerifier:
    llm: LLMClient
    prompt_path: str
    max_tokens: int
    temperature: float
    top_p: float = 1.0

    def __post_init__(self) -> None:
        self.prompt_template = read_text(self.prompt_path)

    def verify(self, problem: str, answer: str) -> TeacherVerdict:
        prompt = render_template(self.prompt_template, {"problem": problem, "answer": answer})
        raw = self.llm.generate(prompt, max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p, n=1)[0]
        try:
            data = json.loads(raw)
            return TeacherVerdict(
                well_posed=bool(data.get("well_posed", False)),
                correct=bool(data.get("correct", False)),
                corrected_answer=data.get("corrected_answer", None),
                difficulty_1_to_5=int(data.get("difficulty_1_to_5", 3)),
                ambiguity_flags=list(data.get("ambiguity_flags", [])) if data.get("ambiguity_flags") else [],
                notes=str(data.get("notes", "")),
                raw_output=raw,
            )
        except Exception:
            # Fail closed (reject) if parsing fails; for early plumbing you can flip this.
            return TeacherVerdict(
                well_posed=False,
                correct=False,
                corrected_answer=None,
                difficulty_1_to_5=3,
                ambiguity_flags=["teacher_parse_failed"],
                notes="Teacher JSON parse failed",
                raw_output=raw,
            )


@dataclass
class TeacherEquivalenceJudge:
    llm: LLMClient
    prompt_path: str
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0

    def __post_init__(self) -> None:
        self.prompt_template = read_text(self.prompt_path)

    def equivalent(self, problem_a: str, problem_b: str) -> Optional[bool]:
        prompt = render_template(self.prompt_template, {"problem_a": problem_a, "problem_b": problem_b})
        raw = self.llm.generate(prompt, max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p, n=1)[0]
        try:
            data = json.loads(raw)
            return bool(data.get("equivalent", False))
        except Exception:
            return None

