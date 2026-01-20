from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .llm import LLMClient
from .text_parse import extract_boxed_answer, extract_tag_content
from .types import CandidateSample, Concept, Spec
from .utils import read_text, render_template


def _concepts_to_strings(concepts: List[Concept]) -> List[Dict[str, str]]:
    return [{"type": c.type, "name": c.name} for c in concepts]


@dataclass
class LLMQuestionGenerator:
    llm: LLMClient
    prompt_path: str
    max_tokens: int
    temperature: float
    top_p: float

    def __post_init__(self) -> None:
        self.prompt_template = read_text(self.prompt_path)

    def generate_one(self, spec: Spec) -> CandidateSample:
        prompt = render_template(
            self.prompt_template,
            {
                "required_concepts": _concepts_to_strings(spec.required_concepts),
                "hop_mode": spec.hop_mode,
                "target_domain": spec.target_domain,
                "answer_type": spec.answer_type,
            },
        )
        raw = self.llm.generate(prompt, max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p, n=1)[0]
        q = extract_tag_content(raw, "question") or raw.strip()
        # Only generate questions - answers will be verified by solvers
        return CandidateSample(spec=spec, problem=q, answer="", raw_output=raw, metadata={})

