from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Concept:
    type: str
    name: str

    @property
    def key(self) -> str:
        return f"{self.type}:{self.name}".lower()


@dataclass
class SeedExample:
    problem_id: str
    problem: str
    answer: str
    domain: Optional[str] = None


@dataclass
class Spec:
    required_concepts: List[Concept]
    hop_mode: str  # explicit | implicit_2hop | implicit_3hop
    target_domain: str = ""
    answer_type: str = "final_answer"


@dataclass
class CandidateSample:
    spec: Spec
    problem: str
    answer: str
    raw_output: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TeacherVerdict:
    well_posed: bool
    correct: bool
    corrected_answer: Optional[str]
    difficulty_1_to_5: int
    ambiguity_flags: List[str] = field(default_factory=list)
    notes: str = ""
    raw_output: str = ""


@dataclass
class ZPDResult:
    p_succ: float
    rollouts: int
    num_correct: int
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AcceptedSample:
    candidate: CandidateSample
    teacher: TeacherVerdict
    zpd: Optional[ZPDResult] = None
    dedup_reason: Optional[str] = None

