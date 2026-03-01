"""
Prompt builders for extraction and normalization.

Extraction outputs invariant mathematical knowledge points.
Normalization validates and deduplicates the extracted knowledge points.
"""

from __future__ import annotations

import json
from typing import Mapping


def build_extraction_prompt(
    qid: str,
    question: str,
    solution: str = "",
    max_concepts: int = 5,
) -> str:
    """
    Build a single-string prompt to feed to a text-generation model.
    """
    solution_section = ""
    if solution:
        solution_section = f"""
Solution:
{solution}
"""

    return f"""You will be given a mathematics problem and its solution. Extract 1 to {max_concepts} mathematical knowledge points that are required to solve this problem.

WHAT ARE KNOWLEDGE POINTS?
Knowledge points are INVARIANT mathematical facts—theorems, identities, properties, formulas, or definitions that:
- Are universally true and not specific to this problem
- Would be found in a mathematics textbook or reference
- Could be applied to ANY problem involving the same concept

KEY CRITERION: QUESTION GENERATION TEST
The extracted knowledge points must be SUFFICIENT for someone to create a SIMILAR question.
If a teacher is given ONLY your knowledge points list, they should be able to construct a new problem of the same type and difficulty level. Ask yourself: "Can I generate a similar problem using just these knowledge points?" If not, you are missing something.

STRICT REQUIREMENTS:

1. PRECISE TERMINOLOGY: Use accurate, professional mathematical terms.
   - Good: "Pythagorean theorem: a² + b² = c²", "difference of squares identity: a² - b² = (a+b)(a-b)"
   - Bad: "geometry", "algebra", "basic math"

2. INVARIANT KNOWLEDGE ONLY: Extract general mathematical truths, NOT problem-specific details.
   - Good: "quadratic formula: x = (-b ± √(b²-4ac))/(2a)" (applies to any quadratic)
   - Bad: "x = 3" or "f(2) = 5" (specific to this problem)

3. COMPLETE FOR QUESTION GENERATION: Include ALL core concepts needed to create a similar problem.
   - If the problem uses right triangles, include "Pythagorean theorem: a² + b² = c²"
   - If it involves finding roots, include "quadratic formula" or "Vieta's formulas"
   - If it uses circle properties, include the specific circle theorem used

4. ATOMIC & UNIQUE: Each item must be a single mathematical fact. No duplicates or overlapping concepts.

5. NO PROCEDURES: Exclude methods like "substitution", "simplification". Only mathematical facts.

6. INCLUDE THE FORMULA/IDENTITY: When a theorem or property has a formula, always include it.
   - Good: "sum of arithmetic sequence: S_n = n(a₁ + aₙ)/2"
   - Good: "Vieta's formulas: for ax² + bx + c = 0, sum of roots = -b/a, product = c/a"
   - Good: "Law of Cosines: c² = a² + b² - 2ab·cos(C)"

OUTPUT FORMAT (JSON only):

{{
  "id": "",
  "question": "",
  "knowledge_points": [
    "<knowledge point 1 with formula if applicable>",
    "<knowledge point 2>",
    ...
  ]
}}

Problem:
{question}
{solution_section}"""


def build_normalization_prompt(
    raw: Mapping[str, object],
    max_concepts: int = 5,
) -> str:
    """
    Build a normalization prompt to validate and deduplicate extracted knowledge points.
    """
    raw_json = json.dumps(raw, ensure_ascii=False, indent=2)

    return f"""Review and normalize the extracted mathematical knowledge points.

KEY CRITERION: QUESTION GENERATION TEST
The final knowledge points must be SUFFICIENT for a teacher to create a SIMILAR question.
If something essential is missing, the list fails. If something is vague or procedural, remove it.

VALIDATION RULES:

1. KEEP only INVARIANT mathematical facts (theorems, identities, properties, definitions with formulas)
2. ENSURE COMPLETENESS: Are all core concepts needed to generate a similar question present?
3. REMOVE vague terms: "basic algebra", "arithmetic", "calculation"
4. REMOVE procedures: "substitution", "simplification", "cross multiplication"
5. REMOVE problem-specific values: "x = 3", "the answer is 5"
6. USE canonical names with formulas: "Pythagorean theorem: a² + b² = c²" not just "Pythagorean theorem"
7. INCLUDE formulas with theorems: "Law of Sines: a/sin(A) = b/sin(B) = c/sin(C)"
8. DEDUPLICATE semantically equivalent items
9. MAX {max_concepts} knowledge points, ordered by importance

Input:
{raw_json}

Return normalized JSON (set "question" to ""):
{{
  "id": "...",
  "question": "",
  "knowledge_points": ["..."]
}}
"""
