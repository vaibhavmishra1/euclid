"""
Prompts for Knowledge-Set-Based Curriculum Learning

This module contains prompt templates for:
1. Challenger: Generate questions based on a SET of knowledge points and difficulty
2. Solver: Solve generated questions (same as original R-Zero)

Difficulty scale: 1-5 (matching original MATH dataset levels)
"""

import re
from typing import List, Tuple

# Difficulty level descriptions (1-5 scale, matching MATH dataset)
DIFFICULTY_DESCRIPTIONS = {
    1: "Level 1 - basic problem suitable for students just learning the concepts",
    2: "Level 2 - straightforward problem requiring solid understanding",
    3: "Level 3 - moderate problem requiring reasoning and concept connections",
    4: "Level 4 - challenging problem requiring creative thinking",
    5: "Level 5 - very challenging competition-level problem",
}


def get_difficulty_description(level: int) -> str:
    """Get description for a difficulty level."""
    level = max(1, min(5, level))  # Clamp to valid range 1-5
    return DIFFICULTY_DESCRIPTIONS.get(level, DIFFICULTY_DESCRIPTIONS[3])


def format_knowledge_points(kp_list: List[str]) -> str:
    """Format a list of knowledge points for the prompt."""
    if not kp_list:
        return "- General number theory concepts"
    
    formatted = []
    for i, kp in enumerate(kp_list, 1):
        formatted.append(f"{i}. {kp}")
    
    return "\n".join(formatted)


def build_knowledge_challenger_system_prompt(knowledge_points: List[str], difficulty: int) -> str:
    """
    Build the system prompt for the knowledge-based challenger.
    
    Args:
        knowledge_points: List of knowledge points to use (all of them)
        difficulty: Target difficulty level (1-5)
    
    Returns:
        System prompt string
    """
    difficulty = max(1, min(5, difficulty))  # Clamp to 1-5
    difficulty_desc = get_difficulty_description(difficulty)
    kp_formatted = format_knowledge_points(knowledge_points)
    
    system_prompt = f"""You are an expert mathematics problem setter specializing in Number Theory.

Your task is to create a new, original math problem that requires ALL of the following knowledge points to solve.

**Knowledge Points Required:**
{kp_formatted}

**Target Difficulty Level:** {difficulty}/5 ({difficulty_desc})

**Instructions:**
1. FIRST, in your private scratch-pad, think step-by-step to design a brand-new problem that:
   - Requires ALL the listed knowledge points to solve (not just one or two)
   - Matches the target difficulty level exactly
   - Is NOT a copy of any existing textbook or contest problem
   - Has a well-defined, unique numerical or symbolic answer

2. THEN, output **exactly** the following two blocks (nothing else):

<question>
{{The full problem statement on one or more lines}}
</question>

\\boxed{{final_answer}}

**Important Rules:**
- The problem MUST require ALL the knowledge points listed above
- The difficulty MUST match the target level ({difficulty}/5)
- The answer MUST be definitive and verifiable
- Do NOT output any explanations or extra markup
- Do NOT reveal your scratch-pad thinking"""

    return system_prompt


def build_knowledge_challenger_user_prompt() -> str:
    """Build the user prompt for the knowledge-based challenger."""
    return "Generate one new math problem now. Remember to format the output exactly as instructed with <question></question> tags and \\boxed{answer}."


def build_knowledge_challenger_messages(knowledge_points: List[str], difficulty: int) -> list:
    """
    Build complete message list for knowledge-based challenger.
    
    Args:
        knowledge_points: List of knowledge points (the full set)
        difficulty: Target difficulty level (1-5)
    
    Returns:
        List of message dicts for chat format
    """
    return [
        {
            "role": "system",
            "content": build_knowledge_challenger_system_prompt(knowledge_points, difficulty)
        },
        {
            "role": "user",
            "content": build_knowledge_challenger_user_prompt()
        }
    ]


# Solver prompt (same as original R-Zero - DO NOT MODIFY)
SOLVER_SYSTEM_PROMPT = r"Please reason step by step, and put your final answer within \boxed{}."


def build_solver_messages(question: str) -> list:
    """Build messages for solver (same as R-Zero)."""
    return [
        {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]


def extract_boxed_answer(text: str) -> str:
    """
    Extract answer from \\boxed{...}, handling nested braces.
    
    Args:
        text: Text containing \\boxed{answer}
    
    Returns:
        The answer inside boxed, or empty string if not found
    """
    # Find \boxed{
    match = re.search(r'\\boxed\{', text)
    if not match:
        return ""
    
    start = match.end()
    depth = 1
    i = start
    
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    
    if depth == 0:
        return text[start:i-1].strip()
    
    return ""


# Example usage
if __name__ == "__main__":
    # Test prompt generation with multiple KPs
    kp_list = [
        "Base conversion: For a number in base b, the value is calculated as Σ(d_i * b^i)",
        "Properties of digits in base systems: In base b, digits must satisfy 0 ≤ d < b",
        "Linear equation: If ax = b, then x = b/a"
    ]
    
    messages = build_knowledge_challenger_messages(kp_list, difficulty=4)
    
    print("System Prompt:")
    print("-" * 50)
    print(messages[0]["content"])
    print("-" * 50)
    print("\nUser Prompt:")
    print(messages[1]["content"])
    
    # Test boxed extraction
    test_cases = [
        r"\boxed{42}",
        r"\boxed{2^{10}}",
        r"\boxed{\frac{1}{2}}",
        r"The answer is \boxed{x^{2} + 1}.",
    ]
    
    print("\n\nBoxed Extraction Tests:")
    for test in test_cases:
        result = extract_boxed_answer(test)
        print(f"  {test!r} → {result!r}")
