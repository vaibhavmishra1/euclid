#!/usr/bin/env python3
"""
exp0_0: Zero-Shot Baseline Pipeline
====================================
Runs the CAQG pipeline without training to establish baseline metrics.

Usage:
    python baseline_pipeline.py --config config.yaml
    python baseline_pipeline.py --config config.yaml --num_seeds 10 --verbose
"""

import os
import sys
import json
import yaml
import argparse
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import random

import torch
import vllm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

# Timeout protection for grade_answer (can hang on certain inputs)

import stopit
 
# Add parent path for R-Zero utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'R-Zero'))


from evaluation.datasets_loader import get_dataset_handler
from mathruler.grader import extract_boxed_content, grade_answer



# =============================================================================
# Timeout-Protected Grading Function
# =============================================================================

def grade_answer_with_timeout(res1: str, res2: str, timeout: int = 10) -> Optional[bool]:
    """
    Wrapper for grade_answer with timeout protection.
    Returns None on timeout, True/False otherwise.
    
    Note: mathruler's grade_answer can hang on certain complex inputs,
    so we use stopit for thread-safe timeout control.
    """

    @stopit.threading_timeoutable(default=None)
    def _grade_with_timeout(r1, r2):
        return grade_answer(r1, r2)
    
    result = _grade_with_timeout(res1, res2, timeout=timeout)
    return result


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class QuestionData:
    """Data for a single generated question."""
    question: str
    transformation: str
    difficulty_estimate: int
    raw_output: str
    valid_format: bool


@dataclass 
class SolutionData:
    """Data for a single solution attempt."""
    solution: str
    extracted_answer: str
    token_length: int
    raw_output: str


@dataclass
class VerificationResult:
    """Result from answer verification (V1)."""
    r1_solution_correct: float
    r2_question_correct: float
    explanation: str
    raw_output: str


@dataclass
class NoveltyResult:
    """Result from novelty verification (V2)."""
    r3_novelty: float
    r3_difficulty: float
    r3_transformation: float
    explanation: str
    raw_output: str


@dataclass
class QuestionResult:
    """Complete result for one generated question."""
    question_data: QuestionData
    solutions: List[SolutionData]
    verification_results: List[VerificationResult]
    novelty_result: Optional[NoveltyResult]
    r_m: float  # Majority voting score
    majority_answer: str  # The majority-voted answer from solutions
    avg_solution_length: float


@dataclass
class IterationResult:
    """Result for one iteration."""
    iteration: int
    questions: List[QuestionResult]
    num_valid: int
    num_solvable: int
    avg_r_m: float
    avg_novelty: float
    termination_reason: Optional[str]


@dataclass
class SeedTrace:
    """Complete trace for one seed question."""
    seed_idx: int
    seed_question: str
    seed_answer: str
    iterations: List[IterationResult]
    final_iteration: int
    termination_reason: str


# =============================================================================
# Utility Functions
# =============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_prompt(prompt_path: str) -> Tuple[str, str]:
    """Load prompt file and extract system/user parts."""
    with open(prompt_path, 'r') as f:
        content = f.read()
    
    # Parse [SYSTEM] and [USER] sections
    system_match = re.search(r'\[SYSTEM\]\s*(.*?)\s*\[USER\]', content, re.DOTALL)
    user_match = re.search(r'\[USER\]\s*(.*?)$', content, re.DOTALL)
    
    system_prompt = system_match.group(1).strip() if system_match else ""
    user_template = user_match.group(1).strip() if user_match else ""
    
    return system_prompt, user_template


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract boxed content from text, using mathruler when available.
    
    Uses mathruler's extract_boxed_content for robust extraction,
    falling back to regex-based extraction if mathruler is not available.
    """
    # Try mathruler first (more robust, handles edge cases)
    if extract_boxed_content is not None:
        try:
            result = extract_boxed_content(text)
            if result:
                return result
        except Exception:
            pass  # Fall through to fallback
    
    # Fallback: regex-based extraction
    # First try simple pattern for non-nested braces
    pattern = r'\\boxed\{([^{}]*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1]
    
    # Handle nested braces manually
    prefix = r'\boxed{'
    start = text.find(prefix)
    if start == -1:
        # Also try without backslash (sometimes formatted differently)
        prefix = '\\boxed{'
        start = text.find(prefix)
        if start == -1:
            return None
    
    j = start + len(prefix)
    depth = 1
    while j < len(text) and depth > 0:
        if text[j] == '{':
            depth += 1
        elif text[j] == '}':
            depth -= 1
        j += 1
    
    if depth == 0:
        return text[start + len(prefix):j - 1]
    return None


def extract_tag_content(text: str, tag: str) -> Optional[str]:
    """
    Extract content between XML-style tags with robust handling.
    
    Handles various edge cases:
    - Properly closed tags: <tag>content</tag>
    - Missing closing tag: extracts until end of text or next tag
    - Nested tags: uses non-greedy matching for inner content
    - Leaked tags: strips any remaining tags from extracted content
    """
    # Primary pattern: proper tags
    pattern = rf'<{tag}>(.*?)</{tag}>'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        content = match.group(1).strip()
    else:
        # Fallback: tag without proper closing (extract until next tag or end)
        open_pattern = rf'<{tag}>\s*(.*?)(?=<[a-zA-Z_]+>|$)'
        match = re.search(open_pattern, text, re.DOTALL)
        if match:
            content = match.group(1).strip()
        else:
            return None
    
    # Validate: ensure content doesn't contain leaked tags (indicates format error)
    if content and re.match(r'^<[a-zA-Z_]+>', content):
        # Content starts with a tag - this is likely a format error
        # Try to extract just the text before the leaked tag
        clean_match = re.match(r'^([^<]+)', content)
        if clean_match:
            content = clean_match.group(1).strip()
        else:
            return None  # Content is all tags, invalid
    
    # Remove any trailing tag artifacts
    content = re.sub(r'</?[a-zA-Z_]+>\s*$', '', content).strip()
    
    return content if content else None


def extract_score(text: str, score_name: str) -> float:
    """Extract a numerical score from text."""
    patterns = [
        rf'{score_name}:\s*([\d.]+)',
        rf'{score_name}\s*=\s*([\d.]+)',
        rf'{score_name}:\s*\[([\d.]+)\]',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return 0.0


def _validate_question_format(question_text: Optional[str], claimed_answer: str, raw_output: str) -> bool:
    """
    Validate that a generated question has proper format and content.

    Note: Question generator no longer provides answers - that's done by the solver.

    Checks for:
    - Presence of question text
    - No leaked XML/HTML tags in question text
    - Minimum length requirements
    - No obvious format errors
    """
    # Basic presence check - question must exist
    if not question_text:
        return False

    # Check for leaked tags (indicates format error)
    if re.search(r'^<[a-zA-Z_]+>', question_text):
        return False
    if '<transformation>' in question_text.lower():
        return False
    if '<answer>' in question_text.lower():
        return False

    # Minimum length check (at least 20 chars for a real question)
    if len(question_text.strip()) < 20:
        return False

    # Check that question actually looks like a question (has some math or asks something)
    has_math = any(c in question_text for c in ['$', '\\', '=', '+', '-', '*', '/', '^', 'find', 'calculate', 'solve', 'simplify', 'evaluate', 'determine', 'prove', 'show'])
    has_question_indicator = any(word in question_text.lower() for word in ['?', 'what', 'find', 'calculate', 'determine', 'solve', 'how', 'prove', 'show', 'given'])

    if not (has_math or has_question_indicator):
        return False

    return True


def compute_majority_vote(answers: List[str]) -> Tuple[str, float]:
    """
    Compute majority vote from list of answers using symbolic equivalence.
    
    Uses mathruler's grade_answer for proper mathematical comparison,
    which handles equivalent representations like "2" vs "2.0" vs "\\frac{2}{1}".
    
    Following R-Zero's approach:
    1. First try cheap string comparison
    2. If that fails, use expensive symbolic grading (with timeout)
    3. Check both directions (A vs B and B vs A) for robustness
    """
    if not answers:
        return "", 0.0
    
    # Filter out empty answers
    valid_answers = [ans.strip() for ans in answers if ans and ans.strip()]
    if not valid_answers:
        return "", 0.0
    
    # Group answers by equivalence using symbolic comparison
    answer_counts: Dict[str, int] = {}
    
    for ans in valid_answers:
        matched = False
        
        for existing_answer in answer_counts:
            # OPTIMIZATION: Perform cheap string comparisons first
            if ans == existing_answer:
                answer_counts[existing_answer] += 1
                matched = True
                break
            
            # Normalized string comparison
            if ans.lower() == existing_answer.lower():
                answer_counts[existing_answer] += 1
                matched = True
                break
            
            # Check for common patterns (e.g., "no solution" variants)
            if 'no ' in ans.lower() and 'no ' in existing_answer.lower():
                answer_counts[existing_answer] += 1
                matched = True
                break
            
            # If cheap checks fail, use expensive symbolic grading
            # Check both directions for robustness (A vs B and B vs A)
            match_1 = grade_answer_with_timeout(ans, existing_answer, timeout=10)
            if match_1 is None:
                # Timeout - skip this comparison
                continue
            if match_1:
                answer_counts[existing_answer] += 1
                matched = True
                break
            
            # Try reverse direction
            match_2 = grade_answer_with_timeout(existing_answer, ans, timeout=10)
            if match_2 is None:
                continue
            if match_2:
                answer_counts[existing_answer] += 1
                matched = True
                break
        
        if not matched:
            # This is a new unique answer
            answer_counts[ans] = 1
    
    if not answer_counts:
        return "", 0.0
    
    # Find majority answer
    majority_ans = max(answer_counts, key=answer_counts.get)
    majority_count = answer_counts[majority_ans]
    
    return majority_ans, majority_count / len(valid_answers)


# =============================================================================
# Pipeline Components
# =============================================================================

class BaselinePipeline:
    """Main pipeline for baseline experiment."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.prompts = {}
        self.results_dir = config['output']['save_dir']
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
    def initialize(self):
        """Initialize model and load prompts."""
        print(f"[INFO] Loading model: {self.config['model']['name']}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['name'])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = LLM(
            model=self.config['model']['name'],
            tokenizer=self.config['model']['name'],
            gpu_memory_utilization=self.config['model']['gpu_memory_utilization'],
            tensor_parallel_size=self.config['model'].get('tensor_parallel_size', 1),
            seed=self.config['experiment']['seed'],
        )
        
        # Load prompts
        prompt_dir = os.path.dirname(self.config['prompts']['generator'])
        for name in ['generator', 'solver', 'verifier', 'novelty']:
            prompt_path = self.config['prompts'][name]
            self.prompts[name] = load_prompt(prompt_path)
            print(f"[INFO] Loaded prompt: {name}")
    
    def _create_chat_prompt(self, system: str, user: str) -> str:
        """Create chat prompt using tokenizer template."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=True
            )
        else:
            return f"system: {system}\nuser: {user}\nassistant:"
    
    def generate_questions(
        self, 
        seed_question: str, 
        seed_answer: str,
        current_question: str,
        current_answer: str,
        transforms_used: List[str],
        target_difficulty: int = 5
    ) -> List[QuestionData]:
        """
        Generate K new questions by transforming the current question.
        
        Args:
            seed_question: Original seed question (for context)
            seed_answer: Original seed answer (for context)
            current_question: The question to transform from
            current_answer: The answer to the current question
            transforms_used: List of transformations already applied in this chain
            target_difficulty: Target difficulty level (1-10)
        """
        system_prompt, user_template = self.prompts['generator']
        
        # Format transforms list
        transforms_str = ", ".join(transforms_used) if transforms_used else "None (this is the first iteration)"
        
        # Format user prompt with compressed context
        user_prompt = user_template.format(
            seed_question=seed_question,
            seed_answer=seed_answer,
            transforms_used=transforms_str,
            current_question=current_question,
            current_answer=current_answer,
            target_difficulty=target_difficulty
        )
        
        prompt = self._create_chat_prompt(system_prompt, user_prompt)
        
        # Generate K questions
        K = self.config['generation']['K']
        sampling_params = SamplingParams(
            max_tokens=self.config['generation']['max_tokens'],
            temperature=self.config['generation']['temperature'],
            top_p=self.config['generation']['top_p'],
            n=1,
            stop_token_ids=[self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else None,
        )
        
        # Generate K times (separate calls for diversity)
        questions = []
        prompts = [prompt] * K
        outputs = self.model.generate(prompts, sampling_params)
        
        for output in outputs:
            raw_text = output.outputs[0].text
            
            # Parse output
            question_text = extract_tag_content(raw_text, 'question')
            transformation = extract_tag_content(raw_text, 'transformation')
            difficulty_text = extract_tag_content(raw_text, 'difficulty_estimate')

            # Question generator does not provide answers - solver will generate them
            claimed_answer = ""
            
            # Parse difficulty
            try:
                difficulty = int(re.search(r'\d+', difficulty_text or "5").group())
            except:
                difficulty = 5
            
            # Validate question format and content
            valid_format = _validate_question_format(question_text, "", raw_text)

            questions.append(QuestionData(
                question=question_text or raw_text[:500],
                transformation=transformation or "unknown",
                difficulty_estimate=difficulty,
                raw_output=raw_text,
                valid_format=valid_format
            ))
        
        return questions
    
    def generate_solutions(self, question: str) -> List[SolutionData]:
        """Generate M solutions for a question."""
        system_prompt, user_template = self.prompts['solver']
        user_prompt = user_template.format(question=question)
        prompt = self._create_chat_prompt(system_prompt, user_prompt)
        
        M = self.config['generation']['M']
        sampling_params = SamplingParams(
            max_tokens=self.config['generation']['max_tokens'],
            temperature=self.config['generation']['temperature'],
            top_p=self.config['generation']['top_p'],
            n=M,  # Generate M samples in one call
            stop_token_ids=[self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else None,
        )
        
        outputs = self.model.generate([prompt], sampling_params)
        
        solutions = []
        for out in outputs[0].outputs:
            raw_text = out.text
            
            # Extract answer
            extracted = extract_boxed_answer(raw_text)
            
            solutions.append(SolutionData(
                solution=raw_text,
                extracted_answer=extracted or "",
                token_length=len(out.token_ids),
                raw_output=raw_text
            ))
        
        return solutions
    
    def verify_solution(
        self,
        question: str,
        solution: str
    ) -> VerificationResult:
        """Verify a solution using V1."""
        system_prompt, user_template = self.prompts['verifier']
        user_prompt = user_template.format(
            question=question,
            solution=solution
        )
        prompt = self._create_chat_prompt(system_prompt, user_prompt)
        
        sampling_params = SamplingParams(
            max_tokens=2048,
            temperature=0.3,  # Lower temp for verification
            top_p=0.95,
            n=1,
        )
        
        outputs = self.model.generate([prompt], sampling_params)
        raw_text = outputs[0].outputs[0].text
        
        # Extract scores
        r1 = extract_score(raw_text, 'r1_solution_correct')
        r2 = extract_score(raw_text, 'r2_question_correct')
        explanation = extract_tag_content(raw_text, 'explanation') or ""
        
        return VerificationResult(
            r1_solution_correct=r1,
            r2_question_correct=r2,
            explanation=explanation,
            raw_output=raw_text
        )
    
    def verify_novelty(
        self,
        seed_question: str,
        current_question: str,
        generated_question: str,
        transformations: str,
        r_m: float,
        avg_length: float,
        solution_variance: float
    ) -> NoveltyResult:
        """Verify novelty using V2. Compares generated question to both seed and current."""
        system_prompt, user_template = self.prompts['novelty']
        user_prompt = user_template.format(
            seed_question=seed_question,
            current_question=current_question,
            generated_question=generated_question,
            transformations=transformations,
            r_m=f"{r_m:.3f}",
            avg_length=f"{avg_length:.1f}",
            solution_variance=f"{solution_variance:.3f}"
        )
        prompt = self._create_chat_prompt(system_prompt, user_prompt)
        
        sampling_params = SamplingParams(
            max_tokens=2048,
            temperature=0.3,
            top_p=0.95,
            n=1,
        )
        
        outputs = self.model.generate([prompt], sampling_params)
        raw_text = outputs[0].outputs[0].text
        
        # Extract scores
        r3_novelty = extract_score(raw_text, 'r3_novelty')
        r3_difficulty = extract_score(raw_text, 'r3_difficulty')
        r3_transformation = extract_score(raw_text, 'r3_transformation')
        explanation = extract_tag_content(raw_text, 'detailed_explanation') or ""
        
        return NoveltyResult(
            r3_novelty=r3_novelty,
            r3_difficulty=r3_difficulty,
            r3_transformation=r3_transformation,
            explanation=explanation,
            raw_output=raw_text
        )
    
    def process_seed(
        self, 
        seed_idx: int,
        seed_question: str, 
        seed_answer: str,
        verbose: bool = True
    ) -> SeedTrace:
        """Process a single seed through all iterations."""
        iterations = []
        transforms_used = []  # Track transformations applied in this chain
        current_q = seed_question
        current_a = seed_answer
        termination_reason = "max_iterations"
        
        for iter_idx in range(self.config['generation']['max_iterations']):
            if verbose:
                print(f"  [Iteration {iter_idx + 1}]")
            
            # Step 1: Generate K questions with compressed context
            questions_data = self.generate_questions(
                seed_question=seed_question,      # Original seed (for context)
                seed_answer=seed_answer,          # Original seed answer
                current_question=current_q,       # Current question to transform
                current_answer=current_a,         # Current answer
                transforms_used=transforms_used,  # What's been tried
                target_difficulty=5 + iter_idx    # Increase difficulty each iteration
            )
            
            question_results = []
            
            for q_idx, q_data in enumerate(questions_data):
                if verbose:
                    print(f"    Question {q_idx + 1}/{len(questions_data)}: format_valid={q_data.valid_format}")
                
                if not q_data.valid_format:
                    # Skip invalid format questions but record them
                    question_results.append(QuestionResult(
                        question_data=q_data,
                        solutions=[],
                        verification_results=[],
                        novelty_result=None,
                        r_m=0.0,
                        majority_answer="",
                        avg_solution_length=0.0
                    ))
                    continue
                
                # Step 2: Generate M solutions
                solutions = self.generate_solutions(q_data.question)
                
                # Step 3: Verify each solution
                verification_results = []
                for sol in solutions:  # Verify first 3 for efficiency
                    ver_result = self.verify_solution(
                        q_data.question,
                        sol.solution
                    )
                    verification_results.append(ver_result)
                
                # Compute majority vote
                answers = [s.extracted_answer for s in solutions if s.extracted_answer]
                majority_answer, r_m = compute_majority_vote(answers)
                
                # Compute solution stats
                lengths = [s.token_length for s in solutions]
                avg_length = sum(lengths) / len(lengths) if lengths else 0
                variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths) if lengths else 0
                
                # Step 4: Verify novelty (compare to both seed and current)
                novelty_result = self.verify_novelty(
                    seed_question=seed_question,
                    current_question=current_q,
                    generated_question=q_data.question,
                    transformations=q_data.transformation,
                    r_m=r_m,
                    avg_length=avg_length,
                    solution_variance=variance
                )
                
                question_results.append(QuestionResult(
                    question_data=q_data,
                    solutions=solutions if self.config['output']['save_solutions'] else [],
                    verification_results=verification_results,
                    novelty_result=novelty_result,
                    r_m=r_m,
                    majority_answer=majority_answer,
                    avg_solution_length=avg_length
                ))
            
            # Compute iteration stats
            valid_questions = [
                q for q in question_results 
                if q.question_data.valid_format
            ]
            
            r2_scores = [
                v.r2_question_correct 
                for q in valid_questions 
                for v in q.verification_results
            ]
            avg_r2 = sum(r2_scores) / len(r2_scores) if r2_scores else 0
            
            r_m_scores = [q.r_m for q in valid_questions]
            avg_r_m = sum(r_m_scores) / len(r_m_scores) if r_m_scores else 0
            
            novelty_scores = [
                q.novelty_result.r3_novelty 
                for q in valid_questions 
                if q.novelty_result
            ]
            avg_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0
            
            # Count valid/solvable
            num_valid = sum(1 for q in valid_questions if avg_r2 > self.config['thresholds']['validity_t'])
            num_solvable = sum(1 for q in valid_questions if q.r_m > self.config['thresholds']['solvability_t'])
            
            # Check termination conditions
            K = self.config['generation']['K']
            
            # Condition (a): Too many invalid
            if num_valid < K * self.config['thresholds']['valid_ratio_t']:
                termination_reason = "invalid_questions"
                iterations.append(IterationResult(
                    iteration=iter_idx,
                    questions=question_results,
                    num_valid=num_valid,
                    num_solvable=num_solvable,
                    avg_r_m=avg_r_m,
                    avg_novelty=avg_novelty,
                    termination_reason=termination_reason
                ))
                break
            
            # Condition (b): Too difficult
            if avg_r_m < self.config['thresholds']['solvability_t']:
                termination_reason = "too_difficult"
                iterations.append(IterationResult(
                    iteration=iter_idx,
                    questions=question_results,
                    num_valid=num_valid,
                    num_solvable=num_solvable,
                    avg_r_m=avg_r_m,
                    avg_novelty=avg_novelty,
                    termination_reason=termination_reason
                ))
                break
            
            # Condition (c): Not novel
            if avg_novelty < self.config['thresholds']['novelty_t']:
                termination_reason = "not_novel"
                iterations.append(IterationResult(
                    iteration=iter_idx,
                    questions=question_results,
                    num_valid=num_valid,
                    num_solvable=num_solvable,
                    avg_r_m=avg_r_m,
                    avg_novelty=avg_novelty,
                    termination_reason=termination_reason
                ))
                break
            
            # Continue to next iteration
            iterations.append(IterationResult(
                iteration=iter_idx,
                questions=question_results,
                num_valid=num_valid,
                num_solvable=num_solvable,
                avg_r_m=avg_r_m,
                avg_novelty=avg_novelty,
                termination_reason="continue"
            ))
            
            # Update context for next iteration
            if valid_questions:
                # Constrained Novelty Selection:
                # 1. Filter to questions meeting stricter selection thresholds
                # 2. From those, pick highest novelty
                # 3. If no questions meet strict thresholds, fall back to all valid questions
                
                selection_solv_t = self.config['thresholds'].get('selection_solvability_t', 0.6)
                selection_valid_t = self.config['thresholds'].get('selection_validity_t', 0.7)
                
                # Filter candidates that meet stricter selection criteria
                premium_candidates = [
                    q for q in valid_questions
                    if q.r_m >= selection_solv_t and 
                       q.verification_result and q.verification_result.r2_question_correct >= selection_valid_t
                ]
                
                # Select from premium candidates if available, else fall back to all valid
                selection_pool = premium_candidates if premium_candidates else valid_questions
                
                # Pick highest novelty from the selection pool
                best_q = max(selection_pool, key=lambda x: x.novelty_result.r3_novelty if x.novelty_result else 0)
                current_q = best_q.question_data.question
                current_a = best_q.majority_answer
                
                if verbose and premium_candidates:
                    print(f"    -> Selected from {len(premium_candidates)} premium candidates (r_m>={selection_solv_t}, r2>={selection_valid_t})")
                elif verbose:
                    print(f"    -> No premium candidates, selected from {len(valid_questions)} valid questions")
                
                # Track the transformation used (for avoiding repetition)
                transform = best_q.question_data.transformation
                if transform and transform != "unknown":
                    # Extract transformation type (e.g., "TIGHTEN" from "TIGHTEN: Restrict...")
                    transform_type = transform.split(":")[0].split(",")[0].strip().upper()
                    if transform_type and transform_type not in transforms_used:
                        transforms_used.append(transform_type)
            
            if verbose:
                print(f"    -> valid={num_valid}, solvable={num_solvable}, avg_r_m={avg_r_m:.3f}, avg_novelty={avg_novelty:.3f}")
        
        return SeedTrace(
            seed_idx=seed_idx,
            seed_question=seed_question,
            seed_answer=seed_answer,
            iterations=iterations,
            final_iteration=len(iterations),
            termination_reason=termination_reason
        )
    
    def run(self, num_seeds: Optional[int] = None, verbose: bool = True) -> List[SeedTrace]:
        """Run the complete baseline pipeline."""
        # Load dataset
        if get_dataset_handler:
            handler = get_dataset_handler("math")
            questions, answers = handler.load_data()
        else:
            # Fallback: load directly
            import pandas as pd
            df = pd.read_csv("https://openaipublic.blob.core.windows.net/simple-evals/math_500_test.csv")
            questions = df['Question'].tolist()
            answers = df['Answer'].tolist()
        
        # Sample seeds
        n_seeds = num_seeds or self.config['dataset']['num_seeds']
        if self.config['dataset']['shuffle']:
            indices = random.sample(range(len(questions)), min(n_seeds, len(questions)))
        else:
            indices = list(range(min(n_seeds, len(questions))))
        
        seed_questions = [questions[i] for i in indices]
        seed_answers = [answers[i] for i in indices]
        
        print(f"[INFO] Processing {len(seed_questions)} seeds...")
        
        # Process each seed
        all_traces = []
        for idx, (q, a) in enumerate(tqdm(zip(seed_questions, seed_answers), total=len(seed_questions))):
            if verbose:
                print(f"\n[Seed {idx + 1}/{len(seed_questions)}]")
                print(f"  Q: {q[:100]}...")
            
            trace = self.process_seed(idx, q, a, verbose=verbose)
            all_traces.append(trace)
            
            # Save intermediate results
            if (idx + 1) % 10 == 0:
                self._save_intermediate(all_traces)
        
        # Save final results
        self._save_results(all_traces)
        
        return all_traces
    
    def _save_intermediate(self, traces: List[SeedTrace]):
        """Save intermediate results."""
        output_path = os.path.join(self.results_dir, "intermediate_results.json")
        self._save_json(traces, output_path)
    
    def _save_results(self, traces: List[SeedTrace]):
        """Save all results."""
        # Convert to serializable format
        def to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: to_dict(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, list):
                return [to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: to_dict(v) for k, v in obj.items()}
            else:
                return obj
        
        # Save raw outputs
        raw_path = os.path.join(self.results_dir, "raw_outputs.json")
        self._save_json([to_dict(t) for t in traces], raw_path)
        
        # Save iteration traces (lighter version)
        traces_path = os.path.join(self.results_dir, "iteration_traces.json")
        light_traces = []
        for t in traces:
            light_traces.append({
                "seed_idx": t.seed_idx,
                "seed_question": t.seed_question[:200],
                "final_iteration": t.final_iteration,
                "termination_reason": t.termination_reason,
                "iterations": [
                    {
                        "iteration": it.iteration,
                        "num_valid": it.num_valid,
                        "num_solvable": it.num_solvable,
                        "avg_r_m": it.avg_r_m,
                        "avg_novelty": it.avg_novelty,
                        "termination_reason": it.termination_reason
                    }
                    for it in t.iterations
                ]
            })
        self._save_json(light_traces, traces_path)
        
        print(f"[INFO] Results saved to {self.results_dir}/")
    
    def _save_json(self, data: Any, path: str):
        """Save data to JSON file."""
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="exp0_0: Zero-Shot Baseline Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--num_seeds", type=int, default=None, help="Override number of seeds")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with command line args
    if args.num_seeds:
        config['dataset']['num_seeds'] = args.num_seeds
    if args.output_dir:
        config['output']['save_dir'] = args.output_dir
    
    # Set random seed
    random.seed(config['experiment']['seed'])
    torch.manual_seed(config['experiment']['seed'])
    
    # Initialize and run pipeline
    pipeline = BaselinePipeline(config)
    pipeline.initialize()
    
    print("=" * 60)
    print("exp0_0: Zero-Shot Baseline Experiment")
    print("=" * 60)
    print(f"Model: {config['model']['name']}")
    print(f"Seeds: {config['dataset']['num_seeds']}")
    print(f"K (questions/iteration): {config['generation']['K']}")
    print(f"M (solutions/question): {config['generation']['M']}")
    print(f"Max iterations: {config['generation']['max_iterations']}")
    print("=" * 60)
    
    start_time = datetime.now()
    traces = pipeline.run(verbose=args.verbose or config['output']['verbose'])
    end_time = datetime.now()
    
    print("\n" + "=" * 60)
    print("Experiment Complete")
    print(f"Duration: {end_time - start_time}")
    print(f"Results saved to: {config['output']['save_dir']}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
