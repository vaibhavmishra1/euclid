#!/usr/bin/env python3
"""
Validate generated questions using OpenAI API (ASYNC VERSION - Much Faster).

This script uses async/await to process multiple questions concurrently,
making it 10-50x faster than the sequential version.

Usage:
     python validate_with_openai_async.py \
            --input_file balanced_questions_darwin_iter2_evaluated_fixed.json \
            --output_file balanced_questions__darwin_iter2openai_validated.json \
            --model gpt-4o \
            --concurrency 50
"""

import argparse
import json
import os
import asyncio
from typing import List, Dict, Optional
from tqdm.asyncio import tqdm as async_tqdm
import time

try:
    from openai import AsyncOpenAI
except ImportError:
    print("ERROR: openai package required. Install with: pip install openai")
    import sys
    sys.exit(1)

try:
    from mathruler.grader import grade_answer
except ImportError as e:
    print(f"ERROR: mathruler package required. Install with: pip install mathruler")
    print(f"Import error details: {e}")
    import sys
    sys.exit(1)


def extract_boxed_content(text: str) -> Optional[str]:
    """Extract content from \boxed{...} in the text."""
    results = []
    i = 0
    prefix = r'\boxed{'
    plen = len(prefix)
    
    while True:
        start = text.find(prefix, i)
        if start == -1:
            break
        
        j = start + plen
        depth = 1
        while j < len(text) and depth:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1
        
        results.append(text[start + plen : j - 1])
        i = j
    
    return results[-1] if results else None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if not answer:
        return ""
    
    answer = answer.strip()
    
    if answer.lower() == "wrong":
        return "WRONG"
    
    answer = answer.replace('\\text{', '').replace('}', '')
    answer = answer.replace('\\', '')
    answer = answer.replace(' ', '')
    
    return answer


def answers_match(answer1: str, answer2: str) -> bool:
    """Check if two answers match using mathruler's grade_answer."""
    if not answer1 or not answer2:
        return False
    
    norm1 = normalize_answer(answer1)
    norm2 = normalize_answer(answer2)
    
    if norm1 == "WRONG" and norm2 == "WRONG":
        return True
    
    if norm1 == "WRONG" or norm2 == "WRONG":
        return False
    
    try:
        match_1 = grade_answer(answer1, answer2)
        if match_1:
            return True
        
        match_2 = grade_answer(answer2, answer1)
        return bool(match_2)
    except Exception:
        return norm1 == norm2


async def solve_with_openai_async(
    client: AsyncOpenAI,
    question: str,
    model: str = "gpt-4o",
    max_retries: int = 3,
) -> Optional[str]:
    """Solve a math question using OpenAI API (async version)."""
    
    system_prompt = (
        "You are an expert mathematician. "
        "Your task is to solve the given math problem carefully and accurately. "
        "If the problem statement is invalid, ambiguous, or contains errors, "
        "respond with \\boxed{wrong}. "
        "Otherwise, provide a detailed solution and put your final answer inside \\boxed{}."
    )
    
    user_prompt = (
        f"Problem: {question}\n\n"
        "Please solve this problem step by step. "
        "Put your final answer inside \\boxed{{}}. "
        "If the problem is invalid or contains errors, respond with \\boxed{{wrong}}."
    )
    
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=2048,
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                return None
    
    return None


async def process_single_question(
    client: AsyncOpenAI,
    item: Dict,
    index: int,
    model: str,
    semaphore: asyncio.Semaphore,
) -> Dict:
    """Process a single question with rate limiting."""
    
    async with semaphore:  # Limit concurrent requests
        question = item.get("question", "")
        majority_answer = item.get("majority_answer", "0")
        
        # Skip if already processed
        if "openai_answer" in item and "openai_match" in item:
            return {"index": index, "status": "skipped", "item": item}
        
        # Solve with OpenAI
        response_text = await solve_with_openai_async(client, question, model=model)
        
        if response_text is None:
            item["openai_answer"] = "ERROR"
            item["openai_match"] = 0
            item["openai_response"] = ""
            return {"index": index, "status": "error", "item": item}
        
        # Extract answer
        openai_answer = extract_boxed_content(response_text)
        
        if openai_answer is None:
            openai_answer = "NO_ANSWER"
        
        # Check if answers match
        match = 1 if answers_match(openai_answer, majority_answer) else 0
        
        # Update item
        item["openai_answer"] = openai_answer
        item["openai_match"] = match
        item["openai_response"] = response_text
        
        status = "wrong" if normalize_answer(openai_answer) == "WRONG" else "success"
        return {"index": index, "status": status, "match": match, "item": item}


async def process_questions_async(
    input_file: str,
    output_file: str,
    api_key: str,
    model: str = "gpt-4o",
    start_index: int = 0,
    max_questions: Optional[int] = None,
    concurrency: int = 50,
    save_interval: int = 100,
) -> None:
    """Process questions with async/concurrent API calls."""
    
    # Initialize async OpenAI client
    client = AsyncOpenAI(api_key=api_key)
    
    # Load input data
    print(f"[Load] Reading from {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"[Load] Loaded {len(data)} questions")
    
    # Determine range to process
    end_index = len(data) if max_questions is None else min(start_index + max_questions, len(data))
    
    if start_index > 0 or end_index < len(data):
        print(f"[Process] Processing questions {start_index} to {end_index - 1}")
    
    print(f"[Config] Concurrency: {concurrency} parallel requests")
    print(f"[Config] Save interval: every {save_interval} questions")
    
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(concurrency)
    
    # Create tasks for all questions
    tasks = []
    for i in range(start_index, end_index):
        task = process_single_question(
            client, data[i], i, model, semaphore
        )
        tasks.append(task)
    
    # Process all tasks with progress bar
    print(f"\n[Process] Starting validation of {len(tasks)} questions...")
    start_time = time.time()
    
    processed_count = 0
    matched_count = 0
    wrong_count = 0
    error_count = 0
    skipped_count = 0
    
    # Process with progress bar
    results = []
    for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Validating"):
        result = await coro
        results.append(result)
        
        # Update statistics
        status = result["status"]
        if status == "skipped":
            skipped_count += 1
        elif status == "error":
            error_count += 1
            processed_count += 1
        elif status == "wrong":
            wrong_count += 1
            processed_count += 1
            if result.get("match"):
                matched_count += 1
        else:  # success
            processed_count += 1
            if result.get("match"):
                matched_count += 1
        
        # Save checkpoint periodically
        if len(results) % save_interval == 0:
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
    
    # Final save
    elapsed_time = time.time() - start_time
    
    print(f"\n[Save] Writing to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Statistics
    questions_per_sec = processed_count / elapsed_time if elapsed_time > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"VALIDATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total processed: {processed_count}")
    print(f"Skipped (already done): {skipped_count}")
    print(f"Matched answers: {matched_count} ({matched_count/max(processed_count,1)*100:.1f}%)")
    print(f"Invalid questions (wrong): {wrong_count} ({wrong_count/max(processed_count,1)*100:.1f}%)")
    print(f"API errors: {error_count}")
    print(f"")
    print(f"Time elapsed: {elapsed_time:.1f} seconds")
    print(f"Speed: {questions_per_sec:.1f} questions/second")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate generated questions using OpenAI API (async/fast version)"
    )
    
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model")
    parser.add_argument("--start_index", type=int, default=0, help="Start index")
    parser.add_argument("--max_questions", type=int, default=None, help="Max questions to process")
    parser.add_argument("--concurrency", type=int, default=50, 
                       help="Number of concurrent API requests (default: 50)")
    parser.add_argument("--save_interval", type=int, default=100,
                       help="Save checkpoint every N questions (default: 100)")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OpenAI API key required. Set --api_key or OPENAI_API_KEY env var")
        return
    
    # Run async processing
    asyncio.run(
        process_questions_async(
            input_file=args.input_file,
            output_file=args.output_file,
            api_key=api_key,
            model=args.model,
            start_index=args.start_index,
            max_questions=args.max_questions,
            concurrency=args.concurrency,
            save_interval=args.save_interval,
        )
    )


if __name__ == "__main__":
    main()
