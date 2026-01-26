import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from openai import OpenAI
from tqdm import tqdm
import argparse

def is_quota_error(error: Optional[str]) -> bool:
    """Check if error is a quota-related error."""
    if not error:
        return False
    error_lower = str(error).lower()
    return "quota" in error_lower or "insufficient" in error_lower or "rate limit" in error_lower or "429" in error_lower

def verify_question_with_openai(client: OpenAI, question_data: Dict[str, Any], retry_count: int = 0) -> Dict[str, Any]:
    """Use OpenAI API to verify a question."""
    problem = question_data['candidate']['problem']
    answer = question_data['candidate']['answer']
    modal_answer = question_data.get('verification', {}).get('modal_answer', '')
    solution = question_data.get('verification', {}).get('solution', '')
    p_succ = question_data.get('zpd', {}).get('p_succ', 0.5)
    
    prompt = f"""You are an expert mathematician evaluating a generated mathematical problem. Please verify the following:

**Problem:**
{problem}

**Solver's Answer (from multiple rollouts):**
{modal_answer}

**Solver's Chain-of-Thought Solution:**
{solution}

Please evaluate the following aspects and provide your assessment in JSON format:

1. **Question Validity**: Is the problem well-posed, mathematically sound, and unambiguous? (valid/invalid)
2. **Answer Correctness**: Is the given answer correct? (correct/incorrect/ambiguous)
3. **COT Quality**: Is the chain-of-thought solution logical, coherent, and correct? (high/medium/low)
4. **Answer Leakage**: Is the answer or solution leaked in the problem statement? (yes/no)
5. **Overall Quality**: Overall assessment (excellent/good/fair/poor)

Provide your response in the following JSON format:
{{
    "question_validity": "valid/invalid",
    "answer_correctness": "correct/incorrect/ambiguous",
    "cot_quality": "high/medium/low",
    "answer_leakage": "yes/no",
    "overall_quality": "excellent/good/fair/poor",
    "reasoning": "Brief explanation of your assessment"
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert mathematician. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return {
            "status": "success",
            "verification": result,
            "retry_count": retry_count
        }
    except Exception as e:
        error_str = str(e)
        
        # Check if it's a quota error and print
        if is_quota_error(error_str):
            print(f"\n⚠️  QUOTA ERROR (retry {retry_count}): {error_str[:200]}")
            if "429" in error_str:
                print("   → Rate limit exceeded. Waiting before retry...")
        
        return {
            "status": "error",
            "error": error_str,
            "verification": None,
            "retry_count": retry_count,
            "is_quota_error": is_quota_error(error_str)
        }

def load_verified_questions(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load verified questions from JSONL file."""
    questions = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions

def retry_quota_errors(
    verified_file: str,
    output_file: str,
    api_key: str,
    max_retries: int = 3,
    retry_delay: int = 60
) -> Dict[str, Any]:
    """Retry verification for questions that failed due to quota errors."""
    
    client = OpenAI(api_key=api_key)
    
    # Load existing verified questions
    print(f"Loading verified questions from {verified_file}...")
    verified_questions = load_verified_questions(verified_file)
    print(f"Loaded {len(verified_questions)} verified questions")
    
    # Identify questions with quota errors
    quota_error_questions = []
    for entry in verified_questions:
        verification = entry.get("verification", {})
        status = verification.get("status", "unknown")
        error = verification.get("error")
        
        if status == "error" and is_quota_error(error):
            quota_error_questions.append(entry)
    
    print(f"\nFound {len(quota_error_questions)} questions with quota errors")
    
    if len(quota_error_questions) == 0:
        print("No quota errors to retry. Exiting.")
        return {
            "total_quota_errors": 0,
            "retried": 0,
            "successful": 0,
            "still_failed": 0
        }
    
    # Retry verification
    stats = {
        "total_quota_errors": len(quota_error_questions),
        "retried": 0,
        "successful": 0,
        "still_failed": 0,
        "by_bucket": {}
    }
    
    print(f"\nRetrying verification for {len(quota_error_questions)} questions...")
    print(f"Max retries: {max_retries}, Delay between retries: {retry_delay}s")
    
    # Create a mapping of question IDs to their indices for quick lookup
    question_id_to_index = {entry["id"]: idx for idx, entry in enumerate(verified_questions)}
    
    for entry in tqdm(quota_error_questions, desc="Retrying quota errors"):
        question_id = entry["id"]
        bucket = entry.get("difficulty_bucket", "unknown")
        original_data = entry.get("original_data", {})
        
        if bucket not in stats["by_bucket"]:
            stats["by_bucket"][bucket] = {"total": 0, "successful": 0, "failed": 0}
        stats["by_bucket"][bucket]["total"] += 1
        
        # Retry with exponential backoff
        retry_count = 0
        last_result = None
        
        while retry_count < max_retries:
            if retry_count > 0:
                # Exponential backoff: wait longer for each retry
                wait_time = retry_delay * (2 ** (retry_count - 1))
                print(f"\n⏳ Waiting {wait_time}s before retry {retry_count + 1}/{max_retries}...")
                time.sleep(wait_time)
            
            result = verify_question_with_openai(client, original_data, retry_count=retry_count)
            stats["retried"] += 1
            
            if result["status"] == "success":
                # Update the entry in verified_questions
                idx = question_id_to_index[question_id]
                verified_questions[idx]["verification"] = {
                    "status": result["status"],
                    "verification_result": result["verification"],
                    "error": None,
                    "retry_count": result.get("retry_count", 0)
                }
                verified_questions[idx]["verification_summary"] = {
                    "question_validity": result["verification"].get("question_validity"),
                    "answer_correctness": result["verification"].get("answer_correctness"),
                    "cot_quality": result["verification"].get("cot_quality"),
                    "answer_leakage": result["verification"].get("answer_leakage"),
                    "overall_quality": result["verification"].get("overall_quality"),
                    "reasoning": result["verification"].get("reasoning"),
                }
                stats["successful"] += 1
                stats["by_bucket"][bucket]["successful"] += 1
                print(f"✅ Successfully verified question {question_id[:8]}... (bucket: {bucket})")
                break
            else:
                last_result = result
                if is_quota_error(result.get("error")):
                    retry_count += 1
                    if retry_count < max_retries:
                        continue
                else:
                    # Non-quota error, don't retry
                    print(f"❌ Non-quota error for question {question_id[:8]}...: {result.get('error', '')[:100]}")
                    break
        
        if last_result and last_result["status"] != "success":
            stats["still_failed"] += 1
            stats["by_bucket"][bucket]["failed"] += 1
            print(f"❌ Still failed after {max_retries} retries: {question_id[:8]}...")
    
    # Save updated verified questions
    print(f"\nSaving updated verified questions to {output_file}...")
    with open(output_file, 'w') as f:
        for entry in verified_questions:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Saved {len(verified_questions)} verified questions")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Retry verification for questions that failed due to quota errors")
    parser.add_argument("--verified-file", type=str, required=True, help="Path to accepted_verified.jsonl file")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: overwrites input file)")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries per question")
    parser.add_argument("--retry-delay", type=int, default=60, help="Initial delay in seconds between retries (exponential backoff)")
    parser.add_argument("--openai-api-key", type=str, default=None, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Initialize OpenAI client
    api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key must be provided via --openai-api-key or OPENAI_API_KEY environment variable")
    
    output_file = args.output or args.verified_file
    
    # Run retry
    stats = retry_quota_errors(
        verified_file=args.verified_file,
        output_file=output_file,
        api_key=api_key,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay
    )
    
    # Print summary
    print("\n" + "="*80)
    print("RETRY SUMMARY")
    print("="*80)
    print(f"Total quota errors found: {stats['total_quota_errors']}")
    print(f"Questions retried: {stats['retried']}")
    print(f"Successfully verified: {stats['successful']}")
    print(f"Still failed: {stats['still_failed']}")
    
    if stats['by_bucket']:
        print(f"\nBy Difficulty Bucket:")
        for bucket, bucket_stats in stats['by_bucket'].items():
            print(f"  {bucket}:")
            print(f"    Total: {bucket_stats['total']}")
            print(f"    Successful: {bucket_stats['successful']}")
            print(f"    Failed: {bucket_stats['failed']}")
    
    print(f"\nUpdated file saved to: {output_file}")
    print("="*80)

if __name__ == "__main__":
    main()
