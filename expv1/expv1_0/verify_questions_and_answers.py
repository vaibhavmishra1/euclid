import json
import random
import os
import uuid
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
from openai import OpenAI
from tqdm import tqdm
import argparse

# Difficulty buckets based on p_succ
DIFFICULTY_BUCKETS = {
    "very_hard": (0.0, 0.2),
    "hard": (0.2, 0.4),
    "medium": (0.4, 0.6),
    "easy": (0.6, 0.8),
    "very_easy": (0.8, 1.0),
}

def load_accepted_questions(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load all accepted questions from JSONL file."""
    questions = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions

def filter_by_p_succ(questions: List[Dict[str, Any]], p_min: float = 0.3, p_max: float = 0.8) -> List[Dict[str, Any]]:
    """Filter questions by p_succ threshold."""
    filtered = []
    for q in questions:
        p_succ = q.get('zpd', {}).get('p_succ', 0.5)
        if p_min <= p_succ <= p_max:
            filtered.append(q)
    return filtered

def bucket_questions(questions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Bucket questions by p_succ into difficulty levels."""
    buckets = defaultdict(list)
    
    for q in questions:
        p_succ = q.get('zpd', {}).get('p_succ', 0.5)
        
        for bucket_name, (min_p, max_p) in DIFFICULTY_BUCKETS.items():
            if min_p <= p_succ < max_p or (bucket_name == "very_easy" and p_succ == 1.0):
                buckets[bucket_name].append(q)
                break
    
    return buckets

def sample_questions(buckets: Dict[str, List[Dict[str, Any]]], n_per_bucket: int = 100) -> Dict[str, List[Dict[str, Any]]]:
    """Sample n questions from each bucket."""
    sampled = {}
    for bucket_name, questions in buckets.items():
        if len(questions) >= n_per_bucket:
            sampled[bucket_name] = random.sample(questions, n_per_bucket)
        else:
            sampled[bucket_name] = questions
            print(f"Warning: Bucket '{bucket_name}' has only {len(questions)} questions, using all of them.")
    return sampled

def is_quota_error(error: str) -> bool:
    """Check if error is a quota-related error."""
    if not error:
        return False
    error_lower = str(error).lower()
    return "quota" in error_lower or "insufficient" in error_lower or "rate limit" in error_lower or "429" in error_lower

def verify_question_with_openai(client: OpenAI, question_data: Dict[str, Any]) -> Dict[str, Any]:
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
            "verification": result
        }
    except Exception as e:
        error_str = str(e)
        
        # Print quota errors immediately
        if is_quota_error(error_str):
            print(f"\n⚠️  QUOTA ERROR: {error_str[:200]}")
            if "429" in error_str:
                print("   → Rate limit exceeded (HTTP 429)")
        
        return {
            "status": "error",
            "error": error_str,
            "verification": None
        }

def save_verified_questions(sampled_questions: Dict[str, List[Dict[str, Any]]],
                           verification_results: Dict[str, List[Dict[str, Any]]],
                           output_path: str) -> int:
    """Save verified questions with unique IDs and all metadata to JSONL file."""
    verified_questions = []
    
    for bucket_name in DIFFICULTY_BUCKETS.keys():
        if bucket_name not in sampled_questions:
            continue
        
        questions = sampled_questions[bucket_name]
        results = verification_results[bucket_name]
        
        for question_data, verification_result in zip(questions, results):
            # Generate unique ID
            question_id = str(uuid.uuid4())
            
            # Build verified question entry with all metadata
            verified_entry = {
                "id": question_id,
                "difficulty_bucket": bucket_name,
                "original_data": question_data,  # Keep all original data
                "verification": {
                    "status": verification_result.get("status", "unknown"),
                    "verification_result": verification_result.get("verification"),
                    "error": verification_result.get("error") if verification_result.get("status") == "error" else None,
                },
                # Extract key verification fields for easy access
                "verification_summary": {
                    "question_validity": verification_result.get("verification", {}).get("question_validity") if verification_result.get("verification") else None,
                    "answer_correctness": verification_result.get("verification", {}).get("answer_correctness") if verification_result.get("verification") else None,
                    "cot_quality": verification_result.get("verification", {}).get("cot_quality") if verification_result.get("verification") else None,
                    "answer_leakage": verification_result.get("verification", {}).get("answer_leakage") if verification_result.get("verification") else None,
                    "overall_quality": verification_result.get("verification", {}).get("overall_quality") if verification_result.get("verification") else None,
                    "reasoning": verification_result.get("verification", {}).get("reasoning") if verification_result.get("verification") else None,
                }
            }
            
            verified_questions.append(verified_entry)
    
    # Save to JSONL file
    with open(output_path, 'w') as f:
        for entry in verified_questions:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    return len(verified_questions)

def generate_report(sampled_questions: Dict[str, List[Dict[str, Any]]], 
                   verification_results: Dict[str, List[Dict[str, Any]]],
                   output_path: str):
    """Generate a comprehensive verification report."""
    
    report = {
        "summary": {},
        "by_difficulty": {},
        "overall_statistics": {},
        "recommendations": []
    }
    
    # Overall statistics
    total_questions = sum(len(questions) for questions in sampled_questions.values())
    total_valid = 0
    total_correct = 0
    total_high_cot = 0
    total_no_leakage = 0
    total_excellent = 0
    
    # Per-bucket statistics
    for bucket_name in DIFFICULTY_BUCKETS.keys():
        if bucket_name not in sampled_questions:
            continue
            
        questions = sampled_questions[bucket_name]
        results = verification_results[bucket_name]
        
        bucket_stats = {
            "total_questions": len(questions),
            "valid_questions": 0,
            "correct_answers": 0,
            "high_cot_quality": 0,
            "no_answer_leakage": 0,
            "excellent_quality": 0,
            "breakdown": {
                "question_validity": {"valid": 0, "invalid": 0},
                "answer_correctness": {"correct": 0, "incorrect": 0, "ambiguous": 0},
                "cot_quality": {"high": 0, "medium": 0, "low": 0},
                "answer_leakage": {"yes": 0, "no": 0},
                "overall_quality": {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
            }
        }
        
        for result in results:
            if result["status"] == "success" and result["verification"]:
                v = result["verification"]
                
                # Count valid questions
                if v.get("question_validity") == "valid":
                    bucket_stats["valid_questions"] += 1
                    total_valid += 1
                
                # Count correct answers
                if v.get("answer_correctness") == "correct":
                    bucket_stats["correct_answers"] += 1
                    total_correct += 1
                
                # Count high COT quality
                if v.get("cot_quality") == "high":
                    bucket_stats["high_cot_quality"] += 1
                    total_high_cot += 1
                
                # Count no answer leakage
                if v.get("answer_leakage") == "no":
                    bucket_stats["no_answer_leakage"] += 1
                    total_no_leakage += 1
                
                # Count excellent overall quality
                if v.get("overall_quality") == "excellent":
                    bucket_stats["excellent_quality"] += 1
                    total_excellent += 1
                
                # Update breakdown
                for key in ["question_validity", "answer_correctness", "cot_quality", "answer_leakage", "overall_quality"]:
                    value = v.get(key)
                    if value and value in bucket_stats["breakdown"][key]:
                        bucket_stats["breakdown"][key][value] += 1
        
        # Calculate percentages
        n = len(questions)
        if n > 0:
            bucket_stats["valid_percentage"] = (bucket_stats["valid_questions"] / n) * 100
            bucket_stats["correct_percentage"] = (bucket_stats["correct_answers"] / n) * 100
            bucket_stats["high_cot_percentage"] = (bucket_stats["high_cot_quality"] / n) * 100
            bucket_stats["no_leakage_percentage"] = (bucket_stats["no_answer_leakage"] / n) * 100
            bucket_stats["excellent_percentage"] = (bucket_stats["excellent_quality"] / n) * 100
        
        report["by_difficulty"][bucket_name] = bucket_stats
    
    # Overall statistics
    if total_questions > 0:
        report["overall_statistics"] = {
            "total_questions": total_questions,
            "valid_percentage": (total_valid / total_questions) * 100,
            "correct_percentage": (total_correct / total_questions) * 100,
            "high_cot_percentage": (total_high_cot / total_questions) * 100,
            "no_leakage_percentage": (total_no_leakage / total_questions) * 100,
            "excellent_percentage": (total_excellent / total_questions) * 100
        }
    
    # Generate recommendations
    if report["overall_statistics"]["valid_percentage"] < 85:
        report["recommendations"].append("Question validity is below 85%. Review the generation pipeline for well-posedness issues.")
    
    if report["overall_statistics"]["correct_percentage"] < 85:
        report["recommendations"].append("Answer correctness is below 85%. Verify the solver's answers or improve answer extraction.")
    
    if report["overall_statistics"]["high_cot_percentage"] < 70:
        report["recommendations"].append("COT quality is below 70%. Consider improving the solver model or prompt.")
    
    if report["overall_statistics"]["no_leakage_percentage"] < 95:
        report["recommendations"].append("Answer leakage detected in >5% of questions. Strengthen format validation in the generation pipeline.")
    
    if report["overall_statistics"]["excellent_percentage"] < 50:
        report["recommendations"].append("Overall quality is below 50%. Consider comprehensive pipeline improvements.")
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("VERIFICATION REPORT SUMMARY")
    print("="*80)
    print(f"\nTotal Questions Verified: {total_questions}")
    print(f"\nOverall Statistics:")
    print(f"  Valid Questions: {report['overall_statistics']['valid_percentage']:.1f}%")
    print(f"  Correct Answers: {report['overall_statistics']['correct_percentage']:.1f}%")
    print(f"  High COT Quality: {report['overall_statistics']['high_cot_percentage']:.1f}%")
    print(f"  No Answer Leakage: {report['overall_statistics']['no_leakage_percentage']:.1f}%")
    print(f"  Excellent Overall: {report['overall_statistics']['excellent_percentage']:.1f}%")
    
    print(f"\nBy Difficulty:")
    for bucket_name, stats in report["by_difficulty"].items():
        print(f"\n  {bucket_name.upper().replace('_', ' ')}:")
        print(f"    Valid: {stats['valid_percentage']:.1f}%")
        print(f"    Correct: {stats['correct_percentage']:.1f}%")
        print(f"    High COT: {stats['high_cot_percentage']:.1f}%")
        print(f"    No Leakage: {stats['no_leakage_percentage']:.1f}%")
        print(f"    Excellent: {stats['excellent_percentage']:.1f}%")
    
    if report["recommendations"]:
        print(f"\nRecommendations:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"  {i}. {rec}")
    
    print(f"\nFull report saved to: {output_path}")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Verify generated questions using OpenAI API")
    parser.add_argument("--input", type=str, required=True, help="Path to accepted.jsonl file")
    parser.add_argument("--output", type=str, default="verification_report.json", help="Output path for report (verified questions saved to accepted_verified.jsonl)")
    parser.add_argument("--verified-output", type=str, default=None, help="Output path for verified questions JSONL (default: accepted_verified.jsonl)")
    parser.add_argument("--samples-per-bucket", type=int, default=100, help="Number of samples per difficulty bucket")
    parser.add_argument("--p-min", type=float, default=0.3, help="Minimum p_succ threshold for filtering questions")
    parser.add_argument("--p-max", type=float, default=0.8, help="Maximum p_succ threshold for filtering questions")
    parser.add_argument("--openai-api-key", type=str, default=None, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Initialize OpenAI client
    api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key must be provided via --openai-api-key or OPENAI_API_KEY environment variable")
    
    client = OpenAI(api_key=api_key)
    
    # Load questions
    print(f"Loading questions from {args.input}...")
    questions = load_accepted_questions(args.input)
    print(f"Loaded {len(questions)} questions")
    
    # Filter by p_succ threshold
    print(f"Filtering questions with p_succ between {args.p_min} and {args.p_max}...")
    filtered_questions = filter_by_p_succ(questions, p_min=args.p_min, p_max=args.p_max)
    print(f"Filtered to {len(filtered_questions)} questions (removed {len(questions) - len(filtered_questions)})")
    
    if len(filtered_questions) == 0:
        print("ERROR: No questions match the p_succ threshold. Exiting.")
        return
    
    # Bucket questions
    print("Bucketing questions by difficulty...")
    buckets = bucket_questions(filtered_questions)
    for bucket_name, bucket_questions_list in buckets.items():
        print(f"  {bucket_name}: {len(bucket_questions_list)} questions")
    
    # Sample questions
    print(f"\nSampling {args.samples_per_bucket} questions per bucket...")
    sampled = sample_questions(buckets, args.samples_per_bucket)
    total_sampled = sum(len(q) for q in sampled.values())
    print(f"Total sampled: {total_sampled} questions")
    
    # Verify questions
    print("\nVerifying questions with OpenAI API...")
    verification_results = {}
    
    for bucket_name, questions_list in sampled.items():
        print(f"\nVerifying {bucket_name} bucket ({len(questions_list)} questions)...")
        results = []
        for q in tqdm(questions_list, desc=f"  {bucket_name}"):
            result = verify_question_with_openai(client, q)
            results.append(result)
        verification_results[bucket_name] = results
    
    # Save verified questions with unique IDs
    if args.verified_output:
        verified_output_path = args.verified_output
    else:
        # Auto-generate path based on input file location
        input_dir = os.path.dirname(args.input) if os.path.dirname(args.input) else '.'
        verified_output_path = os.path.join(input_dir, 'accepted_verified.jsonl')
    
    print(f"\nSaving verified questions to {verified_output_path}...")
    num_verified = save_verified_questions(sampled, verification_results, verified_output_path)
    print(f"Saved {num_verified} verified questions with unique IDs")
    
    # Generate report
    report_output_path = args.output if args.output.endswith('.json') else 'verification_report.json'
    print(f"\nGenerating report to {report_output_path}...")
    generate_report(sampled, verification_results, report_output_path)
    
    print("\nVerification complete!")
    print(f"  - Verified questions: {verified_output_path}")
    print(f"  - Verification report: {report_output_path}")

if __name__ == "__main__":
    main()