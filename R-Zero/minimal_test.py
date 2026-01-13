#!/usr/bin/env python3
"""
Minimal test script for R-Zero to generate and dump challenger outputs.
This runs on a single GPU without the full training pipeline.
"""

import os
import sys
import json
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import regex as re
from mathruler.grader import extract_boxed_content

# Setup paths
STORAGE_PATH = os.getenv("STORAGE_PATH", "/workspace/rzero_storage")
DEBUG_DUMP_DIR = os.path.join(STORAGE_PATH, "rzero_debug_dumps")
os.makedirs(DEBUG_DUMP_DIR, exist_ok=True)

# R-Zero's questioner prompt (from dataset.py)
QUESTIONER_SYSTEM_PROMPT = """You are an expert competition-math problem setter.
FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial problem. The problem could come from any field of mathematics, including but not limited to algebra, geometry, number theory, combinatorics, prealgebra, probability, statistics, and calculus. Aim for a difficulty such that fewer than 30 % of advanced high-school students could solve it. Avoid re-using textbook clichés or famous contest problems.
THEN, without revealing any of your private thoughts, output **exactly** the following two blocks:

<question>
{The full problem statement on one or more lines}
</question>

\\boxed{final_answer}

Do NOT output anything else—no explanations, no extra markup."""

QUESTIONER_USER_PROMPT = "Generate one new, challenging reasoning question now. Remember to format the output exactly as instructed."


def build_prompt(tokenizer):
    """Build the R-Zero questioner prompt"""
    messages = [
        {"role": "system", "content": QUESTIONER_SYSTEM_PROMPT},
        {"role": "user", "content": QUESTIONER_USER_PROMPT}
    ]
    
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        prompt = f"system: {messages[0]['content']}\nuser: {messages[1]['content']}"
    
    return prompt


def parse_output(output):
    """Parse question and answer from challenger output"""
    questions = re.findall(r"<question>(.*?)</question>", output, re.DOTALL)
    answers = extract_boxed_content(output)
    
    if questions and answers:
        return {
            "question": questions[-1].strip(),
            "answer": answers[-1].strip() if isinstance(answers, list) else answers.strip(),
            "is_valid": True
        }
    else:
        return {
            "question": "",
            "answer": "",
            "is_valid": False
        }


def run_minimal_test(model_name: str, num_samples: int = 50, max_new_tokens: int = 2048):
    """Run minimal test to generate and dump challenger outputs"""
    
    print(f"="*60)
    print(f"R-Zero Minimal Test")
    print(f"Model: {model_name}")
    print(f"Samples: {num_samples}")
    print(f"="*60)
    
    # Load model and tokenizer
    print("\n[1/4] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Try to load on GPU, fall back to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"Using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    
    # Build prompt
    print("\n[2/4] Building prompt...")
    prompt = build_prompt(tokenizer)
    print(f"Prompt length: {len(prompt)} chars")
    print(f"\n--- PROMPT PREVIEW ---")
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    print("--- END PREVIEW ---\n")
    
    # Dump prompt
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_file = os.path.join(DEBUG_DUMP_DIR, f"{timestamp}_prompt.txt")
    with open(prompt_file, 'w') as f:
        f.write(prompt)
    print(f"Saved prompt to: {prompt_file}")
    
    # Generate samples
    print(f"\n[3/4] Generating {num_samples} samples...")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    outputs_data = []
    parsed_data = []
    
    for i in tqdm(range(num_samples), desc="Generating"):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_p=0.99,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode output (only the generated part)
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Parse
        parsed = parse_output(raw_output)
        
        outputs_data.append({
            "timestamp": datetime.now().isoformat(),
            "raw_output": raw_output,
            "index": i
        })
        
        parsed_data.append({
            "timestamp": datetime.now().isoformat(),
            "question": parsed["question"],
            "answer": parsed["answer"],
            "is_valid": parsed["is_valid"],
            "index": i
        })
    
    # Dump outputs
    print(f"\n[4/4] Dumping results...")
    
    outputs_file = os.path.join(DEBUG_DUMP_DIR, f"{timestamp}_challenger_outputs.jsonl")
    with open(outputs_file, 'w') as f:
        for entry in outputs_data:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved raw outputs to: {outputs_file}")
    
    parsed_file = os.path.join(DEBUG_DUMP_DIR, f"{timestamp}_parsed_questions.jsonl")
    with open(parsed_file, 'w') as f:
        for entry in parsed_data:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved parsed questions to: {parsed_file}")
    
    # Summary
    valid_count = sum(1 for p in parsed_data if p["is_valid"])
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples:    {num_samples}")
    print(f"Valid outputs:    {valid_count} ({100*valid_count/num_samples:.1f}%)")
    print(f"Invalid outputs:  {num_samples - valid_count} ({100*(num_samples-valid_count)/num_samples:.1f}%)")
    print(f"\nOutput files saved to: {DEBUG_DUMP_DIR}")
    
    # Show a few examples
    print(f"\n{'='*60}")
    print(f"SAMPLE OUTPUTS (first 3)")
    print(f"{'='*60}")
    for i in range(min(3, len(outputs_data))):
        print(f"\n--- Sample {i+1} ---")
        output = outputs_data[i]["raw_output"]
        print(output[:800] + "..." if len(output) > 800 else output)
        print(f"\nParsed: {parsed_data[i]}")
    
    return outputs_data, parsed_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="R-Zero Minimal Test")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Model to use for generation")
    parser.add_argument("--samples", type=int, default=50,
                        help="Number of samples to generate")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max new tokens per sample")
    args = parser.parse_args()
    
    run_minimal_test(args.model, args.samples, args.max_tokens)
