"""Debug dumping utility for R-Zero outputs"""
import os
import json
from datetime import datetime

STORAGE_PATH = os.getenv("STORAGE_PATH", "/workspace/rzero_storage")
DEBUG_DUMP_DIR = os.path.join(STORAGE_PATH, "rzero_debug_dumps")

def ensure_dump_dir():
    os.makedirs(DEBUG_DUMP_DIR, exist_ok=True)
    return DEBUG_DUMP_DIR

def dump_prompts(prompts, metadata=None):
    """Dump prompts sent to challenger"""
    dump_dir = ensure_dump_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(dump_dir, f"{timestamp}_prompts.jsonl")
    
    with open(filepath, 'w') as f:
        for i, prompt in enumerate(prompts):
            entry = {
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "index": i,
            }
            if metadata and i < len(metadata):
                entry.update(metadata[i])
            f.write(json.dumps(entry) + "\n")
    
    print(f"[DEBUG DUMP] Wrote {len(prompts)} prompts to {filepath}")
    return filepath

def dump_outputs(outputs, metadata=None):
    """Dump raw outputs from challenger"""
    dump_dir = ensure_dump_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(dump_dir, f"{timestamp}_challenger_outputs.jsonl")
    
    with open(filepath, 'w') as f:
        for i, output in enumerate(outputs):
            entry = {
                "timestamp": datetime.now().isoformat(),
                "raw_output": output,
                "index": i,
            }
            if metadata and i < len(metadata):
                entry.update(metadata[i])
            f.write(json.dumps(entry) + "\n")
    
    print(f"[DEBUG DUMP] Wrote {len(outputs)} outputs to {filepath}")
    return filepath

def dump_parsed_questions(questions):
    """Dump parsed questions from challenger outputs"""
    dump_dir = ensure_dump_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(dump_dir, f"{timestamp}_parsed_questions.jsonl")
    
    with open(filepath, 'w') as f:
        for i, q in enumerate(questions):
            entry = {
                "timestamp": datetime.now().isoformat(),
                "question": q.get("question", ""),
                "answer": q.get("answer", ""),
                "is_valid": bool(q.get("question")) and bool(q.get("answer")),
                "index": i,
            }
            f.write(json.dumps(entry) + "\n")
    
    valid_count = sum(1 for q in questions if q.get("question") and q.get("answer"))
    print(f"[DEBUG DUMP] Wrote {len(questions)} questions ({valid_count} valid) to {filepath}")
    return filepath
