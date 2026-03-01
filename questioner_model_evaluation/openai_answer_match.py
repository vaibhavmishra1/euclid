#!/usr/bin/env python3
"""
Solve questions with an OpenAI GPT model and compare to an existing solver's majority answer.

Input format:
  JSON with top-level key "results" containing items like:
    {
      "question_index": int,
      "question": str,
      "max_voting_score": float,
      "most_frequent_answer": str,
      "most_frequent_count": int,
      "num_unique_answers": int,
      "answer_counts": {...}
    }

This script:
  - calls OpenAI for each question
  - extracts the final boxed answer (or 'nan')
  - compares OpenAI answer to most_frequent_answer
  - adds `openai_match` = 1 if equivalent else 0
  - writes JSON + JSONL outputs

Auth:
  - set OPENAI_API_KEY env var, or pass --api-key
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the *last* \\boxed{...} content from text (handles nested braces)."""
    if not text:
        return None
    prefix = r"\boxed{"
    i = 0
    last: Optional[str] = None
    while True:
        start = text.find(prefix, i)
        if start == -1:
            break
        j = start + len(prefix)
        depth = 1
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            last = text[start + len(prefix) : j - 1].strip()
        i = j
    return last


def normalize_answer(answer: str) -> str:
    """Light normalization to make string comparisons less brittle."""
    if answer is None:
        return ""
    answer = answer.strip()
    answer = re.sub(r"\s+", "", answer)
    answer = re.sub(r"\\(text|mathrm|mathbf|mathit)\{([^}]*)\}", r"\2", answer)
    answer = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", answer)
    answer = re.sub(r"\\(left|right)", "", answer)
    answer = answer.replace("$", "")
    return answer.lower()


@dataclass(frozen=True)
class OpenAIResult:
    raw_text: str
    boxed: str


def openai_solve_question(*, client: Any, model: str, question: str, timeout_s: int) -> OpenAIResult:
    """
    Call OpenAI Responses API to solve a question. Expects OpenAI python SDK v1+.
    """
    prompt = (
        "Please reason step by step, and put your final answer within \\boxed{}.\n"
        "If the question is invalid/ill-posed, or if you cannot find a solution, output exactly \\boxed{nan}.\n\n"
        f"Problem:\n{question}\n"
    )

    # Use Responses API (preferred). Keep it minimal and deterministic-ish.
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "You are a careful math solver."},
            {"role": "user", "content": prompt},
        ],
        timeout=timeout_s,
    )

    # Best-effort to retrieve text (SDK shapes can vary a bit by version).
    raw_text = ""
    if hasattr(resp, "output_text") and isinstance(resp.output_text, str):
        raw_text = resp.output_text
    else:
        # Fallback: concatenate any text chunks we can find
        try:
            parts = []
            for item in getattr(resp, "output", []) or []:
                for c in getattr(item, "content", []) or []:
                    t = getattr(c, "text", None)
                    if isinstance(t, str):
                        parts.append(t)
            raw_text = "\n".join(parts)
        except Exception:
            raw_text = str(resp)

    boxed = extract_boxed_answer(raw_text) or ""
    boxed = boxed.strip()
    if boxed == "":
        # If the model didn't box, treat as nan per our contract
        boxed = "nan"
    return OpenAIResult(raw_text=raw_text, boxed=boxed)


def build_equivalence_checker(timeout_s: int = 10):
    """
    Return a function eq(a,b)->bool that uses MathRuler when available, else falls back to simple checks.
    """
    try:
        import stopit
        from mathruler.grader import grade_answer
    except Exception:
        stopit = None
        grade_answer = None

    if stopit is None or grade_answer is None:
        def simple_eq(a: str, b: str) -> bool:
            return normalize_answer(a) == normalize_answer(b)
        return simple_eq

    @stopit.threading_timeoutable(default="TIMED_OUT")
    def grade_answer_with_timeout(x: str, y: str):
        return grade_answer(x, y)

    def eq(a: str, b: str) -> bool:
        a = a or ""
        b = b or ""
        if normalize_answer(a) == normalize_answer(b):
            return True
        try:
            r1 = grade_answer_with_timeout(a, b, timeout=timeout_s)
            if r1 != "TIMED_OUT" and r1:
                return True
            r2 = grade_answer_with_timeout(b, a, timeout=timeout_s)
            if r2 != "TIMED_OUT" and r2:
                return True
        except Exception:
            return False
        return False

    return eq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Solve questions with OpenAI and match against most_frequent_answer.")
    p.add_argument(
        "--input",
        required=True,
        help="Path to metadata JSON (with top-level 'results' list).",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write outputs into.",
    )
    p.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model name (default: gpt-4o-mini).",
    )
    p.add_argument(
        "--api-key",
        default="",
        help="OpenAI API key (optional). If omitted, uses OPENAI_API_KEY env var.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of questions (0 = all).",
    )
    p.add_argument(
        "--sleep-s",
        type=float,
        default=0.0,
        help="Sleep between requests (seconds). Useful for rate limiting.",
    )
    p.add_argument(
        "--timeout-s",
        type=int,
        default=120,
        help="Timeout per OpenAI request in seconds (default: 120).",
    )
    p.add_argument(
        "--mathruler-timeout-s",
        type=int,
        default=10,
        help="Timeout per MathRuler equivalence check in seconds (default: 10).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="If output JSONL exists, skip already processed question_index entries.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't call OpenAI; just print counts and exit.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / (in_path.stem + f".openai_{args.model}.json")
    out_jsonl = out_dir / (in_path.stem + f".openai_{args.model}.jsonl")

    obj = json.loads(in_path.read_text(encoding="utf-8"))
    items = obj.get("results", [])
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    already_done = set()
    if args.resume and out_jsonl.exists():
        with out_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    qi = rec.get("question_index")
                    if qi is not None:
                        already_done.add(qi)
                except Exception:
                    continue

    print(f"Loaded {len(items)} items from {in_path}")
    if already_done:
        print(f"Resume enabled: {len(already_done)} items already present in {out_jsonl.name}")

    if args.dry_run:
        return

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: missing OpenAI API key. Set OPENAI_API_KEY or pass --api-key.", file=sys.stderr)
        sys.exit(2)

    try:
        from openai import OpenAI
    except Exception as e:
        print(
            "ERROR: OpenAI SDK not installed. Install with: python3 -m pip install openai\n"
            f"Import error: {e}",
            file=sys.stderr,
        )
        sys.exit(2)

    client = OpenAI(api_key=api_key)
    eq = build_equivalence_checker(timeout_s=args.mathruler_timeout_s)

    processed = []
    # Append-only JSONL so you can resume safely
    jsonl_f = out_jsonl.open("a", encoding="utf-8")
    try:
        for idx, item in enumerate(items):
            qi = item.get("question_index")
            if args.resume and qi in already_done:
                continue

            question = item.get("question", "")
            expected = item.get("most_frequent_answer", "") or ""

            # Basic guard
            if not question or len(str(question).strip()) < 5:
                record = dict(item)
                record.update(
                    {
                        "openai_model": args.model,
                        "openai_boxed_answer": "nan",
                        "openai_raw_text": "",
                        "openai_match": 0,
                        "openai_error": "empty_question",
                    }
                )
                processed.append(record)
                jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl_f.flush()
                continue

            # Retry with exponential backoff
            backoff = 2.0
            last_err = None
            for attempt in range(1, 6):
                try:
                    r = openai_solve_question(
                        client=client,
                        model=args.model,
                        question=question,
                        timeout_s=args.timeout_s,
                    )
                    openai_boxed = r.boxed
                    match = 1 if eq(openai_boxed, expected) else 0
                    record = dict(item)
                    record.update(
                        {
                            "openai_model": args.model,
                            "openai_boxed_answer": openai_boxed,
                            "openai_raw_text": r.raw_text,
                            "openai_match": match,
                        }
                    )
                    processed.append(record)
                    jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    jsonl_f.flush()
                    last_err = None
                    break
                except Exception as e:
                    last_err = f"{type(e).__name__}: {e}"
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, 30.0)

            if last_err is not None:
                record = dict(item)
                record.update(
                    {
                        "openai_model": args.model,
                        "openai_boxed_answer": "nan",
                        "openai_raw_text": "",
                        "openai_match": 0,
                        "openai_error": last_err,
                    }
                )
                processed.append(record)
                jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl_f.flush()

            if args.sleep_s and args.sleep_s > 0:
                time.sleep(args.sleep_s)

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx+1}/{len(items)}")
    finally:
        jsonl_f.close()

    out_payload = {
        "source_file": str(in_path),
        "openai_model": args.model,
        "results": processed,
    }
    out_json.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote JSON: {out_json}")
    print(f"Wrote JSONL: {out_jsonl}")


if __name__ == "__main__":
    main()

