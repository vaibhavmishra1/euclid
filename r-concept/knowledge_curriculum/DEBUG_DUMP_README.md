# Debug Dump Feature

This feature allows you to dump all training data for analysis:
- **Prompts sent to challenger**: All formatted prompts with knowledge points and difficulty
- **Challenger outputs**: All raw outputs from the challenger model
- **Solver responses**: All solutions and uncertainty scores from the solver

## Usage

### Enable Debug Dumping

Set the environment variable `DUMP_DEBUG_DATA=1` before running training:

```bash
export DUMP_DEBUG_DATA=1
bash scripts/kp_main_minimal.sh ...
```

Or set it inline:

```bash
DUMP_DEBUG_DATA=1 bash scripts/kp_main_minimal.sh ...
```

### Disable Debug Dumping

Leave `DUMP_DEBUG_DATA` unset or set it to `0`:

```bash
export DUMP_DEBUG_DATA=0
# or simply don't set it
bash scripts/kp_main_minimal.sh ...
```

## Output Files

All files are saved to `${STORAGE_PATH}/debug_dumps/` (default: `/workspace/rzero_storage/debug_dumps/`)

### File Naming Convention

- `iter_{N}_step_{M}_prompts.jsonl` - Prompts for iteration N, step M
- `iter_{N}_step_{M}_challenger_outputs.jsonl` - Challenger outputs for iteration N, step M
- `iter_{N}_step_{M}_solver_responses.jsonl` - Solver responses for iteration N, step M

If iteration/step is not available, files use timestamp:
- `{timestamp}_prompts.jsonl`
- `{timestamp}_challenger_outputs.jsonl`
- `{timestamp}_solver_responses.jsonl`

### File Format

All files are JSONL (one JSON object per line):

**prompts.jsonl:**
```json
{
  "timestamp": "2026-01-13T05:42:34.123456",
  "prompt": "You are an expert...",
  "index": 0,
  "knowledge_points": ["KP1", "KP2"],
  "difficulty": 4,
  "set_id": 0
}
```

**challenger_outputs.jsonl:**
```json
{
  "timestamp": "2026-01-13T05:42:35.123456",
  "raw_output": "<question>...</question>\\n\\boxed{answer}",
  "index": 0,
  "raw_output": "..."
}
```

**solver_responses.jsonl:**
```json
{
  "timestamp": "2026-01-13T05:42:40.123456",
  "question": "What is...",
  "solver_answer": "42",
  "uncertainty_score": 0.5,
  "all_attempts": ["42", "42", "43", "42", "42"],
  "num_attempts": 5
}
```

## Example

```bash
# Enable debug dumping
export DUMP_DEBUG_DATA=1
export STORAGE_PATH="/workspace/rzero_storage"

# Run training
bash scripts/kp_main_minimal.sh \
    "Qwen/Qwen2.5-Math-1.5B-Instruct" \
    "kp_number_theory" \
    "/workspace/math_concepts_openai_all_number_theory.jsonl" \
    1 \
    50

# Check output
ls -lh ${STORAGE_PATH}/debug_dumps/
# iter_1_step_1_prompts.jsonl
# iter_1_step_1_challenger_outputs.jsonl
# iter_1_step_1_solver_responses.jsonl
# iter_1_step_2_prompts.jsonl
# ...
```

## Performance Impact

When enabled, debug dumping:
- Adds minimal overhead (file I/O only)
- Uses thread-safe writing (no blocking)
- Files are appended incrementally (memory efficient)

The feature is designed to have negligible impact on training performance.
