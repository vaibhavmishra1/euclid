# exp0_0: Zero-Shot Baseline Experiment

**Experiment ID**: exp0_0  
**Type**: Baseline / No Training  
**Model**: Qwen/Qwen2.5-7B-Instruct  
**Dataset**: MATH-500  

## Objective

Run the complete CAQG pipeline **without any training** to establish baseline metrics for:
- Question generation quality (well-posedness, validity)
- Solvability rates (majority voting scores)  
- Novelty scores (semantic distance from seed)
- Iteration depth before termination
- Failure mode distribution

## Quick Start

```bash
# 1. Set environment
export STORAGE_PATH="/path/to/storage"
export HF_TOKEN="your_huggingface_token"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run baseline
bash run_baseline.sh

# 4. Analyze results
python analyze_results.py --results_dir results/
```

## Directory Structure

```
exp0_0/
├── README.md                 # This file
├── config.yaml               # Hyperparameters
├── requirements.txt          # Python dependencies
├── prompts/
│   ├── p1_generator.txt      # Question generator prompt
│   ├── p2_solver.txt         # Solver prompt
│   ├── p3_verifier.txt       # Answer verifier prompt
│   └── p4_novelty.txt        # Novelty verifier prompt
├── baseline_pipeline.py      # Main inference script
├── analyze_results.py        # Metrics computation
├── run_baseline.sh           # Shell wrapper
└── results/                  # Output (created at runtime)
```

## Configuration

See `config.yaml` for all hyperparameters:
- K = 4 questions per seed per iteration
- M = 8 solutions per question
- Max iterations = 5
- Solvability threshold = 0.5
- Novelty threshold = 0.4

## Expected Outputs

After running, `results/` will contain:
- `raw_outputs.json` - All generated Q/A pairs with scores
- `metrics_summary.json` - Aggregate statistics
- `iteration_traces.json` - Per-seed iteration details
- `failure_analysis.json` - Termination reasons breakdown

## Related Documents

- `../exp0_formal_planner/CAQG_research_plan.md` - Theoretical framework
- `../exp0_formal_planner/CAQG_research_plan_v2.md` - Full training pipeline
- `../exp0_formal_planner/exp0_0_baseline_plan.md` - Detailed experiment plan
