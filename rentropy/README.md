# Rentropy: Cluster-Entropy Diversity Reward for R-Zero

This experiment adds a **cluster-entropy diversity reward** to the R-Zero framework to mitigate entropy collapse during self-evolving training.

## Overview

R-Zero suffers from entropy collapse: the Challenger converges to a narrow distribution of tasks over iterations. Rentropy adds an intrinsic reward that explicitly encourages visiting diverse regions of the task space (as approximated by unsupervised embedding clusters).

## Directory Structure

```
rentropy/
├── README.md                    # This file
├── rentropy_config.yaml         # Configuration for diversity reward
├── idea/
│   ├── ideav0.md               # Original idea
│   └── ideav1.md               # Detailed experiment setup
├── cluster_space/               # Cluster construction (separate from R-Zero)
│   ├── build_clusters.py       # Build clusters from corpus (run once)
│   ├── cluster_assigner.py     # Cluster assignment and count tracking
│   ├── config.py               # Cluster config
│   └── requirements.txt        # Dependencies
└── R-Zero/                      # Modified R-Zero codebase
    └── examples/reward_function/
        └── caller_rentropy.py  # Rentropy reward function
```

## Quick Start

### 1. Install Dependencies

```bash
# Install cluster_space dependencies
pip install -r cluster_space/requirements.txt

# Install R-Zero dependencies
pip install -r R-Zero/requirements.txt
```

### 2. Smoke Test (Recommended First)

Run a minimal smoke test to verify everything works before full training:

```bash
cd R-Zero

# Option A: Test reward computation only (fast, no GPU training)
python scripts/smoketest_reward_only.py

# Option B: Full smoke test with minimal training (requires GPU)
export STORAGE_PATH=/path/to/storage
bash scripts/smoketest_all_variants.sh Qwen/Qwen3-0.6B-Base
```

**Smoke test settings** (in `config_smoketest.yaml`):
- Model: `Qwen/Qwen3-0.6B-Base` (smallest)
- Training steps: 2 (instead of 6)
- Questions per batch: 100 (instead of 1000)
- Rollouts: 2 (instead of 4-5)
- Single GPU

### 3. Build Cluster Space (Run Once)

```bash
# Option A: From a single JSON file
python cluster_space/build_clusters.py \
    --corpus_file /path/to/questions.json \
    --output_dir cluster_space/cluster_data \
    --num_clusters 128

# Option B: From a directory of JSON files
python cluster_space/build_clusters.py \
    --corpus_dir /path/to/corpus/ \
    --output_dir cluster_space/cluster_data \
    --num_clusters 128
```

This creates:
- `cluster_data/centroids.npy` - Cluster centroids (required)
- `cluster_data/labels.npy` - Cluster assignments (optional)
- `cluster_data/cluster_stats.json` - Metadata

### 3.5. Upload Clusters to Hugging Face (Optional)

Upload your clusters to share or use across environments:

```bash
# Upload to Hugging Face
python cluster_space/upload_clusters.py \
    --cluster_dir cluster_space/cluster_data \
    --repo_name rentropy-clusters \
    --private

# Or make it public
python cluster_space/upload_clusters.py \
    --cluster_dir cluster_space/cluster_data \
    --repo_name rentropy-clusters \
    --public
```

**To use clusters from Hugging Face:**

Update `rentropy_config.yaml`:
```yaml
centroids_path: "path/to/downloaded/centroids.npy"
```

Or download programmatically:
```python
from huggingface_hub import hf_hub_download
import numpy as np

centroids_path = hf_hub_download(
    "your-org/rentropy-clusters",
    "centroids.npy",
    repo_type="dataset"
)
centroids = np.load(centroids_path)
```

### 4. Configure Diversity Mode

Edit `rentropy_config.yaml`:

```yaml
# Diversity reward mode (1-4):
#   1: Vanilla majority voting only (R-Zero baseline)
#   2: Mode 1 + rarity reward (rare clusters get bonus)
#   3: Mode 2 + batch uniqueness (different from other n-1 questions)
#   4: Mode 3 + within-cluster uniqueness
diversity_mode: 2

# Reward weights
weights:
  rarity: 0.1
  batch_uniqueness: 0.05
  within_cluster_uniqueness: 0.05

# Only apply diversity reward if majority vote >= threshold
majority_vote_threshold: 0.3
```

### 5. Run Training

```bash
cd R-Zero

# Set environment variables
export STORAGE_PATH="/path/to/storage"
export HUGGINGFACENAME="your_hf_name"

# Run Rentropy training
bash scripts/main_rentropy.sh Qwen/Qwen3-4B-Base qwen3-4b-rentropy
```

## Diversity Reward Modes

| Mode | Components | Description |
|------|------------|-------------|
| 1 | Base only | Vanilla R-Zero (no diversity reward) |
| 2 | Base + Rarity | Bonus for generating questions in rare clusters |
| 3 | Mode 2 + Batch | Additional bonus for uniqueness within batch |
| 4 | Mode 3 + Within-cluster | Additional bonus for being far from cluster centroid |

### Reward Formula

For a question `q` with majority vote score `s`:

```
if s >= threshold:
    base_score = min(s, 1-s)  # ZPD-style, peaks at 0.5
    diversity = w_rarity * rarity(q) 
             + w_batch * batch_uniqueness(q)
             + w_within * within_cluster_uniqueness(q)
    final_score = base_score + diversity
else:
    final_score = base_score  # No diversity bonus for low-quality
```

## Evaluation Metrics

Track these metrics to measure diversity improvement:

1. **Cluster entropy**: `H(p_t)` over time (higher = less collapse)
2. **Unique clusters visited**: Per iteration
3. **Within-batch uniqueness**: Fraction of unique clusters per batch
4. **Top-k accuracy**: On held-out benchmark (should not degrade)

## Ablation Experiments

Run these conditions to isolate the effect of each component:

| Condition | Config Setting |
|-----------|----------------|
| B0 (baseline) | `diversity_mode: 1` |
| A2 (+ rarity) | `diversity_mode: 2` |
| A3 (+ batch) | `diversity_mode: 3` |
| A4 (full) | `diversity_mode: 4` |

## Files Modified in R-Zero

Only these files were added/modified:

1. `examples/reward_function/caller_rentropy.py` - New reward function
2. `examples/config_smoketest.yaml` - Minimal config for smoke testing
3. `scripts/questioner_train_rentropy.sh` - Training script using rentropy reward
4. `scripts/main_rentropy.sh` - Main loop using rentropy training
5. `scripts/smoketest_*.sh` - Smoke test scripts
6. `scripts/smoketest_reward_only.py` - Reward computation test

All original R-Zero files are unchanged.

## Smoke Test Details

The smoke test uses minimal resources to quickly verify all 4 variants work:

| Setting | Full Training | Smoke Test |
|---------|--------------|------------|
| Model | Qwen3-4B | Qwen3-0.6B |
| Training steps | 6 | 2 |
| Questions | 1000 | 100 |
| Rollouts | 4 | 2 |
| GPUs | 4-8 | 1 |
| Iterations | 5 | 1 |

Run smoke test first, then proceed to full training if successful.

## References

- R-Zero: https://github.com/Chengsong-Huang/R-Zero
- Paper: https://arxiv.org/abs/2508.05004
