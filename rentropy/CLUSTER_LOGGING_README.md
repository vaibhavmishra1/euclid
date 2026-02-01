# Rentropy Cluster Statistics Logging

## Overview

The Rentropy training pipeline now automatically logs cluster statistics during training, allowing you to analyze:
- Which clusters were visited most frequently
- Average majority voting scores per cluster
- Evolution of cluster probabilities over time
- Distribution of generated questions across the cluster space

## How It Works

### Automatic Logging During Training

The reward function (`caller_rentropy.py`) automatically logs cluster statistics to:
```
$STORAGE_PATH/cluster_logs/cluster_stats_step_XXXXXX.json
```

Each log file contains:
- **Step number**: Training step counter
- **Timestamp**: When the log was created
- **Cluster counts**: EMA-tracked visit frequencies for all clusters
- **Cluster probabilities**: Current probability distribution over clusters
- **Batch distribution**: Number of questions from each cluster in this batch
- **Batch scores**: Average majority voting score per cluster in this batch

### Configuration

Control logging frequency in `rentropy_config.yaml`:
```yaml
# Log cluster stats every N reward computations (default: 1)
log_cluster_stats_freq: 1
```

Set higher values (e.g., 10) to reduce logging overhead during long training runs.

## Analyzing Logs After Training

### Quick Analysis

Run the analysis script:
```bash
python analyze_cluster_logs.py --log_dir /path/to/storage/cluster_logs
```

### Save Summary to File

```bash
python analyze_cluster_logs.py \
    --log_dir /path/to/storage/cluster_logs \
    --output summary.json
```

### Example Output

```
================================================================================
RENTROPY CLUSTER STATISTICS SUMMARY
================================================================================

Training Steps: 100
Total Clusters: 128
Clusters Visited: 87
Clusters Never Visited: 41
Total Questions Generated: 51200

--------------------------------------------------------------------------------
TOP 10 MOST VISITED CLUSTERS
--------------------------------------------------------------------------------
Cluster ID   Questions    % of Total   Avg Score   
--------------------------------------------------------------------------------
42           2847         5.56         0.612
17           2341         4.57         0.581
89           1923         3.76         0.544
...

--------------------------------------------------------------------------------
TOP 10 HIGHEST SCORING CLUSTERS
--------------------------------------------------------------------------------
Cluster ID   Avg Score    Questions    % of Total  
--------------------------------------------------------------------------------
12           0.782        891          1.74
56           0.769        1245         2.43
...

--------------------------------------------------------------------------------
CLUSTER VISIT DISTRIBUTION
--------------------------------------------------------------------------------
Mean questions per cluster: 400.00
Std dev: 234.52
Min: 0
Max: 2847
Median: 312.50
```

## Log File Structure

Each `cluster_stats_step_XXXXXX.json` file contains:

```json
{
  "step": 42,
  "timestamp": 1706745600.123,
  "num_questions": 512,
  "cluster_stats": {
    "cluster_counts": [1.2, 0.8, 1.5, ...],
    "cluster_probabilities": [0.012, 0.008, 0.015, ...],
    "total_count": 100.0,
    "num_clusters": 128
  },
  "batch_cluster_distribution": {
    "0": 12,
    "5": 23,
    "17": 45,
    ...
  },
  "batch_cluster_avg_scores": {
    "0": 0.612,
    "5": 0.581,
    "17": 0.544,
    ...
  }
}
```

## Use Cases

### 1. Verify Diversity Training

Check if the model is exploring diverse clusters:
```bash
python analyze_cluster_logs.py --log_dir storage/cluster_logs
# Look at "Clusters Visited" vs "Total Clusters"
```

### 2. Identify Difficult Clusters

Find clusters with low majority voting scores:
```python
import json
with open('summary.json') as f:
    data = json.load(f)

low_score_clusters = [
    (cid, stats['avg_majority_vote_score']) 
    for cid, stats in data['per_cluster_stats'].items()
    if stats['total_questions'] > 10 and stats['avg_majority_vote_score'] < 0.4
]
```

### 3. Compare Different Modes

Train with different diversity modes and compare:
```bash
# Mode 1 (baseline)
python analyze_cluster_logs.py --log_dir storage_mode1/cluster_logs --output mode1.json

# Mode 4 (full Rentropy)
python analyze_cluster_logs.py --log_dir storage_mode4/cluster_logs --output mode4.json

# Compare clusters_visited, distribution statistics, etc.
```

### 4. Track Evolution Over Time

The summary includes time-series data in `evolution`:
```python
import matplotlib.pyplot as plt
import json

with open('summary.json') as f:
    data = json.load(f)

steps = data['evolution']['steps']
cluster_counts = data['evolution']['cluster_counts']

# Plot evolution of top 5 clusters
for cid in range(5):
    counts = [cc[cid] for cc in cluster_counts]
    plt.plot(steps, counts, label=f'Cluster {cid}')

plt.xlabel('Training Step')
plt.ylabel('Cluster Count (EMA)')
plt.legend()
plt.savefig('cluster_evolution.png')
```

## Performance Impact

- Logging happens **after** reward computation, not during
- Each log file is ~10-50 KB depending on number of clusters
- With `log_cluster_stats_freq=1` and 1000 steps: ~10-50 MB total
- Negligible runtime overhead (<0.1% per step)

## Files Created

```
$STORAGE_PATH/
└── cluster_logs/
    ├── cluster_stats_step_000001.json
    ├── cluster_stats_step_000002.json
    ├── cluster_stats_step_000003.json
    └── ...
```

## Troubleshooting

**No logs created:**
- Check that `diversity_mode > 1` (mode 1 has no cluster tracking)
- Verify `$STORAGE_PATH` is set correctly
- Check that centroids file exists and is loaded

**Too many log files:**
- Increase `log_cluster_stats_freq` in `rentropy_config.yaml`

**Out of disk space:**
- Set higher `log_cluster_stats_freq`
- Delete old logs after analysis

## Example Workflow

```bash
# 1. Train with diversity mode
bash scripts/main_rentropy.sh Qwen/Qwen3-4B-Base qwen3-4b 4

# 2. After training, analyze logs
python analyze_cluster_logs.py \
    --log_dir storage/cluster_logs \
    --output results_mode4.json

# 3. Compare with baseline (mode 1)
bash scripts/main_rentropy.sh Qwen/Qwen3-4B-Base qwen3-4b-baseline 1
python analyze_cluster_logs.py \
    --log_dir storage_baseline/cluster_logs \
    --output results_mode1.json

# 4. Compare summaries
diff results_mode1.json results_mode4.json
```

## Integration with Experiments

The logging is automatic - no changes needed to training scripts. Just run:
```bash
bash scripts/main_rentropy.sh <model> <name> <mode>
```

Logs will appear in `$STORAGE_PATH/cluster_logs/` automatically.
