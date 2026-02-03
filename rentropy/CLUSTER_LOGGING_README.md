# Rentropy Cluster Statistics Logging

## Overview

The Rentropy training pipeline now automatically logs cluster statistics during training **for all diversity modes (1-4)**, allowing you to analyze:
- Which clusters were visited most frequently
- Average majority voting scores per cluster
- Evolution of cluster probabilities over time
- Distribution of generated questions across the cluster space

**Note:** Even in **mode 1 (baseline)**, cluster statistics are logged for comparison purposes, but diversity rewards are **not** applied to the training signal. This allows direct comparison of question distribution between baseline and diversity-enhanced modes.

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

### 3. Compare Different Modes (Mode 1 vs Mode 4)

**Important:** Mode 1 tracks cluster statistics but doesn't use them for rewards, allowing direct comparison.

Train with different diversity modes and compare:
```bash
# Mode 1 (baseline - no diversity reward, only tracking)
bash scripts/main_rentropy.sh Qwen/Qwen3-4B-Base baseline 1
python analyze_cluster_logs.py --log_dir storage/cluster_logs --output mode1.json

# Mode 4 (full Rentropy - diversity rewards active)
bash scripts/main_rentropy.sh Qwen/Qwen3-4B-Base rentropy 4
python analyze_cluster_logs.py --log_dir storage/cluster_logs --output mode4.json

# Compare results
python3 << 'EOF'
import json

with open('mode1.json') as f:
    m1 = json.load(f)
with open('mode4.json') as f:
    m4 = json.load(f)

print(f"Mode 1 - Clusters visited: {m1['clusters_visited']}/{m1['num_clusters']}")
print(f"Mode 4 - Clusters visited: {m4['clusters_visited']}/{m4['num_clusters']}")

print(f"\nMode 1 - Std dev of visits: {m1['cluster_visit_std']:.2f}")
print(f"Mode 4 - Std dev of visits: {m4['cluster_visit_std']:.2f}")

print("\nMode 1 collapsed to fewer clusters!" if m1['clusters_visited'] < m4['clusters_visited'] else "\nMode 4 achieved better diversity!")
EOF
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
- Verify `$STORAGE_PATH` is set correctly
- Check that centroids file exists and is loaded (required for all modes)
- Ensure `rentropy_config.yaml` has correct `centroids_path`

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



iant4clusters
Loading logs from: /workspace/euclid/rentropy/variant4clusters
Found 6 log files

================================================================================
RENTROPY CLUSTER STATISTICS SUMMARY
================================================================================

Training Steps: 6
Total Clusters: 1024
Clusters Visited: 566
Clusters Never Visited: 458
Total Questions Generated: 6928

--------------------------------------------------------------------------------
TOP 10 MOST VISITED CLUSTERS
--------------------------------------------------------------------------------
Cluster ID   Questions    % of Total   Avg Score   
--------------------------------------------------------------------------------
238          356          5.14         0.502       
713          177          2.55         0.551       
348          156          2.25         0.613       
9            149          2.15         0.468       
994          136          1.96         0.473       
722          117          1.69         0.421       
29           115          1.66         0.462       
672          99           1.43         0.567       
1013         99           1.43         0.447       
316          94           1.36         0.538       

--------------------------------------------------------------------------------
TOP 10 HIGHEST SCORING CLUSTERS
--------------------------------------------------------------------------------
Cluster ID   Avg Score    Questions    % of Total  
--------------------------------------------------------------------------------
309          1.000        1            0.01        
154          0.900        1            0.01        
834          0.900        1            0.01        
280          0.900        2            0.03        
824          0.900        1            0.01        
178          0.833        3            0.04        
504          0.800        2            0.03        
904          0.800        1            0.01        
76           0.800        1            0.01        
639          0.800        1            0.01        

--------------------------------------------------------------------------------
CLUSTER VISIT DISTRIBUTION
--------------------------------------------------------------------------------
Mean questions per cluster: 6.77
Std dev: 20.28
Min: 0
Max: 356
Median: 1.00



Loading logs from: /workspace/euclid/rentropy/variant1clusters/cluster_logs
Found 6 log files

================================================================================
RENTROPY CLUSTER STATISTICS SUMMARY
================================================================================

Training Steps: 6
Total Clusters: 1024
Clusters Visited: 545
Clusters Never Visited: 479
Total Questions Generated: 7128

--------------------------------------------------------------------------------
TOP 10 MOST VISITED CLUSTERS
--------------------------------------------------------------------------------
Cluster ID   Questions    % of Total   Avg Score   
--------------------------------------------------------------------------------
238          335          4.70         0.497       
29           181          2.54         0.430       
9            150          2.10         0.493       
1013         142          1.99         0.506       
411          125          1.75         0.544       
303          120          1.68         0.514       
527          116          1.63         0.541       
713          116          1.63         0.618       
434          114          1.60         0.499       
994          113          1.59         0.438       

--------------------------------------------------------------------------------
TOP 10 HIGHEST SCORING CLUSTERS
--------------------------------------------------------------------------------
Cluster ID   Avg Score    Questions    % of Total  
--------------------------------------------------------------------------------
189          1.000        1            0.01        
24           1.000        1            0.01        
388          1.000        1            0.01        
326          1.000        1            0.01        
541          1.000        1            0.01        
776          0.900        1            0.01        
905          0.900        1            0.01        
220          0.900        1            0.01        
718          0.900        1            0.01        
495          0.900        1            0.01        

--------------------------------------------------------------------------------
CLUSTER VISIT DISTRIBUTION
--------------------------------------------------------------------------------
Mean questions per cluster: 6.96
Std dev: 20.64
Min: 0
Max: 335
Median: 1.00
