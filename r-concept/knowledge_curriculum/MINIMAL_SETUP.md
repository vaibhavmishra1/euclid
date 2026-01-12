# Minimal Compute Setup for Testing

This guide shows how to reduce compute requirements to **prove the concept works** without full-scale training.

## 🎛️ Main Knobs to Reduce Compute

### 1. **Model Size** (Biggest Impact)
| Full | Minimal | Savings |
|------|---------|---------|
| Qwen/Qwen3-4B (4B params) | Qwen/Qwen2.5-Math-1.5B-Instruct (1.5B) | **~4x faster** |
| | Qwen/Qwen2-0.5B (500M) | **~8x faster** |
| | TinyLlama-1.1B | **~4x faster** |

**Recommendation**: Use `Qwen/Qwen2.5-Math-1.5B-Instruct` (math-specialized, better for this task) or `Qwen/Qwen2-1.5B` for testing.

### 2. **Number of Knowledge Sets** (Huge Impact)
| Full | Minimal | Savings |
|------|---------|---------|
| 868 sets | 50 sets | **~17x fewer questions** |
| | 20 sets | **~43x fewer questions** |
| | 10 sets | **~87x fewer questions** |

**How to do it**:
```bash
# Create subset
head -n 50 math_concepts_openai_all_number_theory.jsonl > kp_subset_50.jsonl
```

### 3. **Questions per Set** (Linear Impact)
| Full | Minimal | Savings |
|------|---------|---------|
| 5 questions/set | 1 question/set | **5x fewer questions** |
| | 2 questions/set | **2.5x fewer questions** |

**Total questions per iteration**:
- Full: 868 × 5 = **4,340 questions**
- Minimal: 50 × 1 = **50 questions** (87x reduction!)

### 4. **Training Steps** (Linear Impact)
| Full | Minimal | Savings |
|------|---------|---------|
| Challenger: 6 steps | 2 steps | **3x faster** |
| Solver: 15 steps | 5 steps | **3x faster** |

### 5. **Number of Iterations**
| Full | Minimal | Savings |
|------|---------|---------|
| 3 iterations | 1 iteration | **3x faster** |
| | 2 iterations | **1.5x faster** |

**For proof-of-concept**: 1 iteration is enough to show:
- ✅ Knowledge sets load correctly
- ✅ Questions generate with KPs + difficulty
- ✅ Rewards compute correctly
- ✅ Difficulties update based on thresholds

### 6. **GPU Usage**
| Full | Minimal | Savings |
|------|---------|---------|
| 4 GPUs (challenger) | 1 GPU | **4x fewer GPUs** |
| 8 GPUs (evaluation) | 1 GPU | **8x fewer GPUs** |

### 7. **Batch Sizes**
| Full | Minimal | Savings |
|------|---------|---------|
| Global batch: 16 | 4 | **4x less memory** |
| Rollout n: 4 | 2 | **2x fewer samples** |

### 8. **Evaluation Samples**
| Full | Minimal | Savings |
|------|---------|---------|
| 9 samples per question | 4 samples | **2.25x faster** |
| | 2 samples | **4.5x faster** |

## 📊 Compute Comparison

### Full Setup:
```
Model: 4B params
Knowledge Sets: 868
Questions/Set: 5
Total Questions: 4,340
Iterations: 3
GPUs: 4-8
Time Estimate: ~24-48 hours
```

### Minimal Setup (Proof-of-Concept):
```
Model: Qwen2.5-Math-1.5B-Instruct (1.5B, math-specialized, 4x faster)
Knowledge Sets: 50 (17x fewer)
Questions/Set: 1 (5x fewer)
Total Questions: 50 (87x fewer!)
Iterations: 1 (3x faster)
GPUs: 1 (4-8x fewer)
Time Estimate: ~2-4 hours
```

**Total Reduction: ~100-200x faster!**

## 🚀 Quick Start (Minimal)

```bash
# 1. Create subset
head -n 50 /Users/vaibhav/Desktop/brahma/math_concepts_openai_all_number_theory.jsonl > kp_subset_50.jsonl

# 2. Set environment
export STORAGE_PATH="/tmp/rzero_storage"
export HUGGINGFACENAME="your_username"

# 3. Run minimal training
cd /Users/vaibhav/Desktop/brahma/tree/R-Zero
bash scripts/kp_main_minimal.sh \
    Qwen/Qwen2.5-Math-1.5B-Instruct \
    test-minimal \
    kp_subset_50.jsonl \
    1 \
    50
```

## ✅ What This Proves

Even with minimal setup, you can demonstrate:

1. **Knowledge Set Loading**: ✅ Loads 50 sets with correct initial difficulties
2. **Question Generation**: ✅ Generates questions with all KPs + difficulty in prompt
3. **Reward Computation**: ✅ Computes R-Zero rewards correctly
4. **Difficulty Updates**: ✅ Updates difficulties based on alpha/beta thresholds
5. **State Persistence**: ✅ Saves/loads state correctly

## 📈 Scaling Up

Once you prove it works, scale up gradually:

1. **Phase 1** (Current): 50 sets, 1 question/set, 1 iteration
2. **Phase 2**: 100 sets, 2 questions/set, 2 iterations
3. **Phase 3**: 200 sets, 3 questions/set, 2 iterations
4. **Phase 4**: Full 868 sets, 5 questions/set, 3 iterations

## 🎯 Recommended Minimal Config

For fastest proof-of-concept:

```bash
# Use this exact command:
bash scripts/kp_main_minimal.sh \
    Qwen/Qwen2.5-Math-1.5B-Instruct \
    test-poc \
    kp_subset_20.jsonl \
    1 \
    20

# Or omit the model (uses default):
bash scripts/kp_main_minimal.sh \
    "" \
    test-poc \
    kp_subset_20.jsonl \
    1 \
    20
```

This gives you:
- 20 knowledge sets
- 20 questions total
- 1 iteration
- Single GPU
- ~1-2 hours runtime

**This is sufficient to prove the curriculum learning mechanism works!**
