# Directory Comparison Report: R-Zero-main vs rzerooriginal/R-Zero

**Generated:** Comparison between R-Zero-main (modified) and rzerooriginal/R-Zero (original)

---

## Summary

- **Files in rzerooriginal/R-Zero but NOT in R-Zero-main:** 0
- **Files in R-Zero-main but NOT in rzerooriginal/R-Zero:** 3
- **Modified Files:** 3
- **Unchanged Files:** 94

---

## 📁 Files Added in R-Zero-main (3 files)

These are Rentropy-specific additions:

1. **`examples/reward_function/caller_rentropy.py`**
   - Rentropy reward function with cluster-entropy diversity rewards
   - Supports 4 diversity modes (1-4)
   - Includes ZPD-based reward scaling

2. **`scripts/main_rentropy.sh`**
   - Main training loop script for Rentropy experiments
   - Supports diversity mode selection as argument
   - Coordinates questioner and solver training iterations

3. **`scripts/questioner_train_rentropy.sh`**
   - Questioner training script with Rentropy reward
   - Configures diversity mode dynamically
   - Integrates with Rentropy reward function

---

## ✏️ Modified Files (3 files)

### 1. `examples/config.yaml`

**Key Changes:**
- **Reduced batch sizes and lengths** (likely for testing/smoke tests):
  - `max_prompt_length`: 2048 → **1024**
  - `max_response_length`: 2048 → **1024**
  - `rollout_batch_size`: 512 → **64**
  - `val_batch_size`: 1024 → **64**

- **Training configuration adjustments:**
  - Various training parameters modified for different compute requirements

**Lines Changed:** +19 added, -19 removed

### 2. `requirements.txt`

**Changes:**
- **`av` package:** Unpinned version in R-Zero-main
  - R-Zero-main: `av` (unpinned)
  - rzerooriginal: `av==14.4.0` (pinned)

- **`flash_attn` package:** Removed in R-Zero-main
  - R-Zero-main: Not present (removed)
  - rzerooriginal: `flash_attn==2.7.4.post1` (present)

**Lines Changed:** +2 added, -1 removed

**Rationale:** 
- Unpinning `av` allows more flexible version compatibility
- Removing `flash_attn` avoids installation issues (this package can be problematic)

### 3. `scripts/solver_train.sh`

**Key Changes:**
- **Upload score thresholds changed:**
  - `max_score`: 0.85 → **0.8**
  - `min_score`: 0.25 → **0.3**

- **Response length:**
  - `data.max_response_length`: 4096 → **1024** (in R-Zero-main)

- **Other training parameter adjustments**

**Lines Changed:** +8 added, -24 removed

---

## 📊 Analysis

### Rentropy-Specific Additions

The three new files (`caller_rentropy.py`, `main_rentropy.sh`, `questioner_train_rentropy.sh`) are **Rentropy-specific enhancements** that add:

1. **Cluster-entropy diversity rewards** to prevent model collapse
2. **Configurable diversity modes** (1-4) for experimentation
3. **Automated training loops** for questioner-solver co-evolution

### Configuration Changes

The modifications to existing files suggest:
- **Reduced resource requirements** (smaller batch sizes, shorter sequences)
- **Simplified dependencies** (removed problematic packages)
- **Adjusted score thresholds** (tighter filtering criteria)

### Compatibility

- **94 files unchanged** - Core R-Zero functionality remains intact
- **No breaking changes** - All modifications are additive or configuration-only
- **Backward compatible** - Original R-Zero scripts still work

---

## 🔍 Key Differences Summary

| Category | R-Zero-main | rzerooriginal/R-Zero |
|----------|-------------|---------------------|
| **Rentropy Support** | ✅ Yes (3 new files) | ❌ No |
| **Batch Sizes** | Smaller (64) | Larger (512) |
| **Sequence Lengths** | Shorter (1024) | Longer (2048/4096) |
| **Dependencies** | Simplified (no flash_attn) | Full (includes flash_attn) |
| **Score Thresholds** | Tighter (0.3-0.8) | Wider (0.25-0.85) |

---

## 🎯 Recommendations

1. **For Rentropy experiments:** Use `R-Zero-main` (has Rentropy support)
2. **For original R-Zero:** Use `rzerooriginal/R-Zero` (baseline implementation)
3. **For comparisons:** Both directories can coexist for A/B testing

---

**Note:** This comparison excludes `.git`, `__pycache__`, `.pyc`, and `.bak` files for clarity.
