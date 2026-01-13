# R-Concept: Knowledge-Set Curriculum Learning Setup Guide

Complete setup instructions for training the Knowledge-Set Curriculum Learning system.

---

## Table of Contents
1. [Hardware Requirements](#hardware-requirements)
2. [VM/Cloud Setup](#vmcloud-setup)
3. [System Dependencies](#system-dependencies)
4. [Python Environment Setup](#python-environment-setup)
5. [Data Preparation](#data-preparation)
6. [Environment Variables](#environment-variables)
7. [Running Training](#running-training)
8. [Troubleshooting](#troubleshooting)

---

## Hardware Requirements

### Minimum (for `kp_main_minimal.sh`)
- **GPU**: 1x NVIDIA GPU with 48GB VRAM (e.g., A6000, A100-40GB works with reduced batch)
- **RAM**: 64GB system RAM
- **Storage**: 100GB free disk space
- **CPU**: 8+ cores

### Recommended (for full training)
- **GPU**: 2x NVIDIA GPUs with 48GB+ VRAM each (e.g., 2x A6000, 2x A100)
- **RAM**: 128GB+ system RAM
- **Storage**: 500GB+ SSD
- **CPU**: 16+ cores

---

## VM/Cloud Setup

### Option A: Lambda Labs / RunPod / Vast.ai
```bash
# Select instance with:
# - 1-2x A6000 (48GB) or A100 (40GB/80GB)
# - Ubuntu 22.04
# - CUDA 12.x pre-installed
```

### Option B: AWS EC2
```bash
# Instance type: p4d.24xlarge (8x A100) or g5.12xlarge (4x A10G)
# AMI: Deep Learning AMI (Ubuntu 22.04) with CUDA 12.x
```

### Option C: Google Cloud
```bash
# Machine type: a2-highgpu-1g (1x A100) or a2-highgpu-2g (2x A100)
# Image: Deep Learning VM with CUDA 12.x
```

### Verify GPU Setup
```bash
# Check NVIDIA driver and CUDA
nvidia-smi

# Expected output should show:
# - Driver Version: 535.x or higher
# - CUDA Version: 12.x
# - GPU(s) with 40GB+ memory
```

---

## System Dependencies

```bash
# Update system
apt-get update && apt-get upgrade -y

# Install required system packages
apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    ninja-build

# Verify Python version (must be 3.10.x)
python3 --version
```

---

## Python Environment Setup

### Step 1: Clone Repository
```bash
cd /root
git clone <your-repo-url> euclid
cd euclid/r-concept
```

### Step 2: Create Virtual Environment
```bash
# Create venv
python3 -m venv .venv

# Activate venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools
```

### Step 3: Install PyTorch with CUDA 12
```bash
# Install PyTorch 2.7+ with CUDA 12.x support
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu124
```

### Step 4: Install Flash Attention (Required)
```bash
# Flash Attention 2 - speeds up training significantly
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

### Step 5: Install All Dependencies
```bash
# Install from requirements.txt
pip install -r requirements.txt

# Key packages being installed:
# - vllm==0.9.1 (inference engine)
# - transformers==4.52.4 (model loading)
# - ray==2.46.0 (distributed training)
# - datasets==3.6.0 (data handling)
# - accelerate==1.7.0 (training optimization)
```

### Step 6: Verify Installation
```bash
# Test PyTorch CUDA
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Test vLLM
python3 -c "import vllm; print(f'vLLM: {vllm.__version__}')"

# Test transformers
python3 -c "from transformers import AutoModelForCausalLM; print('Transformers OK')"
```

---

## Data Preparation

### Step 1: Create Storage Directory
```bash
mkdir -p /root/rzero_storage/{models,generated_question,ks_state,datasets,temp_results}
```

### Step 2: Prepare Knowledge Points File
The training requires a JSONL file with knowledge points. Each line should be:
```json
{"knowledge_points": ["concept1", "concept2"], "problem_level": 3, "problem_category": "Number Theory"}
```

Example file location: `/root/math_concepts_openai_all_number_theory.jsonl`

### Step 3: (Optional) HuggingFace Token
Create `tokens.json` in the project root for uploading datasets:
```json
{
    "huggingface": "hf_your_token_here"
}
```

---

## Environment Variables

Set these before running training:

```bash
# Required
export STORAGE_PATH="/root/rzero_storage"
export HUGGINGFACENAME="local_user"  # or your HuggingFace username

# Recommended (prevents vLLM cache issues)
export VLLM_DISABLE_COMPILE_CACHE=1

```

---

## Running Training

### Pre-flight Checks
```bash
# 1. Ensure no GPU processes are running
nvidia-smi

# 2. If GPU memory is used by zombie processes, reboot or reset GPU
# nvidia-smi --gpu-reset -i 0  # Requires elevated privileges

# 3. Kill any leftover processes
pkill -9 -f ray 2>/dev/null
pkill -9 -f vllm 2>/dev/null
pkill -9 -f python 2>/dev/null
sleep 5
```

### Run Minimal Training (Recommended First)
```bash
cd /root/euclid/r-concept

# Activate environment
source .venv/bin/activate

# Set environment variables
export STORAGE_PATH="/root/rzero_storage"
export HUGGINGFACENAME="local_user"
export VLLM_DISABLE_COMPILE_CACHE=1

# Clean previous runs (optional)
rm -rf /root/rzero_storage/models/kp_number_theory_* \
       /root/rzero_storage/generated_question/kp_number_theory_* \
       /root/rzero_storage/ks_state/kp_number_theory_* 2>/dev/null

# Run minimal training
bash scripts/kp_main_minimal.sh \
    "Qwen/Qwen2.5-Math-1.5B-Instruct" \
    "kp_number_theory" \
    "/root/math_concepts_openai_all_number_theory.jsonl" \
    1 \
    50
```

### Command Parameters
```
bash scripts/kp_main_minimal.sh <base_model> <model_abbr> <knowledge_points_path> [num_iterations] [num_sets]

Arguments:
  base_model           - HuggingFace model ID (e.g., "Qwen/Qwen2.5-Math-1.5B-Instruct")
  model_abbr           - Short name for saving (e.g., "kp_number_theory")
  knowledge_points_path - Path to JSONL file with knowledge points
  num_iterations       - Training iterations (default: 1)
  num_sets             - Number of knowledge sets to use (default: 50)
```

### Full Training (After Minimal Works)
```bash
bash scripts/kp_main.sh \
    "Qwen/Qwen2.5-Math-1.5B-Instruct" \
    "kp_number_theory_full" \
    "/root/math_concepts_openai_all_number_theory.jsonl" \
    3
```

---

## Troubleshooting

#
### Issue: Dataset Not Found on HuggingFace
```
DatasetNotFoundError: Dataset 'local_user/xxx' doesn't exist
```
```bash
# Solution 1: Login to HuggingFace
huggingface-cli login

# Solution 2: Create tokens.json with valid token
echo '{"huggingface": "hf_your_token"}' > tokens.json
```

### Issue: CUDA/NCCL Errors
```bash
# Set NCCL environment variables
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
```

### Issue: Ray Cluster Issues
```bash
# Stop existing Ray cluster
ray stop --force

# Clear Ray temp files
rm -rf /tmp/ray/*

# Restart training
```

---

## Quick Start Script

Copy and run this entire block for a fresh setup:

```bash
#!/bin/bash
set -e

# Navigate to project
cd /root/euclid/r-concept

# Clean GPU state
pkill -9 -f ray 2>/dev/null || true
pkill -9 -f vllm 2>/dev/null || true
pkill -9 -f python 2>/dev/null || true
sleep 5

# Activate environment
source .venv/bin/activate

# Set environment
export STORAGE_PATH="/root/rzero_storage"
export HUGGINGFACENAME="local_user"
export VLLM_DISABLE_COMPILE_CACHE=1
# Create directories
mkdir -p $STORAGE_PATH/{models,generated_question,ks_state,datasets,temp_results}

# Clean previous artifacts
rm -rf $STORAGE_PATH/models/kp_number_theory_* \
       $STORAGE_PATH/generated_question/kp_number_theory_* \
       $STORAGE_PATH/ks_state/kp_number_theory_* 2>/dev/null || true

# Run training
bash scripts/kp_main_minimal.sh \
    "Qwen/Qwen2.5-Math-1.5B-Instruct" \
    "kp_number_theory" \
    "/root/math_concepts_openai_all_number_theory.jsonl" \
    1 \
    50

echo "Training complete!"
```

---

## Output Locations

After successful training:
- **Challenger Model**: `$STORAGE_PATH/models/kp_number_theory_challenger_v1_minimal/`
- **Solver Model**: `$STORAGE_PATH/models/kp_number_theory_solver_v1_minimal/`
- **Generated Questions**: `$STORAGE_PATH/generated_question/`
- **Knowledge Set State**: `$STORAGE_PATH/ks_state/`

---

## Next Steps After Training

1. **Evaluate the trained solver**:
   ```bash
   bash evaluation/evaluate.bash $STORAGE_PATH/models/kp_number_theory_solver_v1_minimal/global_step_5/actor/huggingface
   ```

2. **Run full training with more iterations**:
   ```bash
   bash scripts/kp_main.sh "Qwen/Qwen2.5-Math-1.5B-Instruct" "kp_full" "/path/to/knowledge_points.jsonl" 3
   ```

---

## Support

If issues persist:
1. Check GPU memory: `nvidia-smi`
2. Check logs in terminal output
3. Verify all environment variables are set
4. Ensure knowledge points file exists and is valid JSONL
