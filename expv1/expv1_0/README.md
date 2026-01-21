### expv1_0 — Method 1 (offline) pipeline

This folder contains the **first runnable implementation** of the Method‑1 experiment described in:
- `tree/euclid/expv1/expv1_formal_planner/expv1_0.md`
- `tree/euclid/expv1/expv1_formal_planner/research_proposal_v0.md`

Goal (ExpV1_0):
- Build a synthetic dataset using **GRIP-style concept graph exploration**
- Keep **concept extraction + graph building** separate from **question generation + solver filtering**
- Add **dedup** and **solver-aware ZPD/self-consistency filtering**

This implementation is intentionally modular:
- Concept extraction, generator, teacher, solver inference, and dedup can be swapped between backends (local HF, vLLM, API).

#### Quickstart (high level)
1) Configure models and budgets in `config.yaml`

2) Run **Pipeline A** (concept extraction + KCRG artifacts):

```bash
python -m tree.euclid.expv1.expv1_0.run_build_concepts --config tree/euclid/expv1/expv1_0/config.yaml
```

3) **(Optional)** Run **Concept Cleaning** to deduplicate concepts using pairwise LLM comparison:

```bash
# Uses vLLM batched inference to compare all ~500k concept pairs
python -m tree.euclid.expv1.expv1_0.run_clean_concepts \
    --config tree/euclid/expv1/expv1_0/config.yaml \
    --rebuild-graph

# With custom model or batch size:
python -m tree.euclid.expv1.expv1_0.run_clean_concepts \
    --config tree/euclid/expv1/expv1_0/config.yaml \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --batch-size 128 \
    --rebuild-graph
```

4) Run **Pipeline B** (generation + solver filtering):

```bash
python -m tree.euclid.expv1.expv1_0.run_generate_dataset --config tree/euclid/expv1/expv1_0/config.yaml
```

5) Or run **end-to-end** (A then B):

```bash
python -m tree.euclid.expv1.expv1_0.run_build_dataset --config tree/euclid/expv1/expv1_0/config.yaml
```

6) Run SFT training (optional; heavy):

```bash
python -m tree.euclid.expv1.expv1_0.run_sft --config tree/euclid/expv1/expv1_0/config.yaml
```

7) Run evaluation:

```bash
python -m tree.euclid.expv1.expv1_0.run_eval --config tree/euclid/expv1/expv1_0/config.yaml
```

#### Concept Cleaning

The concept cleaning pipeline uses **pairwise LLM comparison** with vLLM batched inference:

1. Generates all unique pairs from 1080 concepts → ~500k pairs
2. For each pair, asks LLM: "Are these the same concept?" → YES/NO
3. Uses Union-Find to build clusters from matching pairs
4. Picks canonical name per cluster (highest count, shortest name)

**Time estimate**: ~17 minutes with 7B model on vLLM (batch_size=64)

**Output artifacts** (in `output/concepts/cleaned/`):
- `canonical_mapping.json` - original_key → canonical_key
- `canonical_vocab.json` - deduplicated vocab with merged counts
- `pairwise_matches.json` - pairs that LLM said are the same
- `clusters.json` - merged clusters with members

#### Notes
- This repo already contains an `expv0/exp0_0` baseline and the `R-Zero/` codebase. ExpV1_0 does **not** run co-evolution RL; it focuses on offline dataset construction + SFT for identifiability.
