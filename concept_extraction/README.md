## Math concept extraction scaffold

This folder contains a minimal pipeline to extract and normalize math concepts, theorems, formulae, and axioms/definitions from the MATH dataset (`EleutherAI/hendrycks_math`) using either:
- a local open-weight model (Hugging Face `transformers`), or
- a closed-source API model (OpenAI).

### Files
- `ontology.py` — ontology seed (`Domain:Subdomain:Concept` labels).
- `prompts.py` — prompt builders for extraction and normalization.
- `pipeline.py` — CLI scaffold that loads MATH, runs extraction + normalization, writes JSONL.

### Usage
Install base deps:
```
pip install datasets --upgrade
```

#### Option A: OpenAI backend (recommended for one-time 10k run)
Install:
```
pip install openai --upgrade
```

Set your key:
```
export OPENAI_API_KEY="..."
```

Run:
```
python -m tree.euclid.concept_extraction.pipeline \
  --backend openai \
  --dataset EleutherAI/hendrycks_math \
  --extractor gpt-4o-mini \
  --normalizer gpt-4o-mini \
  --split train \
  --limit 100 \
  --output math_concepts.jsonl
```

#### Option B: Hugging Face backend (local open-weight)
Install:
```
pip install transformers torch --upgrade
```

Run (same model for both passes shown; you can split if desired):
```
python tree/euclid/concept_extraction/pipeline.py \
  --extractor <model_name_or_path> \
  --normalizer <model_name_or_path> \
  --split train \
  --limit 100 \
  --output math_concepts.jsonl
```

Output JSONL schema per row:
```
{
  "id": "<id-string>",
  "question": "<question text>",
  "concepts": ["<ontology or NEW:...>"],          // max 8, deduped
  "named_theorems": ["..."],
  "formulae": ["..."],
  "axioms_or_definitions": ["..."]
}
```

Notes:
- The pipeline is deterministic-leaning (temperature defaults to 0.1). Increase if the model under-extracts.
- `extract_first_json` is permissive; if a model wraps JSON in text, it will grab the first JSON object.
- MATH field `problem` is used as the question text. If the dataset changes, adjust in `run_single_example`.
- Extend or edit `ONTOLOGY` as you find new concepts; `NEW:` concepts will be preserved in outputs until you map them.
