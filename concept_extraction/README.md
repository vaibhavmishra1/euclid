## Math concept extraction scaffold

This folder contains a minimal pipeline to extract and normalize math concepts, theorems, formulae, and axioms/definitions from the MATH dataset (`hendrycks/competition_math`) using a strong open-weight model.

### Files
- `ontology.py` — ontology seed (`Domain:Subdomain:Concept` labels).
- `prompts.py` — prompt builders for extraction and normalization.
- `pipeline.py` — CLI scaffold that loads MATH, runs extraction + normalization, writes JSONL.

### Usage
Requires `datasets`, `transformers`, `torch` and access to strong open-weight models (e.g., Qwen2.5 72B, Mixtral 8x22B). Install:
```
pip install datasets transformers torch --upgrade
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
