#!/usr/bin/env python3
"""
Upload constructed clusters to Hugging Face as a dataset.

Usage:
    python upload_clusters.py \
        --cluster_dir ./cluster_data \
        --repo_name rentropy-clusters \
        --private

This will upload:
- centroids.npy (as a dataset feature)
- labels.npy (as a dataset feature)  
- cluster_stats.json (as metadata)
- README.md (dataset card)
"""
import argparse
import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List

try:
    from datasets import Dataset, DatasetDict, Features, Value, Array2D
    from huggingface_hub import login, HfApi, create_repo
    import huggingface_hub
except ImportError:
    print("Please install required packages:")
    print("  pip install datasets huggingface_hub")
    raise


def load_cluster_data(cluster_dir: str) -> Dict:
    """Load all cluster data files."""
    cluster_path = Path(cluster_dir)
    
    # Load centroids
    centroids_path = cluster_path / "centroids.npy"
    if not centroids_path.exists():
        raise FileNotFoundError(f"Centroids not found at {centroids_path}")
    centroids = np.load(centroids_path)
    print(f"Loaded centroids: shape {centroids.shape}")
    
    # Load labels (optional)
    labels_path = cluster_path / "labels.npy"
    labels = None
    if labels_path.exists():
        labels = np.load(labels_path)
        print(f"Loaded labels: shape {labels.shape}")
    
    # Load embeddings (new)
    embeddings_path = cluster_path / "embeddings.npy"
    embeddings = None
    if embeddings_path.exists():
        embeddings = np.load(embeddings_path)
        print(f"Loaded embeddings: shape {embeddings.shape}")
    
    # Load question metadata (new)
    metadata_path = cluster_path / "question_metadata.json"
    question_metadata = None
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            question_metadata = json.load(f)
        print(f"Loaded question metadata: {len(question_metadata)} questions")
    
    # Load stats
    stats_path = cluster_path / "cluster_stats.json"
    stats = {}
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        print(f"Loaded stats: {stats}")
    
    return {
        "centroids": centroids,
        "labels": labels,
        "embeddings": embeddings,
        "question_metadata": question_metadata,
        "stats": stats,
    }


def create_dataset_from_clusters(cluster_data: Dict) -> Dataset:
    """Convert cluster data to Hugging Face Dataset format."""
    centroids = cluster_data["centroids"]
    labels = cluster_data["labels"]
    embeddings = cluster_data.get("embeddings")
    question_metadata = cluster_data.get("question_metadata")
    stats = cluster_data["stats"]
    
    # Convert centroids to list of lists (HF datasets requirement)
    centroids_list = centroids.tolist()
    
    # Create dataset with centroids
    dataset_dict = {
        "centroid": centroids_list,
        "cluster_id": list(range(len(centroids_list))),
    }
    
    # Add question-level data if available
    if question_metadata is not None:
        # Extract fields from metadata
        dataset_dict["question_id"] = [item["question_id"] for item in question_metadata]
        dataset_dict["question"] = [item["question"] for item in question_metadata]
        dataset_dict["question_cluster_id"] = [item["cluster_id"] for item in question_metadata]
        dataset_dict["dataset_source"] = [item["dataset_source"] for item in question_metadata]
        dataset_dict["index"] = [item["index"] for item in question_metadata]
    
    # Add embeddings if available
    if embeddings is not None:
        # Convert embeddings to list of lists
        embeddings_list = embeddings.tolist()
        dataset_dict["embedding"] = embeddings_list
    
    # Add labels if available
    if labels is not None:
        dataset_dict["question_labels"] = labels.tolist()
    
    # Define features
    features = Features({
        "cluster_id": Value("int32"),
        "centroid": [Value("float32")],  # Variable length list
    })
    
    if question_metadata is not None:
        features["question_id"] = Value("string")
        features["question"] = Value("string")
        features["question_cluster_id"] = Value("int32")
        features["dataset_source"] = Value("string")
        features["index"] = Value("int32")
    
    if embeddings is not None:
        features["embedding"] = [Value("float32")]
    
    if labels is not None:
        features["question_labels"] = [Value("int32")]
    
    dataset = Dataset.from_dict(dataset_dict, features=features)
    
    # Add metadata
    if stats:
        dataset.info.description = f"Rentropy cluster centroids and question data. {stats.get('num_clusters', '?')} clusters, {stats.get('num_questions', '?')} questions, embedding_dim={stats.get('embedding_dim', '?')}"
    
    return dataset


def upload_to_hub(
    cluster_dir: str,
    repo_name: str,
    private: bool = True,
    token: str = None,
    organization: str = None,
):
    """Upload cluster data to Hugging Face Hub."""
    
    # Load cluster data
    print(f"Loading cluster data from {cluster_dir}...")
    cluster_data = load_cluster_data(cluster_dir)
    
    # Create dataset
    print("Creating Hugging Face dataset...")
    dataset = create_dataset_from_clusters(cluster_data)
    
    # Prepare repo name
    if organization:
        full_repo_name = f"{organization}/{repo_name}"
    else:
        # Try to get from environment or tokens.json
        hf_name = os.getenv("HUGGINGFACENAME")
        if hf_name:
            full_repo_name = f"{hf_name}/{repo_name}"
        else:
            full_repo_name = repo_name
    
    # Login
    if token:
        login(token=token)
    else:
        # Try to load from tokens.json (R-Zero style)
        tokens_file = Path(__file__).parent.parent / "R-Zero" / "tokens.json"
        if tokens_file.exists():
            with open(tokens_file, 'r') as f:
                tokens = json.load(f)
            login(token=tokens.get('huggingface'))
        else:
            print("No token provided and tokens.json not found. Attempting login...")
            login()  # Will prompt if needed
    
    # Create repo if it doesn't exist
    api = HfApi()
    try:
        api.repo_info(full_repo_name)
        print(f"Repo {full_repo_name} already exists")
    except:
        print(f"Creating repo {full_repo_name}...")
        create_repo(full_repo_name, private=private, repo_type="dataset")
    
    # Upload dataset
    print(f"Uploading dataset to {full_repo_name}...")
    dataset.push_to_hub(full_repo_name, private=private)
    
    # Also upload raw .npy files for direct access
    print("Uploading raw .npy files...")
    cluster_path = Path(cluster_dir)
    
    files_to_upload = [
        "centroids.npy",
        "labels.npy",
        "embeddings.npy",  # New
        "question_metadata.json",  # New
        "index_to_question_id.json",  # New
        "cluster_stats.json",
    ]
    
    for filename in files_to_upload:
        filepath = cluster_path / filename
        if filepath.exists():
            print(f"  Uploading {filename}...")
            api.upload_file(
                path_or_fileobj=str(filepath),
                path_in_repo=filename,
                repo_id=full_repo_name,
                repo_type="dataset",
            )
        else:
            print(f"  Skipping {filename} (not found)")
    
    print(f"✓ Successfully uploaded to https://huggingface.co/datasets/{full_repo_name}")
    print(f"\nTo load the dataset:")
    print(f"  from datasets import load_dataset")
    print(f"  dataset = load_dataset('{full_repo_name}')")
    print(f"\nTo load raw files:")
    print(f"  from huggingface_hub import hf_hub_download")
    print(f"  import numpy as np")
    print(f"  import json")
    print(f"  ")
    print(f"  # Centroids")
    print(f"  centroids_path = hf_hub_download('{full_repo_name}', 'centroids.npy', repo_type='dataset')")
    print(f"  centroids = np.load(centroids_path)")
    print(f"  ")
    print(f"  # Embeddings")
    print(f"  embeddings_path = hf_hub_download('{full_repo_name}', 'embeddings.npy', repo_type='dataset')")
    print(f"  embeddings = np.load(embeddings_path)")
    print(f"  ")
    print(f"  # Question metadata")
    print(f"  metadata_path = hf_hub_download('{full_repo_name}', 'question_metadata.json', repo_type='dataset')")
    print(f"  with open(metadata_path) as f:")
    print(f"      question_metadata = json.load(f)")


def create_readme(cluster_dir: str, output_path: str):
    """Create a README.md for the dataset."""
    stats_path = Path(cluster_dir) / "cluster_stats.json"
    stats = {}
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            stats = json.load(f)
    
    readme = f"""---
license: mit
task_categories:
- other
tags:
- mathematics
- clustering
- embeddings
- rentropy
---

# Rentropy Cluster Centroids

This dataset contains cluster centroids for the Rentropy diversity reward system.

## Dataset Details

- **Number of clusters**: {stats.get('num_clusters', 'N/A')}
- **Number of questions used**: {stats.get('num_questions', 'N/A')}
- **Embedding dimension**: {stats.get('embedding_dim', 'N/A')}
- **Embedding model**: Qwen/Qwen3-Embedding-0.6B

## Dataset Structure

The dataset contains:
- `centroid`: List of float32 values representing the cluster centroid
- `cluster_id`: Integer ID of the cluster (0 to num_clusters-1)
- `question_labels`: (Optional) Cluster assignments for training questions

## Usage

### Load as Hugging Face Dataset

```python
from datasets import load_dataset

dataset = load_dataset("your-org/rentropy-clusters")
centroids = [row["centroid"] for row in dataset["train"]]
```

### Load Raw .npy Files

```python
from huggingface_hub import hf_hub_download
import numpy as np

# Download centroids
centroids_path = hf_hub_download(
    "your-org/rentropy-clusters",
    "centroids.npy",
    repo_type="dataset"
)
centroids = np.load(centroids_path)

# Download labels (if available)
labels_path = hf_hub_download(
    "your-org/rentropy-clusters",
    "labels.npy",
    repo_type="dataset"
)
labels = np.load(labels_path)
```

### Use with Rentropy

```python
from cluster_space.cluster_assigner import ClusterAssigner

# Download and use
centroids_path = hf_hub_download("your-org/rentropy-clusters", "centroids.npy", repo_type="dataset")
assigner = ClusterAssigner(centroids_path=centroids_path)
```

## Citation

If you use this dataset, please cite the Rentropy paper (when available).

## License

MIT License
"""
    
    with open(output_path, 'w') as f:
        f.write(readme)
    print(f"Created README at {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Upload Rentropy clusters to Hugging Face")
    parser.add_argument("--cluster_dir", type=str, default="./cluster_data",
                       help="Directory containing cluster files")
    parser.add_argument("--repo_name", type=str, required=True,
                       help="Hugging Face dataset repo name (e.g., 'rentropy-clusters')")
    parser.add_argument("--private", action="store_true", default=True,
                       help="Make the dataset private")
    parser.add_argument("--public", action="store_true",
                       help="Make the dataset public (overrides --private)")
    parser.add_argument("--token", type=str, default=None,
                       help="Hugging Face token (or use tokens.json)")
    parser.add_argument("--org", type=str, default=None,
                       help="Organization name (optional)")
    parser.add_argument("--create_readme", action="store_true",
                       help="Create README.md in cluster_dir")
    
    args = parser.parse_args()
    
    # Handle public/private
    is_private = args.private and not args.public
    
    # Create README if requested
    if args.create_readme:
        readme_path = Path(args.cluster_dir) / "README.md"
        create_readme(args.cluster_dir, readme_path)
    
    # Upload
    upload_to_hub(
        cluster_dir=args.cluster_dir,
        repo_name=args.repo_name,
        private=is_private,
        token=args.token,
        organization=args.org,
    )


if __name__ == "__main__":
    main()
