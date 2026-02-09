import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download

# Load questions dataset
print("Loading questions dataset...")
questions_ds = load_dataset("vibhuiitj/variant2-iter4_solver_v1", split="train")
print("Dataset keys:", questions_ds.features.keys())

# Load centroids from huggingface hub
print("Loading centroids...")
centroids_path = hf_hub_download(repo_id="vibhuiitj/math_clusters_new", filename="centroids.npy", repo_type="dataset")
centroids = np.load(centroids_path)
print(f"Loaded {len(centroids)} cluster centroids")

# Initialize embedding model
print("Loading embedding model...")
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

# Extract questions
questions = [item['problem'] for item in questions_ds]
print(f"Total questions: {len(questions)}")

# Generate embeddings
print("Generating embeddings...")
embeddings = model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)
print(f"Generated embeddings shape: {embeddings.shape}")

# Find cluster assignments using cosine similarity
print("Assigning clusters...")
similarities = cosine_similarity(embeddings, centroids)
cluster_assignments = np.argmax(similarities, axis=1)

# Calculate frequency for each cluster
print("Calculating cluster frequencies...")
num_clusters = len(centroids)
cluster_freq = np.bincount(cluster_assignments, minlength=num_clusters)

# Save frequency array
output_file = "cluster_frequencies.npy"
np.save(output_file, cluster_freq)
print(f"Saved cluster frequencies to {output_file}")
print(f"Cluster frequency distribution: min={cluster_freq.min()}, max={cluster_freq.max()}, mean={cluster_freq.mean():.2f}")
