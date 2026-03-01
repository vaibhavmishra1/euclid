# Single-server variant of caller_penalty.py for 1-GPU setups.
#
# It expects ONE local grading server running at http://127.0.0.1:5000/hello
# (see vllm_service_init/start_single.sh).

import json
import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import numpy as np
import regex as re
import requests
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from sklearn.cluster import AgglomerativeClustering

STORAGE_PATH = os.getenv("STORAGE_PATH", "./storage")

os.environ["NO_PROXY"] = "0.0.0.0,127.0.0.1"

# Where the grading server listens. Our `start_vllm_server.py` binds to 127.0.0.1.
GRADER_HOST = os.getenv("RZERO_GRADER_HOST", "127.0.0.1")
# Single request can take minutes (vLLM generate + grading), so default to a long timeout.
GRADER_TIMEOUT_S = int(os.getenv("RZERO_GRADER_TIMEOUT_S", "1800"))


def _bleu_distance_matrix(sentences):
    n = len(sentences)
    dist = np.zeros((n, n))
    smoother = SmoothingFunction().method1
    for i in range(n):
        for j in range(i, n):
            if i == j:
                score = 1.0
            else:
                ref = [sentences[j].split()]
                hyp = sentences[i].split()
                score = sentence_bleu(ref, hyp, smoothing_function=smoother)
            dist[i, j] = dist[j, i] = 1 - score
    return dist


def cluster_share_per_problem(problems, distance_threshold: float = 0.5, linkage: str = "average"):
    if not problems:
        return []
    dist_mat = _bleu_distance_matrix(problems)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage=linkage,
    )
    labels = clustering.fit_predict(dist_mat)
    total = len(problems)
    cluster_size = Counter(labels)
    cluster_ratio = {lab: sz / total for lab, sz in cluster_size.items()}
    proportions = [cluster_ratio[lab] for lab in labels]
    return proportions


def generate_temp_filename(prefix="temp", suffix=".json"):
    timestamp = int(time.time() * 1000)
    rand_part = random.randint(0, 99999)
    os.makedirs(f"{STORAGE_PATH}/temp_results", exist_ok=True)
    return f"{STORAGE_PATH}/temp_results/{prefix}_{timestamp}_{rand_part}{suffix}"


def _fetch(name: str, port: int = 5000) -> bool:
    url = f"http://{GRADER_HOST}:{port}/hello"
    # Use separate connect/read timeouts; connect should be quick, read can be long.
    resp = requests.get(url, params={"name": name}, timeout=(10, GRADER_TIMEOUT_S))
    return resp.status_code == 200


def generate_results(data):
    # Single shard: write one temp file, send one request, read back results.
    tmp = generate_temp_filename(prefix="temp_0", suffix=".json")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)

    ok = _fetch(tmp, port=5000)
    if not ok:
        raise RuntimeError("grader server request failed")

    out = tmp.replace(".json", "_results.json")
    with open(out, "r") as f:
        final_results = json.load(f)

    try:
        os.remove(out)
    except OSError:
        pass
    return final_results


def compute_score(
    predicts: List[str],
    ground_truths: List[str],
    format_weight: float = 0.1,
    file_path: str = "",
) -> List[Dict[str, float]]:
    # Extract <question> + boxed answer from the questioner output
    extracted = []
    for p in predicts:
        questions = re.findall(r"<question>(.*?)</question>", p, re.DOTALL)
        answers = re.findall(r"\\boxed\{(.*?)\}", p, re.DOTALL)
        if questions and answers:
            extracted.append({"question": questions[-1].strip(), "answer": answers[-1].strip()})
        else:
            extracted.append({"question": "", "answer": ""})

    final_results = generate_results(extracted)
    penalty = cluster_share_per_problem([r.get("question", "") for r in final_results], distance_threshold=0.5)
    assert len(penalty) == len(final_results)

    scores = []
    for r, pen in zip(final_results, penalty):
        # r["score"] is the solver's self-consistency score from the grading server
        if r.get("question"):
            base = min(r.get("score", 0.0), 1 - r.get("score", 0.0))
            final_score = base - pen
            scores.append({"overall": final_score, "format": 1.0, "accuracy": pen})
        else:
            scores.append({"overall": -1.0, "format": 0.0, "accuracy": 0.0})
    return scores

