import argparse
import json
import os

from datasets import Dataset, DatasetDict
from huggingface_hub import login


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_name", type=str, default="")
    parser.add_argument("--max_score", type=float, default=0.7)
    parser.add_argument("--min_score", type=float, default=0.3)
    parser.add_argument("--experiment_name", type=str, required=True)
    args = parser.parse_args()

    storage_path = os.getenv("STORAGE_PATH")
    hf_name = os.getenv("HUGGINGFACENAME")
    if not storage_path:
        raise RuntimeError("STORAGE_PATH is not set")
    if not hf_name:
        raise RuntimeError("HUGGINGFACENAME is not set")

    with open("tokens.json", "r") as f:
        token = json.load(f)["huggingface"]
    login(token=token)

    # Single shard only: suffix 0
    path = f"{storage_path}/generated_question/{args.experiment_name}_0_results.json"
    with open(path, "r") as f:
        datas = json.load(f)
    try:
        os.remove(path)
    except OSError:
        pass

    if not args.repo_name:
        print("repo_name is empty; skipping push_to_hub.")
        return

    filtered = [
        {"problem": d["question"], "answer": d["answer"], "score": d["score"]}
        for d in datas
        if d.get("score") is not None
        and d.get("question")
        and d.get("answer")
        and d["answer"] != "None"
        and args.min_score <= float(d["score"]) <= args.max_score
    ]
    print(f"upload_1gpu: keeping {len(filtered)} / {len(datas)} examples")

    train_dataset = Dataset.from_list(filtered)
    dataset = DatasetDict({"train": train_dataset})
    dataset.push_to_hub(f"{hf_name}/{args.repo_name}", private=True, config_name=args.experiment_name)


if __name__ == "__main__":
    main()

