#!/usr/bin/env python3
"""
Upload embeddings and other cluster data to Hugging Face Hub.

Usage:
    python upload_to_hf.py --repo_id your-username/your-repo-name --file embeddings_201027.npy
    
Or upload entire directory:
    python3 upload_to_hf.py --repo_id your-username/your-repo-name --folder ./cluster_data
"""
import argparse
from pathlib import Path
from huggingface_hub import HfApi, login, create_repo

def upload_file(repo_id: str, file_path: str, path_in_repo: str = None, token: str = None):
    """Upload a single file to Hugging Face Hub."""
    api = HfApi()
    
    # Login if token provided (otherwise uses cached token)
    if token:
        login(token=token)
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Use filename as path in repo if not specified
    if path_in_repo is None:
        path_in_repo = file_path.name
    
    print(f"Uploading {file_path} ({file_path.stat().st_size / 1e9:.2f} GB)...")
    print(f"To repo: {repo_id}")
    print(f"Path in repo: {path_in_repo}")
    
    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type="dataset", exist_ok=True)
        print(f"Repository {repo_id} ready")
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload file (automatically uses Git LFS for large files)
    api.upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
    )
    
    print(f"✓ Successfully uploaded to https://huggingface.co/datasets/{repo_id}")

def upload_folder(repo_id: str, folder_path: str, path_in_repo: str = ".", token: str = None):
    """Upload entire folder to Hugging Face Hub."""
    api = HfApi()
    
    # Login if token provided
    if token:
        login(token=token)
    
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    print(f"Uploading folder {folder_path}...")
    print(f"To repo: {repo_id}")
    
    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type="dataset", exist_ok=True)
        print(f"Repository {repo_id} ready")
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload folder
    api.upload_folder(
        folder_path=str(folder_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
    )
    
    print(f"✓ Successfully uploaded to https://huggingface.co/datasets/{repo_id}")

def main():
    parser = argparse.ArgumentParser(description="Upload files to Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, required=True,
                       help="Repository ID (username/repo-name)")
    parser.add_argument("--file", type=str,
                       help="Single file to upload")
    parser.add_argument("--folder", type=str,
                       help="Folder to upload")
    parser.add_argument("--path_in_repo", type=str, default=None,
                       help="Path in repo (default: same as filename/folder)")
    parser.add_argument("--token", type=str, default=None,
                       help="HF token (optional if already logged in)")
    args = parser.parse_args()
    
    if not args.file and not args.folder:
        raise ValueError("Must specify either --file or --folder")
    
    if args.file and args.folder:
        raise ValueError("Specify only one of --file or --folder")
    
    if args.file:
        upload_file(args.repo_id, args.file, args.path_in_repo, args.token)
    elif args.folder:
        upload_folder(args.repo_id, args.folder, args.path_in_repo or ".", args.token)

if __name__ == "__main__":
    main()
