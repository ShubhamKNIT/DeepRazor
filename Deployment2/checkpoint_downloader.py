import os
from pathlib import Path
from huggingface_hub import hf_hub_download

def create_directory(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
    print(f"📁 Ensured directory exists: {path}")

def download_file(repo_id: str, filename: str, dest_dir: str, repo_type: str = "model") -> str:
    """
    Download a single file from a Hugging Face repo.
    Returns the local path or raises on error.
    """
    print(f"📥 Downloading {filename} from {repo_id} …")
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        local_dir=dest_dir,
        local_dir_use_symlinks=False
    )
    print(f"✅ Saved to {local_path}")
    return local_path

def download_all_checkpoints(destination_dir: str = "./checkpoints"):
    """
    Downloads these four files:
      • facebook/sam2.1-hiera-large → sam2.1_hiera_large.pt
      • facebook/sam2.1-hiera-large → sam2.1_hiera_l.yaml
      • HV-Khurdula/big-lama → best.ckpt
      • HV-Khurdula/big-lama → config.yaml
    """
    create_directory(destination_dir)
    downloaded = []

    # 1. SAM2.1 weights
    downloaded.append(download_file(
        repo_id="facebook/sam2.1-hiera-large",
        filename="sam2.1_hiera_large.pt",
        dest_dir=destination_dir
    ))

    # 2. SAM2.1 config
    downloaded.append(download_file(
        repo_id="facebook/sam2.1-hiera-large",
        filename="sam2.1_hiera_l.yaml",
        dest_dir=destination_dir
    ))

    # 3. Big-Lama checkpoint
    downloaded.append(download_file(
        repo_id="HV-Khurdula/big-lama",
        filename="best.ckpt",
        dest_dir=destination_dir+"/big-lama/models"
    ))

    # 4. Big-Lama config
    downloaded.append(download_file(
        repo_id="HV-Khurdula/big-lama",
        filename="config.yaml",
        dest_dir=destination_dir+"/big-lama"
    ))

    print(f"\n🎉 Downloaded {len(downloaded)} files into `{destination_dir}`")
    return downloaded

if __name__ == "__main__":
    # Ensure you have: pip install huggingface_hub
    files = download_all_checkpoints("./checkpoints")
    for f in files:
        print(f" - {f}")
