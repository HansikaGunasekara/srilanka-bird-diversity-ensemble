"""Fetch the dataset from Hugging Face into data_raw/ (requires a prior `huggingface-cli login`)."""
import os
from huggingface_hub import snapshot_download

REPO = "DilushaChandrasiri/SriLanka-Bird-Diversity-Dataset"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    dest = os.path.join(ROOT, "data_raw")
    p = snapshot_download(REPO, repo_type="dataset", local_dir=dest)
    print("downloaded to", p)
    for f in sorted(os.listdir(dest)):
        if not f.startswith("."):
            print(f"  {f}  {os.path.getsize(os.path.join(dest, f))/1e6:.1f} MB")
