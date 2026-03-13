# download_models.py
import os
from huggingface_hub import hf_hub_download

# Directory where models are stored
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# List of model filenames in your HuggingFace repo
FILES = [
    "random_forest.pkl",
    "scaler.pkl",
    "label_encoder.pkl",
    "resnet50finetuned.keras"
]

REPO_ID = "maik122/musicgenreclassifier-models"


def download_models_if_missing(models_dir=MODEL_DIR):
    os.makedirs(models_dir, exist_ok=True)
    for f in FILES:
        path = os.path.join(models_dir, f)
        if not os.path.exists(path):
            print(f"Downloading {f} from HuggingFace...")
            try:
                hf_hub_download(repo_id=REPO_ID, filename=f, local_dir=models_dir)
                print(f"{f} downloaded successfully.")
            except Exception as e:
                print(f"Failed to download {f}: {e}")
        else:
            print(f"{f} already exists locally, skipping download.")


# Run immediately if script is executed directly
if __name__ == "__main__":
    download_models_if_missing()