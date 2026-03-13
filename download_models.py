import os
import urllib.request

# Folder where models will be stored
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Exact model filenames and their HuggingFace links
FILES = {
    "random_forest.pkl": "https://huggingface.co/maik122/musicgenreclassifier-models/resolve/main/random_forest.pkl",
    "scaler.pkl": "https://huggingface.co/maik122/musicgenreclassifier-models/resolve/main/scaler.pkl",
    "label_encoder.pkl": "https://huggingface.co/maik122/musicgenreclassifier-models/resolve/main/label_encoder.pkl",
    "resnet50finetuned.keras": "https://huggingface.co/maik122/musicgenreclassifier-models/resolve/main/resnet50finetuned.keras"
}


def download_models_if_missing(models_dir="models"):
    os.makedirs(models_dir, exist_ok=True)
    for filename, url in FILES.items():
        path = os.path.join(models_dir, filename)
        if not os.path.exists(path):
            try:
                print(f"Downloading {filename}...")
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
        else:
            print(f"{filename} already exists locally, skipping download.")
