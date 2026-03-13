# src/config.py
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Data paths ---
FEATURES_30_SEC = os.path.join(BASE_DIR, "Data", "features_30_sec.csv")
IMG_DIR         = os.path.join(BASE_DIR, "Data", "images_original")

# --- Saved model paths ---
RF_PATH     = os.path.join(BASE_DIR, "models", "random_forest.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
LE_PATH     = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
CNN_PATH    = os.path.join(BASE_DIR, "models", "resnet50_finetuned.keras")

# --- Training ---
TEST_SIZE       = 0.2
RANDOM_STATE    = 42
N_ESTIMATORS    = 200

# --- CNN ---
IMG_SIZE        = (224, 224)
BATCH_SIZE      = 32
EPOCHS_FROZEN   = 15
EPOCHS_FINETUNE = 10
FINETUNE_LR     = 1e-5

GENRES = [
    "blues", "classical", "country", "disco",
    "hiphop", "jazz", "metal", "pop", "reggae", "rock"
]