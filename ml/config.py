import os
import torch

# Directory Resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))               # NetraVision/ml
REPO_ROOT = os.path.dirname(BASE_DIR)                               # NetraVision root

# External Dataset Paths (kept outside repository)
DATASET_DIR = os.environ.get("APTOS_DATASET_DIR", r"D:\aptos2019-blindness-detection")
TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")
TEST_CSV = os.path.join(DATASET_DIR, "test.csv")
TRAIN_IMAGES_DIR = os.path.join(DATASET_DIR, "train_images")
TEST_IMAGES_DIR = os.path.join(DATASET_DIR, "test_images")

# Outputs and Checkpoints (inside NetraVision/outputs)
OUTPUT_DIR = os.environ.get("NETRAVISION_OUTPUT_DIR", os.path.join(REPO_ROOT, "outputs"))
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")

# Dataset & Image Preprocessing
IMG_SIZE = 256  # standard resolution for training (balances speed and detail)
NUM_CLASSES = 5
CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

# Training Hyperparameters
BATCH_SIZE = 16
NUM_WORKERS = 0  # 0 is safest on Windows to prevent multiprocessing lockups
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 15
VAL_SPLIT = 0.2
RANDOM_SEED = 42

# Device selection: automatically uses CUDA GPU if PyTorch CUDA is enabled, else falls back to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
