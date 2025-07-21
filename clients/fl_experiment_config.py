# fl_experiment_config.py

# Experiment Configuration
ORGAN = "gb"  # Options: "pancreas", "liver", "kidneys", "gallbladder", "spleen"
DATA_RATIO = "ALL"  # Options: "1:1", "1:2", "1:5", "ALL"
USE_GIN = False  # Set to True for GIN, False for non-GIN

# Seed Configuration
SEED = 123  # Options: 42, 123, 215

# Base paths
#BASE_DATA_PATH = "/mnt/scratch1/sachindu/FL-smalldata/data"
TRAIN_FOLDER = "/mnt/scratch1/sachindu/FL-ALLDATA/FL/data/gb/train"
VAL_CT_FOLDER = "/mnt/scratch1/sachindu/FL-ALLDATA/FL/data/gb/val_ct"
VAL_MRI_FOLDER = "/mnt/scratch1/sachindu/FL-ALLDATA/FL/data/gb/val_mri"

# Weights & Biases Configuration
WANDB_PROJECT = f"federated-learning-{ORGAN}"

# Set experiment folder and W&B names based on USE_GIN
if USE_GIN:
    CHECKPOINT_SUBDIR = "gin"
    WANDB_SERVER_NAME = f"fl-server-{ORGAN}-{DATA_RATIO}-gin-{SEED}"
    SERVER_PORT = 8082  # Port for GIN experiment
else:
    CHECKPOINT_SUBDIR = "nongin"
    WANDB_SERVER_NAME = f"fl-server-{ORGAN}-{DATA_RATIO}-nongin-{SEED}"
    SERVER_PORT = 8083  # Port for non-GIN experiment

# Checkpoint and metrics paths
CHECKPOINT_DIR = f"checkpoints/{ORGAN}/{DATA_RATIO}/{SEED}/{CHECKPOINT_SUBDIR}/"
CT_METRICS_FILE = f"{CHECKPOINT_DIR}ct_client_metrics_history.json"
MRI_METRICS_FILE = f"{CHECKPOINT_DIR}mri_client_metrics_history.json"
SERVER_METRICS_FILE = f"{CHECKPOINT_DIR}server_metrics_history.json"

# Model Configuration
MODEL_CONFIG = {
    "architecture": "UNet",
    "dataset": ORGAN,
    "input_channels": 1,
    "n_classes": 1,
    "batch_size": 8,
    "learning_rate": 5e-4,
    "weight_decay": 1e-4
}

# Training Configuration
TRAIN_CONFIG = {
    "num_rounds": 100,
    "min_clients": 2,
    "local_epochs": 1
} 