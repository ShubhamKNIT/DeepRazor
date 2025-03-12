import os

# Define root directory
ROOT_DIR = os.path.expanduser(".")

# Define subdirectories
DATA_DIR = os.path.join(ROOT_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
CHKPT_DIR = os.path.join(ROOT_DIR, "Checkpoints")
LOG_DIR = os.path.join(ROOT_DIR, "Logs")
VAL_IMG_DIR = os.path.join(ROOT_DIR, "Results")

# CSV Paths
TRAIN_CSV_PATH = os.path.join(LOG_DIR, "train.csv")
VAL_CSV_PATH = os.path.join(LOG_DIR, "val.csv")