import os
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_RAW        = os.path.join(BASE_DIR, "data", "raw", "UrbanSound8K")
DATA_PROCESSED  = os.path.join(BASE_DIR, "data", "processed")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "outputs", "checkpoints")
PLOTS_DIR       = os.path.join(BASE_DIR, "outputs", "plots")
SPECTROGRAMS_DIR= os.path.join(BASE_DIR, "outputs", "spectrograms")

# ── Audio Preprocessing ────────────────────────────────────────────────────────
SAMPLE_RATE     = 22050
DURATION        = 4
N_MELS          = 64
N_FFT           = 1024
HOP_LENGTH      = 512
F_MIN           = 20
F_MAX           = 8000

# ── Dataset ────────────────────────────────────────────────────────────────────
GUNSHOT_CLASS_ID = 6
NUM_CLASSES      = 2
TEST_FOLD        = 10

# ── Model ──────────────────────────────────────────────────────────────────────
IN_CHANNELS     = 1
DROPOUT         = 0.3

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE      = 64
NUM_EPOCHS      = 30
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-4
USE_AMP         = True

# ── Hardware — auto-detected ───────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE      = "cuda"
    PIN_MEMORY  = True
elif torch.backends.mps.is_available():
    DEVICE      = "mps"
    PIN_MEMORY  = False
else:
    DEVICE      = "cpu"
    PIN_MEMORY  = False

DEVICE_TYPE     = DEVICE.split(":")[0]  # "cuda", "mps", or "cpu"

print(f"[config] Device: {DEVICE} | PIN_MEMORY: {PIN_MEMORY}")

# ── Logging ────────────────────────────────────────────────────────────────────
NUM_WORKERS     = 4
LOG_INTERVAL    = 5
SAVE_BEST_ONLY  = True