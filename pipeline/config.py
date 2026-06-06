"""
config.py
----------
Central configuration for the 3-Class Hierarchical Acoustic CRNN.
All hyperparameters, paths, and settings live here.
Change values here — never hardcode in other files.

GPU: NVIDIA RTX 5070 Ti, 15GB VRAM
"""

from pathlib import Path

# ── PATHS ─────────────────────────────────────────────────────────────────────

# Project root — adjust if running from a different directory
PROJECT_ROOT   = Path(__file__).parent.parent   # one level up from pipeline/

FINAL_DATASET  = PROJECT_ROOT / "final_dataset"
TRAIN_CSV      = FINAL_DATASET / "train_metadata.csv"
VAL_CSV        = FINAL_DATASET / "val_metadata.csv"
TEST_CSV       = FINAL_DATASET / "test_metadata.csv"

# Audio lives in nested folders for train, flat for val/test
TRAIN_AUDIO_DIR = FINAL_DATASET          # DataLoader resolves wildlife/crow/etc
VAL_AUDIO_DIR   = FINAL_DATASET / "validation"
TEST_AUDIO_DIR  = FINAL_DATASET / "test"

# Checkpoints and logs saved here
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR        = PROJECT_ROOT / "logs"

# ── AUDIO SETTINGS ─────────────────────────────────────────────────────────────

SAMPLE_RATE    = 22050          # Hz — all clips normalised to this
CLIP_DURATION  = 5.0            # seconds
NUM_SAMPLES    = int(SAMPLE_RATE * CLIP_DURATION)   # 110,250 samples

# ── MEL SPECTROGRAM SETTINGS ───────────────────────────────────────────────────

N_MELS         = 128            # mel frequency bins (rows of spectrogram)
N_FFT          = 2048           # FFT window size
HOP_LENGTH     = 512            # hop between frames
# Resulting spectrogram shape: (N_MELS, TIME_FRAMES)
# TIME_FRAMES = ceil(NUM_SAMPLES / HOP_LENGTH) = ceil(110250 / 512) = 216
TIME_FRAMES    = 216

# Spectrogram normalisation (per-clip z-score)
# Applied in DataLoader after power_to_db conversion
SPEC_NORM      = True           # True = (S - mean) / std per clip

# ── CLASS MAPPINGS ─────────────────────────────────────────────────────────────

# Main class labels (Head 1 output)
MAIN_CLASSES   = ["wildlife", "nature", "urban"]
NUM_MAIN       = len(MAIN_CLASSES)

MAIN_TO_IDX    = {c: i for i, c in enumerate(MAIN_CLASSES)}
IDX_TO_MAIN    = {i: c for i, c in enumerate(MAIN_CLASSES)}

# Subclass labels (Head 2 output) — ordered by main class
SUBCLASSES     = [
    # wildlife
    "crow", "owl", "frog", "insects",
    # nature
    "rain", "sea_waves", "wind", "crackling_fire",
    # urban
    "car_horn", "engine_idling", "siren", "jackhammer",
]
NUM_SUBCLASSES = len(SUBCLASSES)

SUB_TO_IDX     = {c: i for i, c in enumerate(SUBCLASSES)}
IDX_TO_SUB     = {i: c for i, c in enumerate(SUBCLASSES)}

# Which subclasses belong to which main class
SUBCLASS_TO_MAIN = {
    "crow":           "wildlife",
    "owl":            "wildlife",
    "frog":           "wildlife",
    "insects":        "wildlife",
    "rain":           "nature",
    "sea_waves":      "nature",
    "wind":           "nature",
    "crackling_fire": "nature",
    "car_horn":       "urban",
    "engine_idling":  "urban",
    "siren":          "urban",
    "jackhammer":     "urban",
}

# Subclass index → main class index (used in hierarchical loss)
SUB_TO_MAIN_IDX = {
    SUB_TO_IDX[sub]: MAIN_TO_IDX[main]
    for sub, main in SUBCLASS_TO_MAIN.items()
}

# ── MODEL ARCHITECTURE ─────────────────────────────────────────────────────────

# CNN — extracts spatial features from mel spectrogram
CNN_CHANNELS   = [1, 32, 64, 128, 256]   # input + 4 conv blocks
                                           # input: 1 channel (mono spectrogram)
CNN_KERNEL     = (3, 3)                    # conv kernel size
CNN_POOL       = (2, 2)                    # max pool size per block
CNN_DROPOUT    = 0.2                       # dropout after each conv block

# After 4 pooling layers on (128 × 216) input:
# Frequency: 128 → 64 → 32 → 16 → 8
# Time:      216 → 108 → 54 → 27 → 13
# Feature map: 256 × 8 × 13 = 26,624 values
# Reshape for RNN: sequence of 13 time steps, each with 256×8=2048 features

# RNN — captures temporal patterns across time steps
RNN_INPUT_SIZE  = 256 * 8      # CNN output flattened per time step = 2048
RNN_HIDDEN_SIZE = 256           # hidden state size
RNN_LAYERS      = 2             # stacked GRU layers
RNN_DROPOUT     = 0.3           # dropout between RNN layers
RNN_BIDIRECTIONAL = True        # bidirectional GRU
# Bidirectional doubles output: 256 × 2 = 512

# Output heads
HEAD_HIDDEN     = 256           # FC layer before classification head
HEAD_DROPOUT    = 0.4           # dropout before output

# Head 1 — main class (wildlife / nature / urban)
HEAD1_OUT       = NUM_MAIN      # 3 outputs

# Head 2 — subclass (crow / owl / ... / jackhammer)
HEAD2_OUT       = NUM_SUBCLASSES  # 12 outputs

# ── TRAINING SETTINGS ──────────────────────────────────────────────────────────

BATCH_SIZE      = 64            # RTX 5070 Ti 15GB — 64 is safe with mixed precision
NUM_EPOCHS      = 50            # max epochs (early stopping will trigger before)
NUM_WORKERS     = 4             # DataLoader parallel workers

# Loss weights — hierarchical loss
# total_loss = loss_main + LAMBDA_SUB * loss_subclass
LAMBDA_SUB      = 0.7           # subclass loss weight
                                # main class loss weight = 1.0 (fixed)

# Optimiser — AdamW
LEARNING_RATE   = 3e-4          # initial LR
WEIGHT_DECAY    = 1e-4          # L2 regularisation

# LR Scheduler — ReduceLROnPlateau
LR_PATIENCE     = 5             # epochs without improvement before reducing LR
LR_FACTOR       = 0.5           # multiply LR by this when reducing
LR_MIN          = 1e-6          # minimum LR

# Early stopping
EARLY_STOP_PATIENCE = 10        # stop if val loss doesn't improve for 10 epochs

# Mixed precision training (float16) — speeds up training on RTX 5070 Ti
USE_AMP         = True          # Automatic Mixed Precision

# ── DATA AUGMENTATION (applied during training only) ──────────────────────────

# Applied to mel spectrogram in DataLoader (not to raw audio — already done)
USE_SPEC_AUGMENT = True         # SpecAugment — mask frequency and time bands

# SpecAugment parameters
FREQ_MASK_PARAM  = 20           # max frequency bands to mask (out of 128)
TIME_MASK_PARAM  = 30           # max time frames to mask (out of 216)
NUM_FREQ_MASKS   = 2            # number of frequency masks
NUM_TIME_MASKS   = 2            # number of time masks

# ── CHECKPOINT SETTINGS ────────────────────────────────────────────────────────

SAVE_BEST_ONLY  = True          # only save when val accuracy improves
CHECKPOINT_NAME = "best_crnn.pt"

# ── INFERENCE SETTINGS ─────────────────────────────────────────────────────────

# OOD detection (Out-of-Distribution)
# If max softmax probability < threshold → predict "Unknown"
OOD_CONFIDENCE_THRESHOLD = 0.6  # tune after training
OOD_ENTROPY_THRESHOLD    = 1.5  # entropy threshold (max entropy for 12 classes = ln(12) ≈ 2.48)

# ── REPRODUCIBILITY ────────────────────────────────────────────────────────────

RANDOM_SEED     = 42

# ── DEVICE ─────────────────────────────────────────────────────────────────────

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── DISPLAY ────────────────────────────────────────────────────────────────────

def print_config():
    """Print key config values for verification at training start."""
    print("=" * 60)
    print("  ACOUSTIC CRNN — Configuration")
    print("=" * 60)
    print(f"  Device:          {DEVICE}")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory // 1024**3
        print(f"  GPU:             {torch.cuda.get_device_name(0)} ({vram}GB)")
    print(f"  Dataset:         {FINAL_DATASET}")
    print(f"  Batch size:      {BATCH_SIZE}")
    print(f"  Epochs:          {NUM_EPOCHS}")
    print(f"  Learning rate:   {LEARNING_RATE}")
    print(f"  Mixed precision: {USE_AMP}")
    print(f"  SpecAugment:     {USE_SPEC_AUGMENT}")
    print(f"  Mel shape:       ({N_MELS} × {TIME_FRAMES})")
    print(f"  CNN channels:    {CNN_CHANNELS}")
    print(f"  RNN hidden:      {RNN_HIDDEN_SIZE} × {RNN_LAYERS} layers "
          f"({'bidirectional' if RNN_BIDIRECTIONAL else 'unidirectional'})")
    print(f"  Main classes:    {NUM_MAIN}  {MAIN_CLASSES}")
    print(f"  Subclasses:      {NUM_SUBCLASSES}")
    print(f"  Lambda sub:      {LAMBDA_SUB}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
