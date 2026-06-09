"""
dataset.py
-----------
PyTorch Dataset and DataLoader for the 3-Class Hierarchical Acoustic CRNN.

What this file does:
  1. Reads train/val/test CSV files
  2. Loads WAV files from correct folders
  3. Computes mel spectrogram (128 × 216)
  4. Applies per-clip z-score normalisation
  5. Applies SpecAugment during training (frequency + time masking)
  6. Returns (spectrogram, main_class_idx, subclass_idx)

Train folder structure (nested):
  final_dataset/wildlife/crow/crow_001.wav
  final_dataset/nature/rain/rain_001.wav
  final_dataset/urban/siren/siren_001.wav

Val/Test folder structure (flat):
  final_dataset/validation/val_0001.wav
  final_dataset/test/test_0001.wav

Labels come from CSV — never from filename or folder name.
"""

import csv
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import config as cfg


# ── SUBCLASS TO MAIN CLASS FOLDER MAPPING ────────────────────────────────────

SUBCLASS_FOLDER = {
    "crow":           "wildlife/crow",
    "owl":            "wildlife/owl",
    "frog":           "wildlife/frog",
    "insects":        "wildlife/insects",
    "rain":           "nature/rain",
    "sea_waves":      "nature/sea_waves",
    "wind":           "nature/wind",
    "crackling_fire": "nature/crackling_fire",
    "car_horn":       "urban/car_horn",
    "engine_idling":  "urban/engine_idling",
    "siren":          "urban/siren",
    "jackhammer":     "urban/jackhammer",
}


# ── SPECAUGMENT ───────────────────────────────────────────────────────────────

def freq_mask(spec: torch.Tensor, max_mask: int, num_masks: int) -> torch.Tensor:
    """
    Mask random frequency bands in mel spectrogram.
    spec shape: (n_mels, time_frames)
    Zeroes out num_masks horizontal bands of random width up to max_mask.
    """
    cloned = spec.clone()
    n_mels = spec.shape[0]
    for _ in range(num_masks):
        f     = random.randint(0, max_mask)
        f0    = random.randint(0, n_mels - f)
        cloned[f0:f0 + f, :] = 0.0
    return cloned


def time_mask(spec: torch.Tensor, max_mask: int, num_masks: int) -> torch.Tensor:
    """
    Mask random time frames in mel spectrogram.
    spec shape: (n_mels, time_frames)
    Zeroes out num_masks vertical bands of random width up to max_mask.
    """
    cloned = spec.clone()
    n_time = spec.shape[1]
    for _ in range(num_masks):
        t     = random.randint(0, max_mask)
        t0    = random.randint(0, n_time - t)
        cloned[:, t0:t0 + t] = 0.0
    return cloned


def spec_augment(spec: torch.Tensor) -> torch.Tensor:
    """Apply SpecAugment: frequency masking + time masking. Single channel."""
    spec = freq_mask(spec, cfg.FREQ_MASK_PARAM, cfg.NUM_FREQ_MASKS)
    spec = time_mask(spec, cfg.TIME_MASK_PARAM, cfg.NUM_TIME_MASKS)
    return spec


def spec_augment_3ch(spec: torch.Tensor) -> torch.Tensor:
    """
    Apply SpecAugment to 3-channel spectrogram (mel, delta, delta2).
    Same mask applied to all 3 channels — masks must be consistent
    across channels since they represent the same time-frequency region.
    spec shape: (3, 128, 216)
    """
    n_mels = spec.shape[1]
    n_time = spec.shape[2]
    cloned = spec.clone()

    # Frequency masks — same position across all 3 channels
    for _ in range(cfg.NUM_FREQ_MASKS):
        f  = random.randint(0, cfg.FREQ_MASK_PARAM)
        f0 = random.randint(0, n_mels - f)
        cloned[:, f0:f0+f, :] = 0.0   # zero all 3 channels

    # Time masks — same position across all 3 channels
    for _ in range(cfg.NUM_TIME_MASKS):
        t  = random.randint(0, cfg.TIME_MASK_PARAM)
        t0 = random.randint(0, n_time - t)
        cloned[:, :, t0:t0+t] = 0.0   # zero all 3 channels

    return cloned


# ── AUDIO PROCESSING ──────────────────────────────────────────────────────────

def load_wav(wav_path: Path) -> np.ndarray:
    """
    Load WAV file and return as float32 numpy array.
    Resamples to cfg.SAMPLE_RATE if needed (should already be 22050Hz).
    Ensures exactly cfg.NUM_SAMPLES length.
    """
    y, sr = librosa.load(str(wav_path),
                         sr=cfg.SAMPLE_RATE,
                         mono=True,
                         dtype=np.float32)

    # Ensure exact length
    if len(y) > cfg.NUM_SAMPLES:
        y = y[:cfg.NUM_SAMPLES]
    elif len(y) < cfg.NUM_SAMPLES:
        y = np.pad(y, (0, cfg.NUM_SAMPLES - len(y)))

    return y


def compute_mel_spectrogram(y: np.ndarray) -> np.ndarray:
    """
    Compute mel spectrogram from audio array.
    Returns shape: (n_mels, time_frames) = (128, 216)

    Steps:
      1. Compute mel spectrogram (power)
      2. Convert to dB scale (log compression)
      3. Per-clip z-score normalisation: (S - mean) / std
    """
    # Step 1 — mel spectrogram (power)
    S = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.SAMPLE_RATE,
        n_mels=cfg.N_MELS,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        fmin=20,       # ignore below 20Hz (inaudible)
        fmax=11025,    # Nyquist for 22050Hz
    )

    # Step 2 — convert to dB (log compression)
    # ref=np.max makes the loudest frequency 0dB, rest relative
    S_db = librosa.power_to_db(S, ref=np.max)

    # Step 3 — per-clip z-score normalisation
    if cfg.SPEC_NORM:
        mean = S_db.mean()
        std  = S_db.std() + 1e-8   # epsilon prevents division by zero
        S_db = (S_db - mean) / std

    # Ensure exact time dimension
    if S_db.shape[1] > cfg.TIME_FRAMES:
        S_db = S_db[:, :cfg.TIME_FRAMES]
    elif S_db.shape[1] < cfg.TIME_FRAMES:
        pad = cfg.TIME_FRAMES - S_db.shape[1]
        S_db = np.pad(S_db, ((0, 0), (0, pad)))

    return S_db.astype(np.float32)


# ── DATASET ───────────────────────────────────────────────────────────────────

class AcousticDataset(Dataset):
    """
    PyTorch Dataset for the 3-Class Hierarchical Acoustic CRNN.

    Args:
        csv_path:   Path to train/val/test CSV file
        audio_dir:  Base directory for audio files
                    Train: final_dataset/ (nested class folders)
                    Val:   final_dataset/validation/
                    Test:  final_dataset/test/
        split:      'train', 'val', or 'test'
                    SpecAugment only applied when split='train'
        augment:    Override augmentation (None = use split default)
    """

    def __init__(self,
                 csv_path: Path,
                 audio_dir: Path,
                 split: str = "train",
                 augment: bool = None):

        self.audio_dir = Path(audio_dir)
        self.split     = split
        self.augment   = augment if augment is not None else (split == "train")

        # Load CSV
        self.samples = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                subclass   = row["subclass"]
                main_class = row["main_class"]
                filename   = row["filename"]

                # Resolve full path
                if split == "train":
                    # Nested: final_dataset/wildlife/crow/crow_001.wav
                    wav_path = (self.audio_dir /
                                SUBCLASS_FOLDER[subclass] /
                                filename)
                else:
                    # Flat: final_dataset/validation/val_0001.wav
                    wav_path = self.audio_dir / filename

                self.samples.append({
                    "wav_path":      wav_path,
                    "filename":      filename,
                    "subclass":      subclass,
                    "main_class":    main_class,
                    "sub_idx":       cfg.SUB_TO_IDX[subclass],
                    "main_idx":      cfg.MAIN_TO_IDX[main_class],
                    "source":        row.get("source", "unknown"),
                })

        print(f"  {split.upper()} dataset: {len(self.samples)} clips  "
              f"augment={self.augment}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample   = self.samples[idx]
        wav_path = sample["wav_path"]

        # Load audio
        try:
            y = load_wav(wav_path)
        except Exception as e:
            print(f"  WARNING: error loading {wav_path.name}: {e}")
            # Return silence on error — model will see zero spectrogram
            y = np.zeros(cfg.NUM_SAMPLES, dtype=np.float32)

        # Compute mel spectrogram → shape (128, 216)
        spec = compute_mel_spectrogram(y)

        # Convert to tensor → shape (1, 128, 216)
        # The 1 is the channel dimension — CNN expects (batch, channels, H, W)
        spec_tensor = torch.from_numpy(spec).unsqueeze(0)

        # Apply SpecAugment during training only
        if self.augment and cfg.USE_SPEC_AUGMENT:
            spec_tensor = spec_tensor.squeeze(0)  # (128, 216)
            spec_tensor = spec_augment(spec_tensor)
            spec_tensor = spec_tensor.unsqueeze(0)  # (1, 128, 216)

        return (
            spec_tensor,                                    # (1, 128, 216) float32
            torch.tensor(sample["main_idx"], dtype=torch.long),  # scalar int
            torch.tensor(sample["sub_idx"],  dtype=torch.long),  # scalar int
        )

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse frequency weights for each subclass.
        Used in weighted CrossEntropyLoss to handle class imbalance.
        car_horn has 405 clips vs 500 for others — weights compensate.
        Returns tensor of shape (NUM_SUBCLASSES,)
        """
        counts = np.zeros(cfg.NUM_SUBCLASSES, dtype=np.float32)
        for sample in self.samples:
            counts[sample["sub_idx"]] += 1

        # Inverse frequency: rare classes get higher weight
        weights = 1.0 / (counts + 1e-8)
        weights = weights / weights.sum() * cfg.NUM_SUBCLASSES
        return torch.from_numpy(weights)

    def get_sample_info(self, idx: int) -> dict:
        """Return metadata for a sample — useful for debugging."""
        return self.samples[idx]


# ── DATALOADERS ───────────────────────────────────────────────────────────────

def get_dataloaders(
    batch_size: int = cfg.BATCH_SIZE,
    num_workers: int = cfg.NUM_WORKERS,
) -> tuple:
    """
    Create and return train, val, test DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, class_weights

    Usage:
        train_loader, val_loader, test_loader, weights = get_dataloaders()
        for specs, main_labels, sub_labels in train_loader:
            ...
    """
    print("\nLoading datasets...")

    train_dataset = AcousticDataset(
        csv_path  = cfg.TRAIN_CSV,
        audio_dir = cfg.TRAIN_AUDIO_DIR,
        split     = "train",
    )

    val_dataset = AcousticDataset(
        csv_path  = cfg.VAL_CSV,
        audio_dir = cfg.VAL_AUDIO_DIR,
        split     = "val",
    )

    test_dataset = AcousticDataset(
        csv_path  = cfg.TEST_CSV,
        audio_dir = cfg.TEST_AUDIO_DIR,
        split     = "test",
    )

    # Class weights from training set
    class_weights = train_dataset.get_class_weights()

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,         # shuffle every epoch
        num_workers = num_workers,
        pin_memory  = True,         # faster GPU transfer
        drop_last   = True,         # drop incomplete last batch
                                    # keeps batch size consistent for BatchNorm
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,        # no shuffle for val/test
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
    )

    print(f"\n  Train batches: {len(train_loader)} "
          f"({len(train_dataset)} clips / {batch_size} batch)")
    print(f"  Val batches:   {len(val_loader)} "
          f"({len(val_dataset)} clips)")
    print(f"  Test batches:  {len(test_loader)} "
          f"({len(test_dataset)} clips)")

    return train_loader, val_loader, test_loader, class_weights


# ── QUICK TEST ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run this to verify DataLoader works correctly before training.
    Loads one batch and prints shapes.

    Usage:
        python pipeline/dataset.py
    """
    import time

    print("Testing DataLoader...")
    cfg.print_config()

    train_loader, val_loader, test_loader, weights = get_dataloaders(
        batch_size  = 8,    # small batch for quick test
        num_workers = 0,    # single worker for testing
    )

    print("\n── Train batch test ──")
    start = time.time()
    specs, main_labels, sub_labels = next(iter(train_loader))
    elapsed = time.time() - start

    print(f"  Spectrogram batch shape: {specs.shape}")
    print(f"    Expected:              (8, 1, 128, 216)")
    print(f"  Main class labels:       {main_labels}")
    print(f"  Subclass labels:         {sub_labels}")
    print(f"  Spectrogram dtype:       {specs.dtype}")
    print(f"  Spectrogram min/max:     {specs.min():.3f} / {specs.max():.3f}")
    print(f"  Load time (8 clips):     {elapsed:.2f}s")

    print(f"\n── Class weights ──")
    for i, w in enumerate(weights):
        print(f"  {cfg.IDX_TO_SUB[i]:<20} weight: {w:.4f}")

    print(f"\n── Val batch test ──")
    specs, main_labels, sub_labels = next(iter(val_loader))
    print(f"  Spectrogram batch shape: {specs.shape}")
    print(f"  Main class labels:       {main_labels}")
    print(f"  Subclass labels:         {sub_labels}")

    print(f"\n── Label mapping check ──")
    sample_info = train_loader.dataset.get_sample_info(0)
    print(f"  Sample 0:")
    print(f"    File:      {sample_info['filename']}")
    print(f"    Subclass:  {sample_info['subclass']} → idx {sample_info['sub_idx']}")
    print(f"    Main:      {sample_info['main_class']} → idx {sample_info['main_idx']}")
    print(f"    Source:    {sample_info['source']}")

    print(f"\n✓ DataLoader working correctly")
    print(f"  Ready to proceed to model.py")
