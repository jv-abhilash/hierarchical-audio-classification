# 3-Class Hierarchical Acoustic CRNN

A hierarchical Convolutional Recurrent Neural Network for environmental sound classification across 3 main classes and 12 subclasses.

## Results

| Metric | Score |
|---|---|
| Main class accuracy | 92.47% |
| Subclass accuracy | 78.96% |
| Mean confidence | 0.829 |

## Classes

| Main Class | Subclasses |
|---|---|
| Wildlife | crow, owl, frog, insects |
| Nature | rain, sea_waves, wind, crackling_fire |
| Urban | car_horn, engine_idling, siren, jackhammer |

## Architecture

```
Input: Mel Spectrogram (1 × 128 × 216)
    ↓
CNN Encoder (4 blocks: 1→32→64→128→256 channels)
    ↓
Reshape (13 time steps × 2048 features)
    ↓
Bidirectional GRU (2 layers, hidden=256)
    ↓
Head 1 → Main class (3 outputs)
Head 2 → Subclass  (12 outputs)
```

- **Parameters:** 5,380,591 (~20.5 MB)
- **Input:** 22050Hz mono WAV, 5 seconds
- **Spectrogram:** 128 mel bins, n_fft=2048, hop=512

## Dataset

- **Total clips:** 5,905 (after cleaning and balancing)
- **Sources:** ESC-50, FSD50K, BirdCLEF 2021, AnuraSet, UrbanSound8K, freefield1010, forest_wild_fire
- **Split:** 70% train / 15% val / 15% test
- **Format:** 22050Hz mono PCM_16 5.0s, RMS normalised to 0.1

## Project Structure

```
dl_project/
    pipeline/
        config.py       ← all hyperparameters
        dataset.py      ← DataLoader + mel spectrogram
        model.py        ← CRNN architecture
        train.py        ← training loop
        evaluate.py     ← evaluation + confusion matrix
    preprocessing/
        filter_esc50.py
        rename_esc50.py
        augment_esc50.py
        filter_fsd50k.py
        slice_fsd50k.py
        filter_birdclef.py
        filter_anuraset.py
        filter_urbansound.py
        filter_freefield.py
        filter_wildfire.py
        remove_flagged.py
        trim_crow_owl.py
        create_final_dataset.py
        normalise_audio.py
        normalise_rms.py
        split_dataset.py
        PIPELINE_AUDIT_TRAIL.md
    final_dataset/          ← not tracked (audio files)
    checkpoints/            ← not tracked (model weights)
    logs/                   ← not tracked
    requirements.txt
    .gitignore
```

## Setup

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/dl_project.git
cd dl_project

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install PyTorch (CUDA 12.8 — RTX 5070 Ti)
pip install --pre torch torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128

# Install other dependencies
pip install -r requirements.txt
```

## Training

```bash
# Train from scratch
python pipeline/train.py

# Resume from checkpoint
python pipeline/train.py --resume checkpoints/best_crnn.pt

# Custom epochs
python pipeline/train.py --epochs 30
```

## Evaluation

```bash
# Evaluate on test set
python pipeline/evaluate.py

# Evaluate on validation set
python pipeline/evaluate.py --split val
```

## Preprocessing

See `preprocessing/PIPELINE_AUDIT_TRAIL.md` for the complete
data pipeline documentation including every script, command,
and design decision.

## Hardware

Trained on NVIDIA RTX 5070 Ti (15GB VRAM), ~3 minutes per epoch.
