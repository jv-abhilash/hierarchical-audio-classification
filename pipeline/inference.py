"""
inference.py
------------
Run single-file inference with the trained Acoustic CRNN checkpoint.

Usage:
    python pipeline/inference.py --audio path/to/file.wav
    python pipeline/inference.py --audio final_dataset/test/test_0001.wav
    python pipeline/inference.py --audio file.wav --checkpoint checkpoints/best_crnn.pt --top-k 5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent))
import config as cfg
from dataset import compute_mel_spectrogram, load_wav
from model import AcousticCRNN


def load_model(checkpoint_path: Path) -> tuple[AcousticCRNN, dict]:
    """Load the trained model checkpoint for inference."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE)
    if "model_state" not in checkpoint:
        raise KeyError(
            f"Checkpoint at {checkpoint_path} does not contain 'model_state'."
        )

    model = AcousticCRNN().to(cfg.DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def preprocess_audio(audio_path: Path) -> torch.Tensor:
    """Convert a WAV file into the model's expected input tensor."""
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    waveform = load_wav(audio_path)
    spec = compute_mel_spectrogram(waveform)
    spec = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)
    return spec.to(cfg.DEVICE, dtype=torch.float32)


@torch.no_grad()
def predict_audio(
    model: AcousticCRNN,
    spec: torch.Tensor,
    threshold: float = cfg.OOD_CONFIDENCE_THRESHOLD,
    top_k: int = 3,
) -> dict:
    """Run inference and return structured predictions."""
    main_logits, sub_logits = model(spec)

    main_probs = F.softmax(main_logits, dim=-1)[0]
    sub_probs = F.softmax(sub_logits, dim=-1)[0]

    main_conf, main_idx = torch.max(main_probs, dim=-1)
    sub_conf, sub_idx = torch.max(sub_probs, dim=-1)

    k = min(top_k, cfg.NUM_SUBCLASSES)
    top_probs, top_indices = torch.topk(sub_probs, k=k)

    top_subclasses = [
        {
            "rank": rank + 1,
            "subclass": cfg.IDX_TO_SUB[idx.item()],
            "confidence": float(prob.item()),
            "main_class": cfg.SUBCLASS_TO_MAIN[cfg.IDX_TO_SUB[idx.item()]],
        }
        for rank, (prob, idx) in enumerate(zip(top_probs, top_indices))
    ]

    return {
        "main_class": cfg.IDX_TO_MAIN[main_idx.item()],
        "main_confidence": float(main_conf.item()),
        "subclass": cfg.IDX_TO_SUB[sub_idx.item()],
        "sub_confidence": float(sub_conf.item()),
        "is_ood": bool(
            (main_conf.item() < threshold) or (sub_conf.item() < threshold)
        ),
        "ood_threshold": threshold,
        "top_subclasses": top_subclasses,
        "main_probabilities": {
            cfg.IDX_TO_MAIN[i]: float(main_probs[i].item())
            for i in range(cfg.NUM_MAIN)
        },
    }


def print_prediction(audio_path: Path, checkpoint: dict, result: dict) -> None:
    """Pretty-print inference output."""
    print("=" * 64)
    print("  ACOUSTIC CRNN INFERENCE")
    print("=" * 64)
    print(f"  Audio file:          {audio_path}")
    print(f"  Device:              {cfg.DEVICE}")
    print(f"  Checkpoint epoch:    {checkpoint.get('epoch', 'unknown')}")

    metrics = checkpoint.get("metrics")
    if isinstance(metrics, dict) and "sub_acc" in metrics:
        print(f"  Saved val sub acc:   {metrics['sub_acc']:.2f}%")

    print()
    print("Prediction")
    print(f"  Main class:          {result['main_class']:<16} ({result['main_confidence']:.4f})")
    print(f"  Subclass:            {result['subclass']:<16} ({result['sub_confidence']:.4f})")
    print(f"  OOD flagged:         {result['is_ood']}")
    print()
    print("Main Class Probabilities")
    for name, prob in result["main_probabilities"].items():
        print(f"  {name:<16} {prob:.4f}")
    print()
    print("Top Subclass Predictions")
    for item in result["top_subclasses"]:
        print(
            f"  {item['rank']:>2}. {item['subclass']:<18} "
            f"{item['confidence']:.4f}  ({item['main_class']})"
        )
    print("=" * 64)


def main():
    parser = argparse.ArgumentParser(
        description="Run single-file inference with the trained Acoustic CRNN."
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to the input audio file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(cfg.CHECKPOINT_DIR / cfg.CHECKPOINT_NAME),
        help="Path to the trained checkpoint.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=cfg.OOD_CONFIDENCE_THRESHOLD,
        help="Confidence threshold used for OOD flagging.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top subclass predictions to print.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the prediction result as JSON.",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    checkpoint_path = Path(args.checkpoint)

    model, checkpoint = load_model(checkpoint_path)
    spec = preprocess_audio(audio_path)
    result = predict_audio(model, spec, threshold=args.threshold, top_k=args.top_k)

    if args.json:
        payload = {
            "audio_file": str(audio_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "result": result,
        }
        print(json.dumps(payload, indent=2))
        return

    print_prediction(audio_path, checkpoint, result)


if __name__ == "__main__":
    main()
