"""
evaluate.py
------------
Evaluation script for the trained Acoustic CRNN model.
Runs inference on the test set and produces:
  - Overall accuracy (main class + subclass)
  - Per-class accuracy breakdown
  - Confusion matrix (subclass level)
  - Classification report
  - Saves results to logs/evaluation_report.txt

Usage:
    python pipeline/evaluate.py
    python pipeline/evaluate.py --checkpoint checkpoints/best_crnn.pt
    python pipeline/evaluate.py --split val
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent))
import config as cfg
from dataset import get_dataloaders
from model import AcousticCRNN


# ── EVALUATION ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, split_name: str = "test") -> dict:
    """
    Run full evaluation on a dataloader.
    Returns dict with all metrics and predictions.
    """
    model.eval()

    all_main_preds  = []
    all_sub_preds   = []
    all_main_labels = []
    all_sub_labels  = []
    all_main_confs  = []
    all_sub_confs   = []

    for specs, main_labels, sub_labels in loader:
        specs       = specs.to(cfg.DEVICE, non_blocking=True)
        main_labels = main_labels.to(cfg.DEVICE)
        sub_labels  = sub_labels.to(cfg.DEVICE)

        main_logits, sub_logits = model(specs)

        main_probs = F.softmax(main_logits, dim=-1)
        sub_probs  = F.softmax(sub_logits,  dim=-1)

        main_conf, main_pred = main_probs.max(dim=-1)
        sub_conf,  sub_pred  = sub_probs.max(dim=-1)

        all_main_preds.extend(main_pred.cpu().numpy())
        all_sub_preds.extend(sub_pred.cpu().numpy())
        all_main_labels.extend(main_labels.cpu().numpy())
        all_sub_labels.extend(sub_labels.cpu().numpy())
        all_main_confs.extend(main_conf.cpu().numpy())
        all_sub_confs.extend(sub_conf.cpu().numpy())

    # Convert to numpy
    main_preds  = np.array(all_main_preds)
    sub_preds   = np.array(all_sub_preds)
    main_labels = np.array(all_main_labels)
    sub_labels  = np.array(all_sub_labels)
    main_confs  = np.array(all_main_confs)
    sub_confs   = np.array(all_sub_confs)

    # Overall accuracy
    main_acc = (main_preds == main_labels).mean() * 100
    sub_acc  = (sub_preds  == sub_labels).mean()  * 100

    # Per main class accuracy
    main_class_acc = {}
    for idx, name in cfg.IDX_TO_MAIN.items():
        mask = main_labels == idx
        if mask.sum() > 0:
            main_class_acc[name] = (main_preds[mask] == idx).mean() * 100

    # Per subclass accuracy
    sub_class_acc = {}
    for idx, name in cfg.IDX_TO_SUB.items():
        mask = sub_labels == idx
        if mask.sum() > 0:
            acc   = (sub_preds[mask] == idx).mean() * 100
            count = mask.sum()
            sub_class_acc[name] = {"acc": acc, "count": int(count)}

    # Confusion matrix (subclass level)
    n = cfg.NUM_SUBCLASSES
    confusion = np.zeros((n, n), dtype=int)
    for true, pred in zip(sub_labels, sub_preds):
        confusion[true][pred] += 1

    # Mean confidence
    mean_main_conf = main_confs.mean()
    mean_sub_conf  = sub_confs.mean()

    return {
        "split":           split_name,
        "n_samples":       len(sub_labels),
        "main_acc":        main_acc,
        "sub_acc":         sub_acc,
        "main_class_acc":  main_class_acc,
        "sub_class_acc":   sub_class_acc,
        "confusion":       confusion,
        "mean_main_conf":  mean_main_conf,
        "mean_sub_conf":   mean_sub_conf,
        "main_preds":      main_preds,
        "sub_preds":       sub_preds,
        "main_labels":     main_labels,
        "sub_labels":      sub_labels,
    }


# ── REPORT ────────────────────────────────────────────────────────────────────

def print_report(results: dict, save_path: Path = None):
    """Print and optionally save evaluation report."""
    lines = []

    def log(line=""):
        lines.append(line)
        print(line)

    log("=" * 65)
    log(f"  EVALUATION REPORT — {results['split'].upper()} SET")
    log("=" * 65)
    log(f"  Total samples: {results['n_samples']}")
    log()

    # Overall accuracy
    log("── Overall Accuracy ──────────────────────────────────────")
    log(f"  Main class accuracy:  {results['main_acc']:.2f}%")
    log(f"  Subclass accuracy:    {results['sub_acc']:.2f}%")
    log(f"  Mean main confidence: {results['mean_main_conf']:.3f}")
    log(f"  Mean sub confidence:  {results['mean_sub_conf']:.3f}")
    log()

    # Per main class
    log("── Per Main Class Accuracy ───────────────────────────────")
    for name, acc in results["main_class_acc"].items():
        bar = "█" * int(acc / 5)
        log(f"  {name:<12} {acc:>6.2f}%  {bar}")
    log()

    # Per subclass
    log("── Per Subclass Accuracy ─────────────────────────────────")
    log(f"  {'Subclass':<20} {'Acc':>7}  {'Count':>6}  {'Main Class'}")
    log(f"  {'-'*20} {'-'*7}  {'-'*6}  {'-'*12}")
    for sub in cfg.SUBCLASSES:
        if sub in results["sub_class_acc"]:
            info = results["sub_class_acc"][sub]
            mc   = cfg.SUBCLASS_TO_MAIN[sub]
            bar  = "█" * int(info["acc"] / 10)
            log(f"  {sub:<20} {info['acc']:>6.2f}%  "
                f"{info['count']:>6}  {mc:<12}  {bar}")
    log()

    # Confusion matrix
    log("── Confusion Matrix (subclass level) ────────────────────")
    log("  Rows = True label, Cols = Predicted label")
    log()

    # Header
    short_names = [s[:6] for s in cfg.SUBCLASSES]
    header = "  " + " " * 21 + "  ".join(f"{n:>6}" for n in short_names)
    log(header)
    log("  " + "-" * (21 + 8 * cfg.NUM_SUBCLASSES))

    for i, true_name in enumerate(cfg.SUBCLASSES):
        row = results["confusion"][i]
        row_str = "  ".join(
            f"\033[92m{v:>6}\033[0m" if j == i else f"{v:>6}"
            for j, v in enumerate(row)
        )
        log(f"  {true_name:<20}  {row_str}")
    log()

    # Common confusions
    log("── Top Confusions ────────────────────────────────────────")
    confusions = []
    for i in range(cfg.NUM_SUBCLASSES):
        for j in range(cfg.NUM_SUBCLASSES):
            if i != j and results["confusion"][i][j] > 0:
                confusions.append((
                    results["confusion"][i][j],
                    cfg.IDX_TO_SUB[i],
                    cfg.IDX_TO_SUB[j]
                ))
    confusions.sort(reverse=True)
    for count, true_name, pred_name in confusions[:10]:
        log(f"  {true_name:<20} → predicted as {pred_name:<20}  ({count} times)")
    log()
    log("=" * 65)

    # Save to file
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            for line in lines:
                # Strip ANSI colour codes for file
                clean = line.replace("\033[92m", "").replace("\033[0m", "")
                f.write(clean + "\n")
        print(f"\n  Report saved to: {save_path}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained Acoustic CRNN")
    parser.add_argument("--checkpoint", type=str,
                        default=str(cfg.CHECKPOINT_DIR / cfg.CHECKPOINT_NAME),
                        help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Which split to evaluate")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found at {ckpt_path}")
        print(f"  Run python pipeline/train.py first")
        exit(1)

    # Load model
    print(f"\nLoading model from: {ckpt_path}")
    model = AcousticCRNN().to(cfg.DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=cfg.DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded checkpoint from epoch {ckpt['epoch']}")
    print(f"  Val sub accuracy at save: {ckpt['metrics']['sub_acc']:.2f}%")

    # Load data
    train_loader, val_loader, test_loader, _ = get_dataloaders()
    loaders = {
        "train": train_loader,
        "val":   val_loader,
        "test":  test_loader,
    }
    loader = loaders[args.split]

    # Evaluate
    print(f"\nEvaluating on {args.split} set...")
    results = evaluate(model, loader, split_name=args.split)

    # Report
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
    report_path = cfg.LOG_DIR / f"evaluation_{args.split}.txt"
    print_report(results, save_path=report_path)
