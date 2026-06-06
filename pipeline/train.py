"""
train.py
---------
Training loop for the 3-Class Hierarchical Acoustic CRNN.

What this file does:
  1. Loads train/val DataLoaders
  2. Builds model, optimiser, scheduler
  3. Trains for up to cfg.NUM_EPOCHS epochs
  4. Validates after every epoch
  5. Saves best checkpoint (by val subclass accuracy)
  6. Early stopping if val loss does not improve
  7. Logs all metrics to logs/training_log.csv

Loss function:
  total_loss = loss_main + cfg.LAMBDA_SUB * loss_sub
  Both use weighted CrossEntropyLoss (car_horn has fewer clips)

Mixed precision:
  Uses torch.cuda.amp for float16 computation on RTX 5070 Ti
  Doubles effective throughput, halves memory usage

Usage:
    python pipeline/train.py
    python pipeline/train.py --epochs 30
    python pipeline/train.py --resume checkpoints/best_crnn.pt
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

import sys
sys.path.append(str(Path(__file__).parent))
import config as cfg
from dataset import get_dataloaders
from model import AcousticCRNN, count_parameters, model_summary


# ── METRICS ───────────────────────────────────────────────────────────────────

class MetricsTracker:
    """Tracks loss and accuracy across one epoch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss  = 0.0
        self.main_loss   = 0.0
        self.sub_loss    = 0.0
        self.main_correct = 0
        self.sub_correct  = 0
        self.total        = 0
        self.n_batches    = 0

    def update(self, loss, main_loss, sub_loss,
               main_preds, sub_preds,
               main_labels, sub_labels):
        batch_size = main_labels.size(0)
        self.total_loss   += loss
        self.main_loss    += main_loss
        self.sub_loss     += sub_loss
        self.main_correct += (main_preds == main_labels).sum().item()
        self.sub_correct  += (sub_preds  == sub_labels).sum().item()
        self.total        += batch_size
        self.n_batches    += 1

    def compute(self) -> dict:
        return {
            "loss":       self.total_loss / self.n_batches,
            "main_loss":  self.main_loss  / self.n_batches,
            "sub_loss":   self.sub_loss   / self.n_batches,
            "main_acc":   self.main_correct / self.total * 100,
            "sub_acc":    self.sub_correct  / self.total * 100,
        }


# ── TRAIN ONE EPOCH ───────────────────────────────────────────────────────────

def train_epoch(model, loader, optimiser, scaler,
                criterion_main, criterion_sub) -> dict:
    """Run one full training epoch. Returns metrics dict."""
    model.train()
    tracker = MetricsTracker()

    for specs, main_labels, sub_labels in loader:
        specs        = specs.to(cfg.DEVICE, non_blocking=True)
        main_labels  = main_labels.to(cfg.DEVICE, non_blocking=True)
        sub_labels   = sub_labels.to(cfg.DEVICE, non_blocking=True)

        optimiser.zero_grad()

        # Mixed precision forward pass
        with autocast('cuda', enabled=cfg.USE_AMP):
            main_logits, sub_logits = model(specs)

            loss_main = criterion_main(main_logits, main_labels)
            loss_sub  = criterion_sub(sub_logits,  sub_labels)
            loss      = loss_main + cfg.LAMBDA_SUB * loss_sub

        # Scaled backward pass
        scaler.scale(loss).backward()
        # Gradient clipping — prevents exploding gradients in RNN
        scaler.unscale_(optimiser)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimiser)
        scaler.update()

        # Predictions (argmax of logits)
        main_preds = main_logits.argmax(dim=-1)
        sub_preds  = sub_logits.argmax(dim=-1)

        tracker.update(
            loss.item(), loss_main.item(), loss_sub.item(),
            main_preds, sub_preds, main_labels, sub_labels
        )

    return tracker.compute()


# ── VALIDATE ONE EPOCH ────────────────────────────────────────────────────────

@torch.no_grad()
def val_epoch(model, loader,
              criterion_main, criterion_sub) -> dict:
    """Run one full validation epoch. Returns metrics dict."""
    model.eval()
    tracker = MetricsTracker()

    for specs, main_labels, sub_labels in loader:
        specs       = specs.to(cfg.DEVICE, non_blocking=True)
        main_labels = main_labels.to(cfg.DEVICE, non_blocking=True)
        sub_labels  = sub_labels.to(cfg.DEVICE, non_blocking=True)

        with autocast('cuda', enabled=cfg.USE_AMP):
            main_logits, sub_logits = model(specs)
            loss_main = criterion_main(main_logits, main_labels)
            loss_sub  = criterion_sub(sub_logits,  sub_labels)
            loss      = loss_main + cfg.LAMBDA_SUB * loss_sub

        main_preds = main_logits.argmax(dim=-1)
        sub_preds  = sub_logits.argmax(dim=-1)

        tracker.update(
            loss.item(), loss_main.item(), loss_sub.item(),
            main_preds, sub_preds, main_labels, sub_labels
        )

    return tracker.compute()


# ── CHECKPOINT ────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimiser, scheduler, epoch,
                    metrics, path: Path):
    """Save model checkpoint with full training state."""
    torch.save({
        "epoch":          epoch,
        "model_state":    model.state_dict(),
        "optimiser_state":optimiser.state_dict(),
        "scheduler_state":scheduler.state_dict(),
        "metrics":        metrics,
    }, path)


def load_checkpoint(model, optimiser, scheduler, path: Path) -> int:
    """Load checkpoint. Returns epoch number to resume from."""
    ckpt = torch.load(path, map_location=cfg.DEVICE)
    model.load_state_dict(ckpt["model_state"])
    optimiser.load_state_dict(ckpt["optimiser_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    print(f"  Resumed from epoch {ckpt['epoch']} "
          f"(val sub acc: {ckpt['metrics']['sub_acc']:.2f}%)")
    return ckpt["epoch"]


# ── LOGGER ────────────────────────────────────────────────────────────────────

class CSVLogger:
    """Logs training metrics to CSV file after every epoch."""

    def __init__(self, path: Path):
        self.path = path
        self.fields = [
            "epoch",
            "train_loss", "train_main_loss", "train_sub_loss",
            "train_main_acc", "train_sub_acc",
            "val_loss", "val_main_loss", "val_sub_loss",
            "val_main_acc", "val_sub_acc",
            "lr", "epoch_time_s",
        ]
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()

    def log(self, epoch, train_m, val_m, lr, epoch_time):
        row = {
            "epoch":            epoch,
            "train_loss":       f"{train_m['loss']:.4f}",
            "train_main_loss":  f"{train_m['main_loss']:.4f}",
            "train_sub_loss":   f"{train_m['sub_loss']:.4f}",
            "train_main_acc":   f"{train_m['main_acc']:.2f}",
            "train_sub_acc":    f"{train_m['sub_acc']:.2f}",
            "val_loss":         f"{val_m['loss']:.4f}",
            "val_main_loss":    f"{val_m['main_loss']:.4f}",
            "val_sub_loss":     f"{val_m['sub_loss']:.4f}",
            "val_main_acc":     f"{val_m['main_acc']:.2f}",
            "val_sub_acc":      f"{val_m['sub_acc']:.2f}",
            "lr":               f"{lr:.2e}",
            "epoch_time_s":     f"{epoch_time:.1f}",
        }
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(row)


# ── MAIN TRAINING LOOP ────────────────────────────────────────────────────────

def train(num_epochs: int = cfg.NUM_EPOCHS,
          resume_path: str = None):
    """
    Main training function.

    Args:
        num_epochs:   number of epochs to train
        resume_path:  path to checkpoint to resume from (optional)
    """
    # Setup directories
    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)

    cfg.print_config()

    # ── DATA ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _, class_weights = get_dataloaders()

    # ── MODEL ─────────────────────────────────────────────────────────────────
    print("\nBuilding model...")
    model = AcousticCRNN().to(cfg.DEVICE)
    model_summary(model)
    print(f"  Parameters: {count_parameters(model):,}")

    # ── LOSS FUNCTIONS ────────────────────────────────────────────────────────
    # Weighted CrossEntropy — gives higher loss for car_horn errors
    # (car_horn has fewer clips so each error matters more)
    class_weights = class_weights.to(cfg.DEVICE)

    criterion_sub  = nn.CrossEntropyLoss(weight=class_weights)
    criterion_main = nn.CrossEntropyLoss()  # main classes are balanced

    # ── OPTIMISER ─────────────────────────────────────────────────────────────
    # AdamW — Adam with proper weight decay (L2 regularisation)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.LEARNING_RATE,
        weight_decay = cfg.WEIGHT_DECAY,
    )

    # ── LR SCHEDULER ──────────────────────────────────────────────────────────
    # ReduceLROnPlateau — halves LR when val loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode     = "min",           # monitor val loss (minimise)
        factor   = cfg.LR_FACTOR,
        patience = cfg.LR_PATIENCE,
        min_lr   = cfg.LR_MIN,
    )

    # ── MIXED PRECISION SCALER ────────────────────────────────────────────────
    scaler = GradScaler('cuda', enabled=cfg.USE_AMP)

    # ── RESUME FROM CHECKPOINT ────────────────────────────────────────────────
    start_epoch    = 0
    best_val_acc   = 0.0
    no_improve     = 0
    ckpt_path      = cfg.CHECKPOINT_DIR / cfg.CHECKPOINT_NAME

    if resume_path:
        start_epoch = load_checkpoint(
            model, optimiser, scheduler, Path(resume_path))

    # ── LOGGER ────────────────────────────────────────────────────────────────
    log_path = cfg.LOG_DIR / "training_log.csv"
    logger   = CSVLogger(log_path)
    print(f"\n  Logging to: {log_path}")
    print(f"  Checkpoints: {ckpt_path}")

    # ── TRAINING LOOP ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Starting training — {num_epochs} epochs")
    print(f"{'='*65}")
    print(f"  {'Ep':>4}  {'TrLoss':>8}  {'TrMain%':>8}  {'TrSub%':>7}  "
          f"{'VaLoss':>8}  {'VaMain%':>8}  {'VaSub%':>7}  {'LR':>8}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*7}  "
          f"{'-'*8}  {'-'*8}  {'-'*7}  {'-'*8}")

    for epoch in range(start_epoch + 1, num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimiser, scaler,
            criterion_main, criterion_sub
        )

        # Validate
        val_metrics = val_epoch(
            model, val_loader,
            criterion_main, criterion_sub
        )

        # LR scheduler step (based on val loss)
        scheduler.step(val_metrics["loss"])
        current_lr = optimiser.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start

        # Print epoch summary
        print(f"  {epoch:>4}  "
              f"{train_metrics['loss']:>8.4f}  "
              f"{train_metrics['main_acc']:>7.2f}%  "
              f"{train_metrics['sub_acc']:>6.2f}%  "
              f"{val_metrics['loss']:>8.4f}  "
              f"{val_metrics['main_acc']:>7.2f}%  "
              f"{val_metrics['sub_acc']:>6.2f}%  "
              f"{current_lr:>8.1e}  "
              f"[{epoch_time:.0f}s]")

        # Log to CSV
        logger.log(epoch, train_metrics, val_metrics,
                   current_lr, epoch_time)

        # Save best checkpoint
        if val_metrics["sub_acc"] > best_val_acc:
            best_val_acc = val_metrics["sub_acc"]
            save_checkpoint(model, optimiser, scheduler,
                            epoch, val_metrics, ckpt_path)
            print(f"  ✓ New best val sub accuracy: "
                  f"{best_val_acc:.2f}% — checkpoint saved")
            no_improve = 0
        else:
            no_improve += 1

        # Early stopping
        if no_improve >= cfg.EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {cfg.EARLY_STOP_PATIENCE} epochs)")
            break

    # ── TRAINING COMPLETE ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Training complete")
    print(f"  Best val subclass accuracy: {best_val_acc:.2f}%")
    print(f"  Checkpoint saved to: {ckpt_path}")
    print(f"  Training log: {log_path}")
    print(f"{'='*65}")
    print(f"\n  Next: run evaluate.py to get full test set metrics")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the Acoustic CRNN model")
    parser.add_argument("--epochs",  type=int, default=cfg.NUM_EPOCHS,
                        help=f"Number of epochs (default: {cfg.NUM_EPOCHS})")
    parser.add_argument("--resume",  type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(cfg.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.RANDOM_SEED)

    train(num_epochs=args.epochs, resume_path=args.resume)
