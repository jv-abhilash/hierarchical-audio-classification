"""
model.py
---------
3-Class Hierarchical Acoustic CRNN Architecture.

Input:  mel spectrogram tensor (batch, 1, 128, 216)
Output: two classification heads
          Head 1 → main class logits  (batch, 3)   wildlife/nature/urban
          Head 2 → subclass logits    (batch, 12)  crow/owl/.../jackhammer

Architecture:
  CNN Block × 4  → spatial feature extraction from spectrogram
  GRU (bidir)    → temporal pattern learning across time frames
  Head 1         → main class prediction
  Head 2         → subclass prediction

Flow:
  (B, 1, 128, 216)
      ↓ CNN Block 1: Conv→BN→ReLU→Pool → (B, 32,  64, 108)
      ↓ CNN Block 2: Conv→BN→ReLU→Pool → (B, 64,  32,  54)
      ↓ CNN Block 3: Conv→BN→ReLU→Pool → (B, 128, 16,  27)
      ↓ CNN Block 4: Conv→BN→ReLU→Pool → (B, 256,  8,  13)
      ↓ Reshape for RNN               → (B, 13, 2048)
        (13 time steps, each 256×8=2048 features)
      ↓ Bidirectional GRU × 2         → (B, 13, 512)
      ↓ Take last time step            → (B, 512)
      ↓ Head 1 FC layers               → (B, 3)
      ↓ Head 2 FC layers               → (B, 12)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import config as cfg


# ── CNN BLOCK ─────────────────────────────────────────────────────────────────

class CNNBlock(nn.Module):
    """
    Single CNN block:
      Conv2d → BatchNorm2d → ReLU → MaxPool2d → Dropout

    BatchNorm: normalises activations — stabilises training,
               allows higher learning rates
    ReLU:      non-linearity — model learns complex patterns
    MaxPool:   downsamples by 2×2 — reduces spatial size,
               increases receptive field
    Dropout:   randomly zeros activations — prevents overfitting
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: tuple = (3, 3),
                 pool_size: tuple   = (2, 2),
                 dropout: float     = 0.2):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size = kernel_size,
            padding     = 1,        # same padding — keeps spatial size
            bias        = False,    # no bias when using BatchNorm
        )
        self.bn      = nn.BatchNorm2d(out_channels)
        self.pool    = nn.MaxPool2d(pool_size)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


# ── CLASSIFICATION HEAD ───────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    Two-layer FC classification head:
      Linear → ReLU → Dropout → Linear → (logits)

    No softmax here — applied in loss function (CrossEntropyLoss
    combines log_softmax + NLLLoss internally for numerical stability)
    """

    def __init__(self, in_features: int, hidden: int,
                 num_classes: int, dropout: float = 0.4):
        super().__init__()

        self.fc1     = nn.Linear(in_features, hidden)
        self.bn      = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2     = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ── CRNN MODEL ────────────────────────────────────────────────────────────────

class AcousticCRNN(nn.Module):
    """
    Hierarchical Convolutional Recurrent Neural Network
    for 3-class acoustic sound classification.

    Two output heads:
      head1 → main class (wildlife / nature / urban)
      head2 → subclass   (crow / owl / ... / jackhammer)

    Both heads trained simultaneously with combined loss.
    """

    def __init__(self):
        super().__init__()

        # ── CNN ENCODER ───────────────────────────────────────────────────────
        # Builds 4 CNN blocks from channel config
        # cfg.CNN_CHANNELS = [1, 32, 64, 128, 256]
        # Block 1: 1  → 32  channels
        # Block 2: 32 → 64  channels
        # Block 3: 64 → 128 channels
        # Block 4: 128→ 256 channels

        self.cnn_blocks = nn.ModuleList([
            CNNBlock(
                in_channels  = cfg.CNN_CHANNELS[i],
                out_channels = cfg.CNN_CHANNELS[i + 1],
                kernel_size  = cfg.CNN_KERNEL,
                pool_size    = cfg.CNN_POOL,
                dropout      = cfg.CNN_DROPOUT,
            )
            for i in range(len(cfg.CNN_CHANNELS) - 1)
        ])

        # Calculate CNN output size
        # Input: (1, 128, 216)
        # After 4 × MaxPool(2,2):
        #   Frequency: 128 → 64 → 32 → 16 → 8
        #   Time:      216 → 108 → 54 → 27 → 13
        # Output: (256, 8, 13)
        self.cnn_freq_out = cfg.N_MELS    // (2 ** 4)   # 128 // 16 = 8
        self.cnn_time_out = cfg.TIME_FRAMES // (2 ** 4)  # 216 // 16 = 13
        self.cnn_out_ch   = cfg.CNN_CHANNELS[-1]          # 256

        # RNN input size = channels × frequency bins per time step
        rnn_input = self.cnn_out_ch * self.cnn_freq_out  # 256 × 8 = 2048

        # ── RNN ENCODER ───────────────────────────────────────────────────────
        # Bidirectional GRU processes the time sequence
        # Input:  (batch, time_steps, features) = (B, 13, 2048)
        # Output: (batch, time_steps, hidden×2)  = (B, 13, 512)
        # We take the last time step: (B, 512)

        self.rnn = nn.GRU(
            input_size    = rnn_input,
            hidden_size   = cfg.RNN_HIDDEN_SIZE,
            num_layers    = cfg.RNN_LAYERS,
            batch_first   = True,           # (batch, seq, features)
            bidirectional = cfg.RNN_BIDIRECTIONAL,
            dropout       = cfg.RNN_DROPOUT if cfg.RNN_LAYERS > 1 else 0,
        )

        # RNN output size
        rnn_out = cfg.RNN_HIDDEN_SIZE * (2 if cfg.RNN_BIDIRECTIONAL else 1)  # 512

        # ── CLASSIFICATION HEADS ──────────────────────────────────────────────

        # Head 1 — main class (wildlife / nature / urban)
        self.head_main = ClassificationHead(
            in_features = rnn_out,
            hidden      = cfg.HEAD_HIDDEN,
            num_classes = cfg.NUM_MAIN,
            dropout     = cfg.HEAD_DROPOUT,
        )

        # Head 2 — subclass (crow / owl / ... / jackhammer)
        self.head_sub = ClassificationHead(
            in_features = rnn_out,
            hidden      = cfg.HEAD_HIDDEN,
            num_classes = cfg.NUM_SUBCLASSES,
            dropout     = cfg.HEAD_DROPOUT,
        )

        # Initialise weights
        self._init_weights()

    def _init_weights(self):
        """
        Kaiming initialisation for conv layers — better than random
        for ReLU networks. Prevents vanishing/exploding gradients.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.

        Args:
            x: mel spectrogram tensor (batch, 1, 128, 216)

        Returns:
            main_logits: (batch, 3)   — raw scores for main class
            sub_logits:  (batch, 12)  — raw scores for subclass
        """
        batch_size = x.shape[0]

        # ── CNN FORWARD ───────────────────────────────────────────────────────
        # (B, 1, 128, 216) → (B, 256, 8, 13)
        for block in self.cnn_blocks:
            x = block(x)

        # ── RESHAPE FOR RNN ───────────────────────────────────────────────────
        # (B, 256, 8, 13) → (B, 13, 256×8) = (B, 13, 2048)
        # Permute: (B, C, F, T) → (B, T, C, F)
        x = x.permute(0, 3, 1, 2)
        # Flatten C and F into one feature dimension
        x = x.reshape(batch_size, self.cnn_time_out,
                       self.cnn_out_ch * self.cnn_freq_out)

        # ── RNN FORWARD ───────────────────────────────────────────────────────
        # (B, 13, 2048) → output (B, 13, 512), hidden (layers×2, B, 256)
        rnn_out, _ = self.rnn(x)

        # Take last time step — contains summary of full temporal sequence
        # (B, 13, 512) → (B, 512)
        features = rnn_out[:, -1, :]

        # ── CLASSIFICATION HEADS ──────────────────────────────────────────────
        main_logits = self.head_main(features)   # (B, 3)
        sub_logits  = self.head_sub(features)    # (B, 12)

        return main_logits, sub_logits

    def predict(self, x: torch.Tensor,
                threshold: float = cfg.OOD_CONFIDENCE_THRESHOLD) -> dict:
        """
        Inference method — returns predicted classes with confidence.
        Includes OOD detection via confidence threshold.

        Args:
            x:         mel spectrogram tensor (batch, 1, 128, 216)
            threshold: confidence threshold for OOD rejection

        Returns:
            dict with keys:
              main_class:     predicted main class name
              subclass:       predicted subclass name
              main_conf:      confidence score for main class (0-1)
              sub_conf:       confidence score for subclass (0-1)
              main_probs:     full probability distribution (3,)
              sub_probs:      full probability distribution (12,)
              is_ood:         True if confidence below threshold
        """
        self.eval()
        with torch.no_grad():
            main_logits, sub_logits = self.forward(x)

            main_probs = F.softmax(main_logits, dim=-1)
            sub_probs  = F.softmax(sub_logits,  dim=-1)

            main_conf, main_idx = main_probs.max(dim=-1)
            sub_conf,  sub_idx  = sub_probs.max(dim=-1)

            # OOD detection — check both heads
            is_ood = (main_conf < threshold) | (sub_conf < threshold)

            results = []
            for i in range(x.shape[0]):
                results.append({
                    "main_class": cfg.IDX_TO_MAIN[main_idx[i].item()],
                    "subclass":   cfg.IDX_TO_SUB[sub_idx[i].item()],
                    "main_conf":  main_conf[i].item(),
                    "sub_conf":   sub_conf[i].item(),
                    "main_probs": main_probs[i].cpu().numpy(),
                    "sub_probs":  sub_probs[i].cpu().numpy(),
                    "is_ood":     is_ood[i].item(),
                })

            return results[0] if x.shape[0] == 1 else results


# ── MODEL SUMMARY ─────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module):
    """Print model architecture summary."""
    print("\n── Model Summary ──────────────────────────────────────")
    print(f"  Architecture: Hierarchical CRNN")
    print(f"  Input shape:  (batch, 1, {cfg.N_MELS}, {cfg.TIME_FRAMES})")
    print()

    print(f"  CNN Encoder:")
    for i, block in enumerate(model.cnn_blocks):
        ch_in  = cfg.CNN_CHANNELS[i]
        ch_out = cfg.CNN_CHANNELS[i + 1]
        print(f"    Block {i+1}: Conv2d({ch_in}→{ch_out}) "
              f"→ BN → ReLU → MaxPool(2×2) → Dropout({cfg.CNN_DROPOUT})")
    print(f"    Output: (batch, {cfg.CNN_CHANNELS[-1]}, "
          f"{model.cnn_freq_out}, {model.cnn_time_out})")

    print(f"\n  RNN Encoder:")
    rnn_in  = model.cnn_out_ch * model.cnn_freq_out
    rnn_out = cfg.RNN_HIDDEN_SIZE * (2 if cfg.RNN_BIDIRECTIONAL else 1)
    print(f"    Input:  (batch, {model.cnn_time_out}, {rnn_in})")
    print(f"    GRU:    hidden={cfg.RNN_HIDDEN_SIZE}, "
          f"layers={cfg.RNN_LAYERS}, "
          f"bidirectional={cfg.RNN_BIDIRECTIONAL}")
    print(f"    Output: (batch, {rnn_out})  [last time step]")

    print(f"\n  Classification Heads:")
    print(f"    Head 1 (main):  {rnn_out} → {cfg.HEAD_HIDDEN} → {cfg.NUM_MAIN}")
    print(f"    Head 2 (sub):   {rnn_out} → {cfg.HEAD_HIDDEN} → {cfg.NUM_SUBCLASSES}")

    total = count_parameters(model)
    print(f"\n  Total parameters: {total:,}")
    print(f"  Model size:       ~{total * 4 / 1024**2:.1f} MB (float32)")
    print("─" * 55)


# ── QUICK TEST ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run this to verify model architecture before training.

    Usage:
        python pipeline/model.py
    """
    print("Testing model architecture...")
    cfg.print_config()

    # Build model
    model = AcousticCRNN().to(cfg.DEVICE)
    model_summary(model)

    # Forward pass with random input
    print("\n── Forward pass test ──")
    batch_size = 4
    x = torch.randn(batch_size, 1, cfg.N_MELS,
                    cfg.TIME_FRAMES).to(cfg.DEVICE)
    print(f"  Input shape:  {x.shape}")

    model.train()
    main_logits, sub_logits = model(x)
    print(f"  Main logits:  {main_logits.shape}  "
          f"(expected: [{batch_size}, {cfg.NUM_MAIN}])")
    print(f"  Sub logits:   {sub_logits.shape}  "
          f"(expected: [{batch_size}, {cfg.NUM_SUBCLASSES}])")

    # Check probabilities sum to 1
    main_probs = torch.softmax(main_logits, dim=-1)
    sub_probs  = torch.softmax(sub_logits,  dim=-1)
    print(f"\n  Main probs sum: {main_probs.sum(dim=-1)}")
    print(f"  Sub probs sum:  {sub_probs.sum(dim=-1)}")

    # Test predict method
    print("\n── Predict method test ──")
    model.eval()
    x_single = torch.randn(1, 1, cfg.N_MELS,
                           cfg.TIME_FRAMES).to(cfg.DEVICE)
    result = model.predict(x_single)
    print(f"  Predicted main:  {result['main_class']} "
          f"(conf: {result['main_conf']:.3f})")
    print(f"  Predicted sub:   {result['subclass']} "
          f"(conf: {result['sub_conf']:.3f})")
    print(f"  Is OOD:          {result['is_ood']}")

    # Memory usage
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1024**2
        print(f"\n  GPU memory used: {mem:.1f} MB")

    print(f"\n✓ Model working correctly")
    print(f"  Ready to proceed to train.py")
