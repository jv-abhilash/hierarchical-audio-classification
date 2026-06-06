"""
Complete EDA for audio classification dataset
Checks 1–8 + Final Summary
All outputs saved to ./eda/
"""

import csv
import os
import random
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

# ── PATHS ────────────────────────────────────────────────────────────────────
BASE = Path("/home/abhi-ubuntu-pc/Datascience, ML and DL/data_science/dataset")
META = BASE / "main_metadata.csv"
EDA  = BASE / "eda"
EDA.mkdir(exist_ok=True)

# Source → folder mapping
SOURCE_FOLDER = {
    "ESC-50":           BASE / "ESC-50-master" / "audio",
    "ESC-50-aug":       BASE / "ESC-50-master" / "audio",
    "FSD50K":           BASE / "fsd50k" / "sliced_audio",
    "BirdCLEF2021":     BASE / "birdclef_2021" / "sliced_audio",
    "AnuraSet":         BASE / "anuraset" / "sliced_audio",
    "UrbanSound8K":     BASE / "urbansound8k" / "sliced_audio",
    "freefield1010":    BASE / "freefield1010" / "sliced_audio",
    "forest_wild_fire": BASE / "forest_wild_fire_sound_dataset" / "sliced_audio",
}

# Colour map for main_class
MAIN_CLASS_COLORS = {
    "wildlife": "#2ca02c",  # green
    "nature":   "#1f77b4",  # blue
    "urban":    "#ff7f0e",  # orange
}

# ── LOAD METADATA ────────────────────────────────────────────────────────────
print("Loading metadata...")
df = pd.read_csv(META)
print(f"  Total rows: {len(df)}")
print(f"  Columns: {list(df.columns)}")
print(f"  Subclasses: {sorted(df['subclass'].unique())}")
print(f"  Sources: {sorted(df['source'].unique())}")

SUBCLASSES = sorted(df["subclass"].unique())


def resolve_path(row):
    """Resolve the full path of an audio file from metadata row."""
    src = row["source"]
    fn = row["filename"]
    folder = SOURCE_FOLDER.get(src)
    if folder is None:
        return None
    p = folder / fn
    return p if p.exists() else None


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 1 — Class balance
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CHECK 1 — Class balance")
print("=" * 70)

sub_counts = df["subclass"].value_counts().sort_index()
main_counts = df["main_class"].value_counts().sort_index()

print("\nClips per subclass:")
for s in SUBCLASSES:
    print(f"  {s:<20} {sub_counts.get(s, 0):>6}")

print("\nClips per main_class:")
for mc, cnt in main_counts.items():
    print(f"  {mc:<20} {cnt:>6}")

# Plot
fig, ax = plt.subplots(figsize=(16, 7))
sub_sorted = df.groupby("subclass").size().sort_values(ascending=False)
colors = [MAIN_CLASS_COLORS.get(
    df[df["subclass"] == s]["main_class"].iloc[0], "gray"
) for s in sub_sorted.index]

bars = ax.bar(range(len(sub_sorted)), sub_sorted.values, color=colors, edgecolor="black", linewidth=0.5)
ax.set_xticks(range(len(sub_sorted)))
ax.set_xticklabels(sub_sorted.index, rotation=45, ha="right", fontsize=11)
ax.set_ylabel("Clip count", fontsize=12)
ax.set_title("CHECK 1 — Class Balance (clips per subclass)", fontsize=14, fontweight="bold")
ax.axhline(y=500, color="red", linestyle="--", linewidth=2, label="Target = 500")

# Add count labels on bars
for bar, val in zip(bars, sub_sorted.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
            str(val), ha="center", va="bottom", fontsize=9, fontweight="bold")

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#2ca02c", label="wildlife"),
    Patch(facecolor="#1f77b4", label="nature"),
    Patch(facecolor="#ff7f0e", label="urban"),
    Patch(facecolor="red", label="target (500)"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=11)
plt.tight_layout()
plt.savefig(EDA / "01_class_balance.png", dpi=150)
plt.close()
print("  Saved: eda/01_class_balance.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 2 & 3 — Duration & Sample rate verification (shared sampling)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CHECK 2 — Duration verification")
print("CHECK 3 — Sample rate verification")
print("=" * 70)

SAMPLE_N = 200
durations = []
sample_rates = []
dur_records = []  # (subclass, filename, duration, sr)

for sub in SUBCLASSES:
    sub_df = df[df["subclass"] == sub]
    n = min(SAMPLE_N, len(sub_df))
    sampled = sub_df.sample(n=n, random_state=42)
    found = 0
    for _, row in sampled.iterrows():
        fp = resolve_path(row)
        if fp is None:
            continue
        try:
            info = sf.info(str(fp))
            durations.append(info.duration)
            sample_rates.append(info.samplerate)
            dur_records.append((sub, row["filename"], info.duration, info.samplerate))
            found += 1
        except Exception as e:
            pass
    print(f"  {sub:<20} sampled {found}/{n} clips")

dur_arr = np.array(durations)
sr_arr = np.array(sample_rates)

print(f"\n  Duration stats (n={len(dur_arr)}):")
print(f"    min:  {dur_arr.min():.4f}s")
print(f"    max:  {dur_arr.max():.4f}s")
print(f"    mean: {dur_arr.mean():.4f}s")
print(f"    std:  {dur_arr.std():.4f}s")
not_5s = np.sum(np.abs(dur_arr - 5.0) > 0.1)
print(f"    Clips NOT 5.0s (±0.1s tolerance): {not_5s}")

# CHECK 2 plot — duration histogram
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(dur_arr, bins=100, color="#4a86e8", edgecolor="black", linewidth=0.3)
ax.axvline(x=5.0, color="red", linestyle="--", linewidth=2, label="5.0s target")
ax.axvline(x=4.9, color="orange", linestyle=":", linewidth=1.5, label="±0.1s tolerance")
ax.axvline(x=5.1, color="orange", linestyle=":", linewidth=1.5)
ax.set_xlabel("Duration (seconds)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title(f"CHECK 2 — Duration Distribution (n={len(dur_arr)}, not-5s={not_5s})", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(EDA / "02_duration_histogram.png", dpi=150)
plt.close()
print("  Saved: eda/02_duration_histogram.png")

# CHECK 3 — Sample rate
sr_counts = Counter(sr_arr)
print(f"\n  Sample rate distribution:")
for sr, cnt in sorted(sr_counts.items()):
    flag = "" if sr in (22050, 44100) else " *** UNEXPECTED ***"
    print(f"    {int(sr)} Hz: {cnt} clips{flag}")

flagged_sr = sum(cnt for sr, cnt in sr_counts.items() if sr not in (22050, 44100))
print(f"    Clips NOT at 22050 or 44100 Hz: {flagged_sr}")

fig, ax = plt.subplots(figsize=(10, 5))
sr_labels = [str(int(k)) for k in sorted(sr_counts.keys())]
sr_values = [sr_counts[k] for k in sorted(sr_counts.keys())]
bar_colors = ["#2ca02c" if int(k) in (22050, 44100) else "#e74c3c" for k in sorted(sr_counts.keys())]
ax.bar(sr_labels, sr_values, color=bar_colors, edgecolor="black", linewidth=0.5)
for i, (lbl, val) in enumerate(zip(sr_labels, sr_values)):
    ax.text(i, val + 10, str(val), ha="center", fontsize=10, fontweight="bold")
ax.set_xlabel("Sample Rate (Hz)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title(f"CHECK 3 — Sample Rate Distribution (unexpected={flagged_sr})", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(EDA / "03_sample_rate_distribution.png", dpi=150)
plt.close()
print("  Saved: eda/03_sample_rate_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 4 — Mel-spectrogram visual inspection
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CHECK 4 — Mel-spectrogram visual inspection")
print("=" * 70)

N_SPEC = 3  # clips per subclass
fig, axes = plt.subplots(len(SUBCLASSES), N_SPEC, figsize=(5 * N_SPEC, 3.5 * len(SUBCLASSES)))
if len(SUBCLASSES) == 1:
    axes = axes.reshape(1, -1)

for i, sub in enumerate(SUBCLASSES):
    sub_df = df[df["subclass"] == sub]
    sampled = sub_df.sample(n=min(N_SPEC * 5, len(sub_df)), random_state=42 + i)
    loaded = 0
    for _, row in sampled.iterrows():
        if loaded >= N_SPEC:
            break
        fp = resolve_path(row)
        if fp is None:
            continue
        try:
            y, sr = librosa.load(str(fp), sr=22050, mono=True)
            S = librosa.feature.melspectrogram(y=y, sr=22050, n_mels=128, hop_length=512, n_fft=2048)
            S_dB = librosa.power_to_db(S, ref=np.max)

            ax = axes[i, loaded]
            img = librosa.display.specshow(S_dB, sr=22050, hop_length=512,
                                           x_axis="time", y_axis="mel", ax=ax, cmap="magma")
            ax.set_title(f"{sub} | {row['source']}\n{row['filename']}", fontsize=8)
            ax.set_xlabel("")
            ax.set_ylabel("")
            loaded += 1
        except Exception as e:
            pass

    # blank out unused subplots
    for j in range(loaded, N_SPEC):
        axes[i, j].axis("off")

    print(f"  {sub:<20} plotted {loaded}/{N_SPEC} spectrograms")

plt.suptitle("CHECK 4 — Mel Spectrograms (3 per subclass)", fontsize=16, fontweight="bold", y=1.0)
plt.tight_layout()
plt.savefig(EDA / "04_mel_spectrograms.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: eda/04_mel_spectrograms.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 5 — RMS energy distribution
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CHECK 5 — RMS energy distribution")
print("=" * 70)

rms_data = []  # (subclass, filename, rms)
SILENT_THRESH = 0.001

for sub in SUBCLASSES:
    sub_df = df[df["subclass"] == sub]
    n = min(SAMPLE_N, len(sub_df))
    sampled = sub_df.sample(n=n, random_state=42)
    found = 0
    for _, row in sampled.iterrows():
        fp = resolve_path(row)
        if fp is None:
            continue
        try:
            y, sr = librosa.load(str(fp), sr=22050, mono=True)
            rms_val = float(np.sqrt(np.mean(y ** 2)))
            rms_data.append((sub, row["filename"], rms_val))
            found += 1
        except Exception:
            pass
    print(f"  {sub:<20} loaded {found}/{n} clips")

rms_df = pd.DataFrame(rms_data, columns=["subclass", "filename", "rms"])

# Count silent
silent_per_sub = rms_df[rms_df["rms"] < SILENT_THRESH].groupby("subclass").size()
print("\n  Silent clips (RMS < 0.001) per subclass (sampled):")
if len(silent_per_sub) == 0:
    print("    None found in sample")
else:
    for sub, cnt in silent_per_sub.items():
        print(f"    {sub:<20} {cnt}")

# Box plot sorted by median
median_order = rms_df.groupby("subclass")["rms"].median().sort_values().index.tolist()

fig, ax = plt.subplots(figsize=(16, 7))
sns.boxplot(data=rms_df, x="subclass", y="rms", order=median_order, ax=ax,
            palette="viridis", fliersize=2)
ax.axhline(y=SILENT_THRESH, color="red", linestyle="--", linewidth=1.5, label=f"Silent threshold ({SILENT_THRESH})")
ax.set_xlabel("Subclass (sorted by median RMS)", fontsize=12)
ax.set_ylabel("Mean RMS Energy", fontsize=12)
ax.set_title("CHECK 5 — RMS Energy Distribution (200 clips/subclass)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(EDA / "05_rms_energy.png", dpi=150)
plt.close()
print("  Saved: eda/05_rms_energy.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 6 — Source distribution per subclass
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CHECK 6 — Source distribution per subclass")
print("=" * 70)

cross = pd.crosstab(df["subclass"], df["source"])
cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100

print("\n  Source % per subclass:")
for sub in SUBCLASSES:
    row = cross_pct.loc[sub]
    parts = [f"{src}={row[src]:.1f}%" for src in row.index if row[src] > 0]
    print(f"  {sub:<20} {', '.join(parts)}")

# Flag dominant sources
print("\n  Subclasses where one source > 85%:")
dominant = []
for sub in SUBCLASSES:
    row = cross_pct.loc[sub]
    max_src = row.idxmax()
    max_pct = row.max()
    if max_pct > 85:
        dominant.append((sub, max_src, max_pct))
        print(f"    {sub:<20} {max_src} = {max_pct:.1f}%")
if not dominant:
    print("    None")

# Stacked bar chart
fig, ax = plt.subplots(figsize=(16, 7))
cross_pct_sorted = cross_pct.loc[SUBCLASSES]
sources = cross_pct_sorted.columns.tolist()
source_cmap = plt.cm.get_cmap("tab10", len(sources))
bottom = np.zeros(len(SUBCLASSES))

for j, src in enumerate(sources):
    vals = cross_pct_sorted[src].values
    ax.bar(range(len(SUBCLASSES)), vals, bottom=bottom,
           label=src, color=source_cmap(j), edgecolor="white", linewidth=0.3)
    bottom += vals

ax.set_xticks(range(len(SUBCLASSES)))
ax.set_xticklabels(SUBCLASSES, rotation=45, ha="right", fontsize=11)
ax.set_ylabel("Percentage (%)", fontsize=12)
ax.set_title("CHECK 6 — Source Distribution per Subclass", fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=9, ncol=2)
ax.axhline(y=85, color="red", linestyle="--", linewidth=1, alpha=0.7, label="85% dominance line")
plt.tight_layout()
plt.savefig(EDA / "06_source_distribution.png", dpi=150)
plt.close()
print("  Saved: eda/06_source_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 7 — Augmented vs original comparison
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CHECK 7 — Augmented vs original comparison")
print("=" * 70)

aug_df = df[df["source"] == "ESC-50-aug"]
aug_subclasses = sorted(aug_df["subclass"].unique())
print(f"  Augmented subclasses: {aug_subclasses}")

n_aug_subs = len(aug_subclasses)
fig, axes = plt.subplots(n_aug_subs, 2, figsize=(12, 3.5 * n_aug_subs))
if n_aug_subs == 1:
    axes = axes.reshape(1, -1)

for i, sub in enumerate(aug_subclasses):
    # Find an augmented clip and its original pair
    sub_aug = aug_df[aug_df["subclass"] == sub].sort_values("filename")
    paired = False
    for _, aug_row in sub_aug.iterrows():
        aug_fn = aug_row["filename"]  # e.g., crow_001_aug.wav
        orig_fn = aug_fn.replace("_aug.wav", ".wav")  # e.g., crow_001.wav
        orig_row = df[(df["filename"] == orig_fn) & (df["source"] == "ESC-50")]
        if len(orig_row) == 0:
            continue

        orig_row = orig_row.iloc[0]
        orig_path = resolve_path(orig_row)
        aug_path = resolve_path(aug_row)

        if orig_path is None or aug_path is None:
            continue

        try:
            y_orig, _ = librosa.load(str(orig_path), sr=22050, mono=True)
            y_aug, _ = librosa.load(str(aug_path), sr=22050, mono=True)

            S_orig = librosa.power_to_db(
                librosa.feature.melspectrogram(y=y_orig, sr=22050, n_mels=128, hop_length=512, n_fft=2048),
                ref=np.max)
            S_aug = librosa.power_to_db(
                librosa.feature.melspectrogram(y=y_aug, sr=22050, n_mels=128, hop_length=512, n_fft=2048),
                ref=np.max)

            librosa.display.specshow(S_orig, sr=22050, hop_length=512,
                                     x_axis="time", y_axis="mel", ax=axes[i, 0], cmap="magma")
            axes[i, 0].set_title(f"ORIGINAL: {orig_fn}\n({sub})", fontsize=9)

            librosa.display.specshow(S_aug, sr=22050, hop_length=512,
                                     x_axis="time", y_axis="mel", ax=axes[i, 1], cmap="magma")
            axes[i, 1].set_title(f"AUGMENTED: {aug_fn}\n({sub})", fontsize=9)

            paired = True
            print(f"  {sub:<20} {orig_fn} <-> {aug_fn}")
            break
        except Exception as e:
            pass

    if not paired:
        axes[i, 0].text(0.5, 0.5, f"No pair found for {sub}", transform=axes[i, 0].transAxes,
                         ha="center", va="center")
        axes[i, 0].axis("off")
        axes[i, 1].axis("off")
        print(f"  {sub:<20} NO PAIR FOUND")

plt.suptitle("CHECK 7 — Original vs Augmented Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(EDA / "07_augmentation_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: eda/07_augmentation_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 8 — Silent & clipped clip detection (ALL clips)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CHECK 8 — Silent & clipped clip detection (ALL clips)")
print("=" * 70)

SILENT_THRESH = 0.001
CLIP_THRESH = 0.999

flagged = []  # (filename, subclass, source, issue, rms, peak)
total_checked = 0
errors_c8 = 0

for idx, row in df.iterrows():
    fp = resolve_path(row)
    if fp is None:
        continue
    try:
        data, sr = sf.read(str(fp), dtype="float32")
        if data.ndim == 2:
            data = data.mean(axis=1)

        rms_val = float(np.sqrt(np.mean(data ** 2)))
        peak = float(np.max(np.abs(data)))

        issues = []
        if rms_val < SILENT_THRESH:
            issues.append("SILENT")
        if peak > CLIP_THRESH:
            issues.append("CLIPPED")

        if issues:
            flagged.append({
                "filename": row["filename"],
                "subclass": row["subclass"],
                "source": row["source"],
                "issue": "|".join(issues),
                "rms": round(rms_val, 6),
                "peak": round(peak, 6),
            })
        total_checked += 1
    except Exception:
        errors_c8 += 1

    if (idx + 1) % 5000 == 0:
        print(f"  Progress: {idx + 1}/{len(df)} clips checked...")

print(f"\n  Total clips checked: {total_checked}")
print(f"  Read errors: {errors_c8}")

flagged_df = pd.DataFrame(flagged)
if len(flagged_df) > 0:
    silent_df = flagged_df[flagged_df["issue"].str.contains("SILENT")]
    clipped_df = flagged_df[flagged_df["issue"].str.contains("CLIPPED")]
else:
    silent_df = pd.DataFrame()
    clipped_df = pd.DataFrame()

n_silent = len(silent_df)
n_clipped = len(clipped_df)

print(f"  Silent clips (RMS < {SILENT_THRESH}): {n_silent}")
if n_silent > 0:
    print("  Silent clips per subclass:")
    for sub, grp in silent_df.groupby("subclass"):
        print(f"    {sub:<20} {len(grp)} clips")
        for _, r in grp.iterrows():
            print(f"      {r['filename']}  (source={r['source']}, rms={r['rms']})")

print(f"  Clipped clips (peak > {CLIP_THRESH}): {n_clipped}")
if n_clipped > 0:
    print("  Clipped clips per subclass:")
    for sub, grp in clipped_df.groupby("subclass"):
        print(f"    {sub:<20} {len(grp)} clips")

# Save flagged clips CSV
if len(flagged_df) > 0:
    flagged_df.to_csv(EDA / "08_flagged_clips.csv", index=False)
    print(f"  Saved: eda/08_flagged_clips.csv ({len(flagged_df)} rows)")
else:
    # Write empty CSV with headers
    pd.DataFrame(columns=["filename", "subclass", "source", "issue", "rms", "peak"]).to_csv(
        EDA / "08_flagged_clips.csv", index=False)
    print("  Saved: eda/08_flagged_clips.csv (0 flagged clips)")


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

total_clips = len(df)
print(f"\n  Total clips in metadata: {total_clips}")

print(f"\n  Clips per subclass (sorted):")
print(f"  {'Subclass':<20} {'Count':>6}  {'Main class':<12}")
print(f"  {'-'*20} {'-'*6}  {'-'*12}")
for sub in df.groupby("subclass").size().sort_values(ascending=False).index:
    cnt = df[df["subclass"] == sub].shape[0]
    mc = df[df["subclass"] == sub]["main_class"].iloc[0]
    print(f"  {sub:<20} {cnt:>6}  {mc:<12}")

print(f"\n  Duration issues (not 5s ±0.1s): {not_5s} clips (from {len(dur_arr)} sampled)")
print(f"  Sample rate issues (not 22050/44100): {flagged_sr} clips (from {len(sr_arr)} sampled)")
print(f"  Silent clips found: {n_silent} (from {total_checked} checked)")
print(f"  Clipped clips found: {n_clipped} (from {total_checked} checked)")

print(f"\n  Subclasses where one source > 85%:")
if dominant:
    for sub, src, pct in dominant:
        print(f"    {sub:<20} {src} = {pct:.1f}%")
else:
    print("    None")

# Decision
silent_pct = (n_silent / total_checked * 100) if total_checked > 0 else 0
ready = (not_5s == 0) and (flagged_sr == 0) and (silent_pct < 5.0)
status = "YES" if ready else "NO"

reasons = []
if not_5s > 0:
    reasons.append(f"{not_5s} duration issues")
if flagged_sr > 0:
    reasons.append(f"{flagged_sr} sample rate issues")
if silent_pct >= 5.0:
    reasons.append(f"{silent_pct:.1f}% silent clips (>= 5%)")

print(f"\n  READY FOR TRAINING: {status}")
if not ready:
    print(f"    Reasons: {'; '.join(reasons)}")
else:
    print("    All checks passed!")

print("\n" + "=" * 70)
print("EDA COMPLETE — all outputs saved to ./eda/")
print("=" * 70)
