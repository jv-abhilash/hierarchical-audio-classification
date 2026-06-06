"""
ESC-50 Augmentation Script
----------------------------
Doubles each subclass from 40 → 80 clips (insects already 80, stays 80).
Each original clip gets exactly ONE augmentation variant.
Technique assigned per subclass based on spectral characteristics:

  crow          → pitch shift  -2 semitones  (lower caw variants)
  frog          → add noise    SNR=30dB       (field recording simulation)
  insects       → time stretch rate=0.9       (slower buzz/stridulation)
  rain          → time stretch rate=1.1       (faster/heavier rain)
  sea_waves     → pitch shift  +2 semitones   (higher frequency surf)
  wind          → add noise    SNR=30dB       (wind with ambient bleed)
  crackling_fire→ time stretch rate=0.9       (slower crackling)
  car_horn      → pitch shift  +2 semitones   (different horn pitch)
  siren         → time stretch rate=1.1       (faster siren cycle)

Output: crow_001_aug.wav alongside crow_001.wav in the same audio/ folder
Updates: local esc50.csv and appends to global main_metadata.csv

Usage:
    python augment_esc50.py --base /path/to/ESC-50-master --global-meta /path/to/main_metadata.csv
    python augment_esc50.py --base /path/to/ESC-50-master --global-meta /path/to/main_metadata.csv --dry-run
"""

import csv
import shutil
import argparse
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────

# Augmentation technique per subclass
# Format: (technique, param)
#   technique: "pitch_shift" | "time_stretch" | "add_noise"
#   param:
#     pitch_shift  → n_steps (semitones, float)
#     time_stretch → rate (float, >1 = faster, <1 = slower)
#     add_noise    → snr_db (float)

AUG_CONFIG = {
    "crow":           ("pitch_shift",  -2.0),
    "frog":           ("add_noise",    30.0),
    "insects":        ("time_stretch",  0.9),
    "rain":           ("time_stretch",  1.1),
    "sea_waves":      ("pitch_shift",  +2.0),
    "wind":           ("add_noise",    30.0),
    "crackling_fire": ("time_stretch",  0.9),
    "car_horn":       ("pitch_shift",  +2.0),
    "siren":          ("time_stretch",  1.1),
}

SUBCLASS_TO_MAIN = {
    "crow":           "wildlife",
    "frog":           "wildlife",
    "insects":        "wildlife",
    "rain":           "nature",
    "sea_waves":      "nature",
    "wind":           "nature",
    "crackling_fire": "nature",
    "car_horn":       "urban",
    "siren":          "urban",
}

TARGET_SR   = 44100   # ESC-50 native, keep as-is
TARGET_DUR  = 5.0     # seconds
TARGET_SAMPLES = int(TARGET_SR * TARGET_DUR)

GLOBAL_CSV_FIELDS = [
    "filename", "subclass", "main_class", "source",
    "original_filename", "sample_rate", "duration_s",
    "augmented", "split"
]

LOCAL_CSV_FIELDS = [
    "filename", "fold", "target", "category",
    "esc10", "src_file", "take",
    "subclass", "main_class", "original_filename"
]

# ── AUGMENTATION FUNCTIONS ───────────────────────────────────────────────────

def apply_pitch_shift(y: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Shift pitch by n_steps semitones without changing duration."""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def apply_time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """Stretch/compress time by rate without changing pitch."""
    stretched = librosa.effects.time_stretch(y, rate=rate)
    # Pad or trim to original length
    if len(stretched) < TARGET_SAMPLES:
        stretched = np.pad(stretched, (0, TARGET_SAMPLES - len(stretched)))
    else:
        stretched = stretched[:TARGET_SAMPLES]
    return stretched

def apply_add_noise(y: np.ndarray, snr_db: float) -> np.ndarray:
    """Add Gaussian noise at specified SNR in dB."""
    signal_power = np.mean(y ** 2)
    if signal_power == 0:
        return y
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(y))
    return np.clip(y + noise, -1.0, 1.0)

def augment(y: np.ndarray, sr: int, technique: str, param: float) -> np.ndarray:
    if technique == "pitch_shift":
        return apply_pitch_shift(y, sr, param)
    elif technique == "time_stretch":
        return apply_time_stretch(y, param)
    elif technique == "add_noise":
        return apply_add_noise(y, param)
    else:
        raise ValueError(f"Unknown technique: {technique}")

# ── MAIN ──────────────────────────────────────────────────────────────────────

def run(base_path: Path, global_meta_path: Path, dry_run: bool):
    audio_dir = base_path / "audio"
    local_csv = base_path / "meta" / "esc50.csv"

    if not audio_dir.exists():
        print(f"ERROR: audio/ not found at {audio_dir}")
        return
    if not local_csv.exists():
        print(f"ERROR: esc50.csv not found at {local_csv}")
        return

    # ── STEP 1: Read current local CSV ───────────────────────────────────────
    print("\n── Step 1: Reading esc50.csv ──")
    rows = []
    with open(local_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"  Rows loaded: {len(rows)}")

    # Only augment originals (not already augmented)
    orig_rows = [r for r in rows if "_aug" not in r["filename"]]
    print(f"  Original (non-aug) clips: {len(orig_rows)}")

    # Group by subclass
    by_subclass = defaultdict(list)
    for row in orig_rows:
        by_subclass[row["subclass"]].append(row)

    # ── STEP 2: Build augmentation plan ──────────────────────────────────────
    print("\n── Step 2: Augmentation plan ──")
    aug_plan = []  # (src_path, aug_path, aug_filename, technique, param, row)

    for sub in sorted(by_subclass.keys()):
        technique, param = AUG_CONFIG[sub]
        clips = by_subclass[sub]

        # insects already has 80 clips (40 insects + 40 crickets)
        # Still augment all 80 to reach 160? No — insects target is 80 total
        # so we skip augmentation for insects
        if sub == "insects":
            print(f"  {sub:<20} {len(clips):>3} clips — SKIP (already at 80, target met)")
            continue

        print(f"  {sub:<20} {len(clips):>3} clips → +{len(clips)} aug  "
              f"technique: {technique}({param})")
        for row in clips:
            orig_fn  = row["filename"]                         # crow_001.wav
            aug_fn   = orig_fn.replace(".wav", "_aug.wav")    # crow_001_aug.wav
            src_path = audio_dir / orig_fn
            aug_path = audio_dir / aug_fn
            aug_plan.append((src_path, aug_path, aug_fn, technique, param, row))

    print(f"\n  Total files to generate: {len(aug_plan)}")

    # ── STEP 3: Generate augmented files ─────────────────────────────────────
    print(f"\n── Step 3: {'[DRY RUN] ' if dry_run else ''}Generating augmented files ──")
    generated  = 0
    skipped    = 0
    errors     = 0
    new_rows   = []   # for CSV updates

    for src_path, aug_path, aug_fn, technique, param, row in aug_plan:
        if not src_path.exists():
            print(f"  WARNING: source not found — {src_path.name}")
            errors += 1
            continue

        if aug_path.exists():
            print(f"  SKIP (already exists): {aug_fn}")
            skipped += 1
            continue

        if not dry_run:
            try:
                # Load
                y, sr = librosa.load(str(src_path), sr=TARGET_SR, mono=True)

                # Augment
                y_aug = augment(y, sr, technique, param)

                # Ensure exact 5s length
                if len(y_aug) < TARGET_SAMPLES:
                    y_aug = np.pad(y_aug, (0, TARGET_SAMPLES - len(y_aug)))
                else:
                    y_aug = y_aug[:TARGET_SAMPLES]

                # Save
                sf.write(str(aug_path), y_aug, TARGET_SR, subtype="PCM_16")
                generated += 1

                # Build CSV row for this aug file
                aug_row = dict(row)
                aug_row["filename"]          = aug_fn
                aug_row["original_filename"] = row["filename"]
                new_rows.append(aug_row)

                if generated % 10 == 0:
                    print(f"  Progress: {generated}/{len(aug_plan)} generated...")

            except Exception as e:
                print(f"  ERROR generating {aug_fn}: {e}")
                errors += 1
        else:
            generated += 1
            aug_row = dict(row)
            aug_row["filename"]          = aug_fn
            aug_row["original_filename"] = row["filename"]
            new_rows.append(aug_row)

    tag = "[DRY RUN] " if dry_run else ""
    print(f"  {tag}Generated: {generated}")
    print(f"  {tag}Skipped (exists): {skipped}")
    print(f"  {tag}Errors: {errors}")

    if not dry_run:
        total_wavs = len(list(audio_dir.glob("*.wav")))
        print(f"  WAV files in audio/ now: {total_wavs}")

    # ── STEP 4: Update local CSV ──────────────────────────────────────────────
    print(f"\n── Step 4: {'[DRY RUN] ' if dry_run else ''}Updating local esc50.csv ──")

    all_rows = rows + new_rows
    all_rows.sort(key=lambda r: r["filename"])

    if not dry_run:
        shutil.copy(local_csv, local_csv.parent / "esc50_before_aug.csv")
        print(f"  Backed up to: esc50_before_aug.csv")
        with open(local_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOCAL_CSV_FIELDS)
            writer.writeheader()
            for row in all_rows:
                writer.writerow({k: row.get(k, "") for k in LOCAL_CSV_FIELDS})
        print(f"  Updated esc50.csv: {len(all_rows)} rows")
    else:
        print(f"  [DRY RUN] Would write {len(all_rows)} total rows to esc50.csv")

    # ── STEP 5: Append to global main_metadata.csv ───────────────────────────
    print(f"\n── Step 5: {'[DRY RUN] ' if dry_run else ''}Updating global main_metadata.csv ──")

    # Read existing filenames to avoid duplicates
    existing_fns = set()
    if global_meta_path.exists():
        with open(global_meta_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                existing_fns.add(r["filename"])

    global_new_rows = []
    for aug_row in new_rows:
        aug_fn = aug_row["filename"]
        if aug_fn in existing_fns:
            continue
        aug_path = audio_dir / aug_fn
        if not dry_run and aug_path.exists():
            info = sf.info(str(aug_path))
            sr   = int(info.samplerate)
            dur  = round(info.duration, 3)
        else:
            sr, dur = TARGET_SR, TARGET_DUR

        global_new_rows.append({
            "filename":          aug_fn,
            "subclass":          aug_row["subclass"],
            "main_class":        SUBCLASS_TO_MAIN[aug_row["subclass"]],
            "source":            "ESC-50-aug",
            "original_filename": aug_row["original_filename"],
            "sample_rate":       sr,
            "duration_s":        dur,
            "augmented":         "yes",
            "split":             "",
        })

    if not dry_run:
        with open(global_meta_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=GLOBAL_CSV_FIELDS)
            writer.writerows(global_new_rows)
        total_global = sum(1 for _ in open(global_meta_path)) - 1
        print(f"  Appended {len(global_new_rows)} aug rows to main_metadata.csv")
        print(f"  Global CSV total rows now: {total_global}")
    else:
        print(f"  [DRY RUN] Would append {len(global_new_rows)} rows to main_metadata.csv")

    # ── STEP 6: Final summary ─────────────────────────────────────────────────
    print("\n── Final Summary ──")
    final_counts = defaultdict(int)
    for r in all_rows:
        final_counts[r["subclass"]] += 1

    for sub in sorted(final_counts):
        mc    = SUBCLASS_TO_MAIN[sub]
        count = final_counts[sub]
        tech, param = AUG_CONFIG.get(sub, ("none", "-"))
        note  = f"({tech} {param})" if sub != "insects" else "(no aug needed)"
        print(f"  {mc:<12} {sub:<20} {count:>3} clips   {note}")

    total_clips = sum(final_counts.values())
    print(f"\n  Total ESC-50 clips after augmentation: {total_clips}")
    print(f"  (originals: {len(orig_rows)}  +  augmented: {len(new_rows)})")
    print()
    print("  ESC-50 augmentation complete.")
    print("  Next: process fsd50k — filter, rename, append to main_metadata.csv")

    if dry_run:
        print("\n  [DRY RUN — no files modified. Run without --dry-run to apply.]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment ESC-50 clips 40 → 80 per subclass")
    parser.add_argument("--base",        required=True, help="Path to ESC-50-master folder")
    parser.add_argument("--global-meta", required=True, help="Path to global main_metadata.csv")
    parser.add_argument("--dry-run",     action="store_true")
    args = parser.parse_args()

    np.random.seed(42)  # reproducibility

    base        = Path(args.base)
    global_meta = Path(args.global_meta)

    if not base.exists():
        print(f"ERROR: Path does not exist: {base}")
        exit(1)

    run(base, global_meta, args.dry_run)
