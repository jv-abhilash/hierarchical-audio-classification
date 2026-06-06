"""
RMS Normalisation Script
--------------------------
Normalises all WAV files in final_dataset/ so every clip
has mean RMS energy = TARGET_RMS (0.1).

Formula:
    rms          = sqrt(mean(y^2))
    y_normalised = y / (rms + epsilon) * TARGET_RMS
    epsilon      = 1e-8  (prevents division by zero)
    TARGET_RMS   = 0.1

What this fixes:
    - Quiet field recordings (insects, crow from BirdCLEF) normalised up
    - Loud studio recordings (siren, car_horn from ESC-50) normalised down
    - All 5,905 clips end up at same loudness level
    - CNN focuses on spectral pattern not amplitude

What is preserved:
    - Frequency content    (mel bands unchanged)
    - Temporal pattern     (energy over time unchanged)
    - Harmonic structure   (pitch relationships unchanged)
    - Spectral texture     (fine-grained patterns unchanged)

In-place processing — overwrites WAV files with normalised versions.
Saves a normalisation log to final_dataset/normalisation_log.csv

Usage:
    python normalise_rms.py --base /path/to/datasets --dry-run
    python normalise_rms.py --base /path/to/datasets

Requirements:
    pip install soundfile numpy
"""

import csv
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────

TARGET_RMS  = 0.1
EPSILON     = 1e-8
TARGET_SR   = 22050
TARGET_SUBTYPE = "PCM_16"

SUBCLASS_FOLDERS = [
    "wildlife/crow", "wildlife/owl", "wildlife/frog", "wildlife/insects",
    "nature/rain", "nature/sea_waves", "nature/wind", "nature/crackling_fire",
    "urban/car_horn", "urban/engine_idling", "urban/siren", "urban/jackhammer",
]

# ── MAIN ──────────────────────────────────────────────────────────────────────

def run(base_path: Path, dry_run: bool):
    final_dir = base_path / "final_dataset"
    log_path  = final_dir / "normalisation_log.csv"

    if not final_dir.exists():
        print(f"ERROR: final_dataset/ not found at {final_dir}")
        return

    # ── STEP 1: Collect all WAV files ─────────────────────────────────────────
    print("\n── Step 1: Scanning final_dataset/ ──")
    all_wavs = []
    for folder_rel in SUBCLASS_FOLDERS:
        folder = final_dir / folder_rel
        if folder.exists():
            all_wavs.extend(sorted(folder.glob("*.wav")))
        else:
            print(f"  WARNING: {folder_rel} not found")

    print(f"  Total WAV files: {len(all_wavs)}")

    # ── STEP 2: Compute RMS stats before normalisation ─────────────────────────
    print("\n── Step 2: Computing RMS statistics before normalisation ──")
    rms_before = defaultdict(list)
    below_005  = defaultdict(int)
    below_001  = defaultdict(int)

    for wav in all_wavs:
        subclass = wav.parent.name
        try:
            y, _ = sf.read(str(wav), dtype="float32", always_2d=False)
            if y.ndim == 2: y = y.mean(axis=1)
            rms = float(np.sqrt(np.mean(y ** 2)))
            rms_before[subclass].append(rms)
            if rms < 0.005: below_005[subclass] += 1
            if rms < 0.001: below_001[subclass] += 1
        except Exception as e:
            print(f"  ERROR reading {wav.name}: {e}")

    print(f"\n  {'Subclass':<20} {'Mean RMS':>10}  {'<0.005':>8}  {'%':>6}")
    print(f"  {'-'*20} {'-'*10}  {'-'*8}  {'-'*6}")
    for sub in sorted(rms_before):
        vals    = rms_before[sub]
        mean    = np.mean(vals)
        n_low   = below_005[sub]
        pct     = n_low / len(vals) * 100
        flag    = " ← flagged" if pct > 5 else ""
        print(f"  {sub:<20} {mean:>10.4f}  {n_low:>8}  {pct:>5.1f}%{flag}")

    total_low = sum(below_005.values())
    print(f"\n  Total clips below 0.005 RMS: {total_low} "
          f"({total_low/len(all_wavs)*100:.1f}%)")

    # ── STEP 3: Normalise all clips ───────────────────────────────────────────
    print(f"\n── Step 3: {'[DRY RUN] ' if dry_run else ''}Normalising {len(all_wavs)} clips ──")
    print(f"  Target RMS: {TARGET_RMS}")

    processed  = 0
    skipped    = 0
    errors     = 0
    log_rows   = []
    rms_after  = defaultdict(list)

    for wav in all_wavs:
        subclass = wav.parent.name
        try:
            y, sr = sf.read(str(wav), dtype="float32", always_2d=False)
            if y.ndim == 2: y = y.mean(axis=1)

            rms_orig = float(np.sqrt(np.mean(y ** 2)))

            # Skip truly silent clips (no real content)
            if rms_orig < 1e-6:
                skipped += 1
                log_rows.append({
                    "filename": wav.name,
                    "subclass": subclass,
                    "rms_before": rms_orig,
                    "rms_after": rms_orig,
                    "scale_factor": 1.0,
                    "action": "skipped_silent"
                })
                continue

            # Normalise
            scale_factor = TARGET_RMS / (rms_orig + EPSILON)
            y_norm       = y * scale_factor

            # Clip to prevent overflow from aggressive scaling
            y_norm = np.clip(y_norm, -1.0, 1.0)

            rms_new = float(np.sqrt(np.mean(y_norm ** 2)))
            rms_after[subclass].append(rms_new)

            log_rows.append({
                "filename":    wav.name,
                "subclass":    subclass,
                "rms_before":  round(rms_orig, 6),
                "rms_after":   round(rms_new, 6),
                "scale_factor":round(scale_factor, 4),
                "action":      "normalised"
            })

            if not dry_run:
                sf.write(str(wav), y_norm, sr, subtype=TARGET_SUBTYPE)

            processed += 1

            if processed % 500 == 0 and not dry_run:
                print(f"  Progress: {processed}/{len(all_wavs)} normalised...")

        except Exception as e:
            print(f"  ERROR: {wav.name}: {e}")
            errors += 1

    tag = "[DRY RUN] " if dry_run else ""
    print(f"\n  {tag}Normalised: {processed}")
    print(f"  {tag}Skipped (silent): {skipped}")
    print(f"  {tag}Errors: {errors}")

    # ── STEP 4: RMS stats after normalisation ──────────────────────────────────
    if not dry_run and rms_after:
        print(f"\n── Step 4: RMS statistics after normalisation ──")
        print(f"\n  {'Subclass':<20} {'Mean RMS':>10}  {'Std':>8}  {'<0.005':>8}")
        print(f"  {'-'*20} {'-'*10}  {'-'*8}  {'-'*8}")
        for sub in sorted(rms_after):
            vals   = rms_after[sub]
            mean   = np.mean(vals)
            std    = np.std(vals)
            n_low  = sum(1 for v in vals if v < 0.005)
            print(f"  {sub:<20} {mean:>10.4f}  {std:>8.4f}  {n_low:>8}")

    # ── STEP 5: Save normalisation log ────────────────────────────────────────
    print(f"\n── Step 5: {'[DRY RUN] ' if dry_run else ''}Saving normalisation log ──")
    log_fields = ["filename", "subclass", "rms_before",
                  "rms_after", "scale_factor", "action"]

    if not dry_run:
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"  Saved {len(log_rows)} rows → normalisation_log.csv")
    else:
        print(f"  [DRY RUN] Would save {len(log_rows)} rows to normalisation_log.csv")

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    print(f"\n══ Summary ════════════════════════════════════════════════")
    print(f"  Total clips:      {len(all_wavs)}")
    print(f"  Normalised:       {processed}")
    print(f"  Skipped (silent): {skipped}")
    print(f"  Errors:           {errors}")
    print(f"  Target RMS:       {TARGET_RMS}")
    print()
    total_low_after = sum(
        1 for r in log_rows
        if r["action"] == "normalised" and float(r["rms_after"]) < 0.005
    )
    print(f"  Clips below 0.005 RMS after: {total_low_after} "
          f"(was {total_low} before)")
    print()

    if not dry_run:
        print(f"  All clips now at RMS ≈ {TARGET_RMS}")
        print(f"  Dataset ready for mel spectrogram generation and training")
    else:
        print(f"  [DRY RUN — no files modified]")
        print(f"  Run without --dry-run to apply normalisation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RMS normalise all clips in final_dataset/")
    parser.add_argument("--base",    required=True,
                        help="Path to datasets root (contains final_dataset/)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = Path(args.base)
    if not base.exists():
        print(f"ERROR: {base} does not exist")
        exit(1)

    run(base, args.dry_run)
