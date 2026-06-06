"""
BirdCLEF 2021 Filter + Convert + Slice + Metadata Script
----------------------------------------------------------
Filters BirdCLEF 2021 A-M species for crow and owl subclasses,
converts MP3 → WAV (22050Hz mono), slices to 5s clips,
and updates both local and global metadata + filename registry.

Species mapping (confirmed from audit):
  crow subclass → amecro (147) + fiscro (8) + comrav (889) = 1,044 files
  owl subclass  → grhowl (194) + brdowl (28)               =   222 files

Audio handling:
  - MP3 → WAV conversion during load (librosa handles all bitrates)
  - Resample all to 22050 Hz mono regardless of source SR
  - Slice to 5s non-overlapping windows
  - Discard remainder < 1s, zero-pad remainder 1s–5s
  - Stereo → mono by averaging channels

Naming:
  crow_116_s01.wav (continuing from crow_115 in registry)
  owl_001_s01.wav  (owl starts from 0 in registry)

Usage:
    python filter_birdclef.py --base ./birdclef_2021 --global-meta ./main_metadata.csv --registry ./filename_registry.csv
    python filter_birdclef.py --base ./birdclef_2021 --global-meta ./main_metadata.csv --registry ./filename_registry.csv --dry-run

Requirements:
    pip install librosa soundfile numpy pandas
"""

import csv
import argparse
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────

# ebird_code → subclass mapping
EBIRD_TO_SUBCLASS = {
    "amecro": "crow",    # American Crow
    "fiscro": "crow",    # Fish Crow
    "comrav": "crow",    # Common Raven — same family, similar acoustics
    "grhowl": "owl",     # Great Horned Owl
    "brdowl": "owl",     # Barred Owl
}

SUBCLASS_TO_MAIN = {
    "crow": "wildlife",
    "owl":  "wildlife",
}

ALL_SUBCLASSES = [
    "crow", "owl", "frog", "insects",
    "rain", "sea_waves", "wind", "crackling_fire",
    "car_horn", "engine_idling", "siren", "jackhammer",
]

TARGET_SR      = 22050
TARGET_DUR     = 5.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DUR)   # 110,250 samples
MIN_DUR        = 1.0
MIN_SAMPLES    = int(TARGET_SR * MIN_DUR)

GLOBAL_CSV_FIELDS = [
    "filename", "subclass", "main_class", "source",
    "original_filename", "sample_rate", "duration_s",
    "augmented", "split"
]

REGISTRY_FIELDS = [
    "subclass", "last_counter", "last_filename",
    "total_originals", "total_augmented", "last_source", "last_updated"
]

# ── REGISTRY ──────────────────────────────────────────────────────────────────

def load_registry(registry_path: Path) -> dict:
    registry = {}
    for sub in ALL_SUBCLASSES:
        registry[sub] = {
            "subclass": sub, "last_counter": 0, "last_filename": "—",
            "total_originals": 0, "total_augmented": 0,
            "last_source": "—", "last_updated": "—",
        }
    if registry_path.exists():
        with open(registry_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sub = row["subclass"]
                if sub in registry:
                    registry[sub] = {
                        "subclass":        sub,
                        "last_counter":    int(row["last_counter"]),
                        "last_filename":   row["last_filename"],
                        "total_originals": int(row["total_originals"]),
                        "total_augmented": int(row["total_augmented"]),
                        "last_source":     row["last_source"],
                        "last_updated":    row["last_updated"],
                    }
    return registry

def save_registry(registry: dict, registry_path: Path):
    with open(registry_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        for sub in ALL_SUBCLASSES:
            writer.writerow(registry[sub])

# ── SLICING ───────────────────────────────────────────────────────────────────

def slice_audio(y: np.ndarray) -> list:
    """Slice 1D array into TARGET_SAMPLES chunks. Pad last if >= MIN_SAMPLES."""
    chunks = []
    start  = 0
    total  = len(y)
    while start < total:
        segment = y[start:start + TARGET_SAMPLES]
        if len(segment) < MIN_SAMPLES:
            break
        if len(segment) < TARGET_SAMPLES:
            segment = np.pad(segment, (0, TARGET_SAMPLES - len(segment)))
        chunks.append(segment.astype(np.float32))
        start += TARGET_SAMPLES
    return chunks

# ── MAIN ──────────────────────────────────────────────────────────────────────

def run(base_path: Path, global_meta_path: Path, registry_path: Path, dry_run: bool):
    am_dir     = base_path / "A-M"
    csv_path   = base_path / "train_extended.csv"
    output_dir = base_path / "sliced_audio"
    local_meta = base_path / "birdclef_metadata.csv"

    if not am_dir.exists():
        print(f"ERROR: A-M/ folder not found at {am_dir}")
        return
    if not csv_path.exists():
        print(f"ERROR: train_extended.csv not found at {csv_path}")
        return

    if not dry_run:
        output_dir.mkdir(exist_ok=True)

    # ── STEP 1: Load registry ─────────────────────────────────────────────────
    print("\n── Step 1: Loading registry ──")
    registry = load_registry(registry_path)
    print(f"  {'Subclass':<20} {'Counter':>7}  {'Last file':<30}  {'Originals':>9}")
    print(f"  {'-'*20} {'-'*7}  {'-'*30}  {'-'*9}")
    for sub in ["crow", "owl"]:
        r = registry[sub]
        print(f"  {sub:<20} {r['last_counter']:>7}  {r['last_filename']:<30}  "
              f"{r['total_originals']:>9}")

    # ── STEP 2: Load train_extended.csv ───────────────────────────────────────
    print("\n── Step 2: Loading train_extended.csv ──")
    df = pd.read_csv(str(csv_path))
    print(f"  Total rows: {len(df)}")

    target_codes = list(EBIRD_TO_SUBCLASS.keys())
    target_df    = df[df["ebird_code"].isin(target_codes)].copy()
    print(f"  Rows for target species: {len(target_df)}")

    for code in target_codes:
        sub   = EBIRD_TO_SUBCLASS[code]
        count = len(target_df[target_df["ebird_code"] == code])
        print(f"    {code:<10} → {sub:<6}  {count} rows in CSV")

    # ── STEP 3: Build file list ───────────────────────────────────────────────
    print("\n── Step 3: Scanning A-M/ for target species ──")
    file_list = []  # (mp3_path, ebird_code, subclass, csv_row)

    for code in target_codes:
        sub      = EBIRD_TO_SUBCLASS[code]
        code_dir = am_dir / code
        if not code_dir.exists():
            print(f"  WARNING: folder not found — {code_dir}")
            continue

        mp3s = sorted(code_dir.glob("*.mp3"))
        print(f"  {code:<10} → {sub:<6}  {len(mp3s)} MP3 files on disk")

        # Match to CSV rows for metadata
        code_csv = target_df[target_df["ebird_code"] == code]
        csv_by_fn = {}
        for _, row in code_csv.iterrows():
            fn = str(row["filename"])
            if not fn.endswith(".mp3"):
                fn = fn + ".mp3"
            csv_by_fn[fn] = row

        for mp3 in mp3s:
            csv_row = csv_by_fn.get(mp3.name, None)
            file_list.append((mp3, code, sub, csv_row))

    print(f"\n  Total MP3 files to process: {len(file_list)}")

    # ── STEP 4: Estimate slice counts ─────────────────────────────────────────
    print("\n── Step 4: Estimating slice counts (from CSV duration column) ──")
    est_slices = defaultdict(int)
    est_files  = defaultdict(int)

    for mp3_path, code, sub, csv_row in file_list:
        est_files[sub] += 1
        if csv_row is not None and "duration" in csv_row:
            try:
                dur = float(csv_row["duration"])
                n   = max(1, int(dur / TARGET_DUR))
                est_slices[sub] += n
            except (ValueError, TypeError):
                est_slices[sub] += 5  # assume ~25s average

    for sub in ["crow", "owl"]:
        reg_start = registry[sub]["last_counter"]
        print(f"  {sub:<6}  source files: {est_files[sub]:>4}  "
              f"estimated slices: {est_slices[sub]:>6}  "
              f"registry starts at: {reg_start}")

    # ── STEP 5: Process files — convert, slice, write ─────────────────────────
    print(f"\n── Step 5: {'[DRY RUN] ' if dry_run else ''}Converting + Slicing ──")

    counters      = {sub: registry[sub]["last_counter"] for sub in ALL_SUBCLASSES}
    local_rows    = []
    global_rows   = []
    stats         = defaultdict(lambda: {"files": 0, "slices": 0,
                                          "discarded": 0, "errors": 0})
    processed     = 0

    for mp3_path, code, sub, csv_row in file_list:
        stats[sub]["files"] += 1

        if dry_run:
            # Estimate from CSV duration
            dur = 25.0
            if csv_row is not None:
                try:
                    dur = float(csv_row.get("duration", 25.0))
                except (ValueError, TypeError):
                    dur = 25.0
            n_slices = max(1, int(dur / TARGET_DUR))
            for i in range(n_slices):
                counters[sub] += 1
                slice_fn = f"{sub}_{counters[sub]:03d}_s{i+1:02d}.wav"
                stats[sub]["slices"] += 1
                local_rows.append({
                    "filename":        slice_fn,
                    "subclass":        sub,
                    "main_class":      SUBCLASS_TO_MAIN[sub],
                    "ebird_code":      code,
                    "source_file":     mp3_path.name,
                    "slice_index":     i + 1,
                    "sample_rate":     TARGET_SR,
                    "duration_s":      TARGET_DUR,
                })
            continue

        # Real run
        try:
            y, sr = librosa.load(str(mp3_path), sr=TARGET_SR, mono=True)
        except Exception as e:
            print(f"  ERROR loading {mp3_path.name}: {e}")
            stats[sub]["errors"] += 1
            continue

        if len(y) < MIN_SAMPLES:
            stats[sub]["discarded"] += 1
            continue

        chunks = slice_audio(y)
        if not chunks:
            stats[sub]["discarded"] += 1
            continue

        # Write each slice
        for i, chunk in enumerate(chunks):
            counters[sub] += 1
            slice_fn  = f"{sub}_{counters[sub]:03d}_s{i+1:02d}.wav"
            out_path  = output_dir / slice_fn

            if not out_path.exists():
                sf.write(str(out_path), chunk, TARGET_SR, subtype="PCM_16")

            stats[sub]["slices"] += 1

            local_rows.append({
                "filename":    slice_fn,
                "subclass":    sub,
                "main_class":  SUBCLASS_TO_MAIN[sub],
                "ebird_code":  code,
                "source_file": mp3_path.name,
                "slice_index": i + 1,
                "sample_rate": TARGET_SR,
                "duration_s":  TARGET_DUR,
            })

            global_rows.append({
                "filename":          slice_fn,
                "subclass":          sub,
                "main_class":        SUBCLASS_TO_MAIN[sub],
                "source":            "BirdCLEF2021",
                "original_filename": mp3_path.name,
                "sample_rate":       TARGET_SR,
                "duration_s":        TARGET_DUR,
                "augmented":         "no",
                "split":             "",
            })

        processed += 1
        if processed % 50 == 0:
            print(f"  Progress: {processed}/{len(file_list)} files  "
                  f"crow slices: {stats['crow']['slices']}  "
                  f"owl slices: {stats['owl']['slices']}")

    tag = "[DRY RUN] " if dry_run else ""
    print(f"\n  {tag}Files processed: {processed if not dry_run else len(file_list)}")
    print(f"  {tag}crow slices: {stats['crow']['slices']}")
    print(f"  {tag}owl  slices: {stats['owl']['slices']}")
    print(f"  {tag}Errors: {stats['crow']['errors'] + stats['owl']['errors']}")

    # ── STEP 6: Write local birdclef_metadata.csv ─────────────────────────────
    print(f"\n── Step 6: {'[DRY RUN] ' if dry_run else ''}Writing birdclef_metadata.csv ──")
    local_fields = ["filename", "subclass", "main_class", "ebird_code",
                    "source_file", "slice_index", "sample_rate", "duration_s"]

    if not dry_run:
        with open(local_meta, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=local_fields)
            writer.writeheader()
            writer.writerows(local_rows)
        print(f"  Written {len(local_rows)} rows → {local_meta.name}")
    else:
        print(f"  [DRY RUN] Would write ~{sum(s['slices'] for s in stats.values())} rows")

    # ── STEP 7: Append to global main_metadata.csv ────────────────────────────
    print(f"\n── Step 7: {'[DRY RUN] ' if dry_run else ''}Updating global main_metadata.csv ──")

    if not dry_run:
        existing_fns = set()
        if global_meta_path.exists():
            with open(global_meta_path, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    existing_fns.add(r["filename"])

        new_rows = [r for r in global_rows if r["filename"] not in existing_fns]

        write_header = not global_meta_path.exists()
        with open(global_meta_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=GLOBAL_CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerows(new_rows)

        total = sum(1 for _ in open(global_meta_path)) - 1
        print(f"  Appended {len(new_rows)} rows — global total now: {total}")
    else:
        total_est = sum(s["slices"] for s in stats.values())
        print(f"  [DRY RUN] Would append ~{total_est} rows")

    # ── STEP 8: Update registry ───────────────────────────────────────────────
    print(f"\n── Step 8: {'[DRY RUN] ' if dry_run else ''}Updating registry ──")

    for sub in ["crow", "owl"]:
        if stats[sub]["slices"] > 0:
            registry[sub]["last_counter"]    = counters[sub]
            registry[sub]["last_filename"]   = f"{sub}_{counters[sub]:03d}_s01.wav"
            registry[sub]["total_originals"] = (registry[sub]["total_originals"]
                                                + stats[sub]["slices"])
            registry[sub]["last_source"]     = "BirdCLEF2021"
            registry[sub]["last_updated"]    = datetime.now().strftime("%Y-%m-%d %H:%M")

    if not dry_run:
        save_registry(registry, registry_path)
        print(f"  Registry saved")

    print(f"\n  {'Subclass':<6}  {'Counter':>7}  {'New slices':>10}  {'Total originals':>15}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*10}  {'-'*15}")
    for sub in ["crow", "owl"]:
        print(f"  {sub:<6}  {counters[sub]:>7}  "
              f"{stats[sub]['slices']:>10}  "
              f"{registry[sub]['total_originals']:>15}")

    # ── STEP 9: Final summary ──────────────────────────────────────────────────
    print("\n── Final Summary ──")
    for sub in ["crow", "owl"]:
        s  = stats[sub]
        mc = SUBCLASS_TO_MAIN[sub]
        print(f"  {mc:<12} {sub:<6}  "
              f"files: {s['files']:>4}  "
              f"slices: {s['slices']:>6}  "
              f"discarded: {s['discarded']:>3}  "
              f"errors: {s['errors']:>3}")

    print()
    print("  BirdCLEF processing complete.")
    print("  Next: anuraset (frog subclass)")

    if dry_run:
        print("\n  [DRY RUN — no files modified. Run without --dry-run to apply.]")
        print("  Note: dry-run slice counts are estimates from CSV duration column")
        print("  Actual counts may differ slightly after real MP3 loading")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter BirdCLEF 2021 for crow + owl")
    parser.add_argument("--base",        required=True, help="Path to birdclef_2021 folder")
    parser.add_argument("--global-meta", required=True, help="Path to global main_metadata.csv")
    parser.add_argument("--registry",    required=True, help="Path to filename_registry.csv")
    parser.add_argument("--dry-run",     action="store_true")
    args = parser.parse_args()

    base        = Path(args.base)
    global_meta = Path(args.global_meta)
    registry    = Path(args.registry)

    if not base.exists():
        print(f"ERROR: Path does not exist: {base}")
        exit(1)

    run(base, global_meta, registry, args.dry_run)
