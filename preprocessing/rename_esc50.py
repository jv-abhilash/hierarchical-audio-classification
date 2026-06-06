"""
ESC-50 Rename + Metadata Script
---------------------------------
1. Renames all 400 kept WAV files from ESC-50 naming convention
   (e.g. 1-103298-A-9.wav) to subclass-sequential style
   (e.g. crow_001.wav, crow_002.wav ...)

2. Updates local meta/esc50.csv with new filenames

3. Appends to a global main_metadata.csv in the parent datasets folder
   Format: filename, subclass, main_class, source, original_filename, sample_rate, duration_s, split

Usage:
    python rename_esc50.py --base /path/to/ESC-50-master --global-meta /path/to/datasets/main_metadata.csv
    python rename_esc50.py --base /path/to/ESC-50-master --global-meta /path/to/datasets/main_metadata.csv --dry-run
"""

import os
import csv
import shutil
import argparse
import soundfile as sf
from pathlib import Path
from collections import defaultdict

# ── CONFIG ───────────────────────────────────────────────────────────────────

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

# ── HELPERS ──────────────────────────────────────────────────────────────────

def get_audio_info(path: Path):
    try:
        info = sf.info(str(path))
        return int(info.samplerate), round(info.duration, 3)
    except Exception:
        return None, None

def global_csv_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0

# ── MAIN ─────────────────────────────────────────────────────────────────────

def run(base_path: Path, global_meta_path: Path, dry_run: bool):
    audio_dir    = base_path / "audio"
    local_csv    = base_path / "meta" / "esc50.csv"
    local_backup = base_path / "meta" / "esc50_before_rename.csv"

    if not audio_dir.exists():
        print(f"ERROR: audio/ not found at {audio_dir}")
        return
    if not local_csv.exists():
        print(f"ERROR: esc50.csv not found at {local_csv}")
        return

    # ── STEP 1: Read filtered local CSV ──────────────────────────────────────
    print("\n── Step 1: Reading filtered esc50.csv ──")
    rows = []
    with open(local_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"  Rows loaded: {len(rows)}")

    # Group by subclass, sort within each group for deterministic numbering
    by_subclass = defaultdict(list)
    for row in rows:
        by_subclass[row["subclass"]].append(row)
    for sub in by_subclass:
        by_subclass[sub].sort(key=lambda r: r["filename"])

    # ── STEP 2: Build rename plan ─────────────────────────────────────────────
    print("\n── Step 2: Building rename plan ──")
    rename_plan = []  # (old_path, new_path, new_filename, row)

    counters = defaultdict(int)
    for sub in sorted(by_subclass.keys()):
        for row in by_subclass[sub]:
            counters[sub] += 1
            old_filename = row["filename"]
            new_filename = f"{sub}_{counters[sub]:03d}.wav"
            old_path     = audio_dir / old_filename
            new_path     = audio_dir / new_filename
            rename_plan.append((old_path, new_path, new_filename, old_filename, row))

    print(f"  Total files to rename: {len(rename_plan)}")
    print("\n  Sample renames (first 3 per subclass):")
    shown = defaultdict(int)
    for old_path, new_path, new_fn, old_fn, row in rename_plan:
        sub = row["subclass"]
        if shown[sub] < 3:
            print(f"    {old_fn:<35} → {new_fn}")
            shown[sub] += 1

    # ── STEP 3: Rename files ──────────────────────────────────────────────────
    print(f"\n── Step 3: {'[DRY RUN] ' if dry_run else ''}Renaming files ──")
    renamed     = 0
    not_found   = 0
    conflicts   = 0

    for old_path, new_path, new_fn, old_fn, row in rename_plan:
        if not old_path.exists():
            print(f"  WARNING: not found — {old_fn}")
            not_found += 1
            continue
        if new_path.exists() and old_path != new_path:
            print(f"  WARNING: target already exists — {new_fn}, skipping")
            conflicts += 1
            continue
        if not dry_run:
            old_path.rename(new_path)
        renamed += 1

    tag = "[DRY RUN] " if dry_run else ""
    print(f"  {tag}Renamed:   {renamed}")
    print(f"  {tag}Not found: {not_found}")
    print(f"  {tag}Conflicts: {conflicts}")

    if not dry_run:
        remaining = list(audio_dir.glob("*.wav"))
        print(f"  WAV files in audio/ after rename: {len(remaining)}")

    # ── STEP 4: Update local CSV ──────────────────────────────────────────────
    print(f"\n── Step 4: {'[DRY RUN] ' if dry_run else ''}Updating local esc50.csv ──")

    # Build lookup: old_filename → new_filename
    rename_lookup = {old_fn: new_fn for _, _, new_fn, old_fn, _ in rename_plan}

    updated_rows = []
    for _, _, new_fn, old_fn, row in rename_plan:
        updated_row = dict(row)
        updated_row["original_filename"] = old_fn
        updated_row["filename"]          = new_fn
        updated_rows.append(updated_row)

    if not dry_run:
        # Backup before modifying
        shutil.copy(local_csv, local_backup)
        print(f"  Backed up to: {local_backup.name}")

        with open(local_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOCAL_CSV_FIELDS)
            writer.writeheader()
            for row in updated_rows:
                writer.writerow({k: row.get(k, "") for k in LOCAL_CSV_FIELDS})
        print(f"  Updated esc50.csv with {len(updated_rows)} rows")
        print(f"  Added column: original_filename")
    else:
        print(f"  [DRY RUN] Would update {len(updated_rows)} rows in esc50.csv")

    # ── STEP 5: Append to global main_metadata.csv ───────────────────────────
    print(f"\n── Step 5: {'[DRY RUN] ' if dry_run else ''}Updating global main_metadata.csv ──")

    global_rows = []
    for _, new_path, new_fn, old_fn, row in rename_plan:
        # Get audio info from renamed file (or estimate)
        if not dry_run and new_path.exists():
            sr, dur = get_audio_info(new_path)
        else:
            sr, dur = 44100, 5.0  # ESC-50 is always 44100Hz, 5s

        global_row = {
            "filename":          new_fn,
            "subclass":          row["subclass"],
            "main_class":        SUBCLASS_TO_MAIN[row["subclass"]],
            "source":            "ESC-50",
            "original_filename": old_fn,
            "sample_rate":       sr if sr else 44100,
            "duration_s":        dur if dur else 5.0,
            "augmented":         "no",
            "split":             "",   # filled later during train/val/test split
        }
        global_rows.append(global_row)

    if not dry_run:
        file_exists = global_csv_exists(global_meta_path)
        mode = "a" if file_exists else "w"
        write_header = not file_exists

        # Check for duplicates if appending
        if file_exists:
            existing_fns = set()
            with open(global_meta_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    existing_fns.add(r["filename"])
            before = len(global_rows)
            global_rows = [r for r in global_rows if r["filename"] not in existing_fns]
            skipped = before - len(global_rows)
            if skipped:
                print(f"  Skipped {skipped} duplicate entries already in global CSV")

        with open(global_meta_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=GLOBAL_CSV_FIELDS)
            if write_header:
                writer.writeheader()
                print(f"  Created new global CSV: {global_meta_path}")
            writer.writerows(global_rows)
            print(f"  Appended {len(global_rows)} rows to main_metadata.csv")

        # Print global CSV stats
        total_global = 0
        with open(global_meta_path, newline="", encoding="utf-8") as f:
            total_global = sum(1 for _ in f) - 1  # subtract header
        print(f"  Global CSV total rows now: {total_global}")
    else:
        print(f"  [DRY RUN] Would append {len(global_rows)} rows to main_metadata.csv")

    # ── STEP 6: Final summary ─────────────────────────────────────────────────
    print("\n── Final Summary ──")
    sub_counts = defaultdict(int)
    for _, _, new_fn, _, row in rename_plan:
        sub_counts[row["subclass"]] += 1
    for sub in sorted(sub_counts):
        mc = SUBCLASS_TO_MAIN[sub]
        sample = f"{sub}_001.wav ... {sub}_{sub_counts[sub]:03d}.wav"
        print(f"  {mc:<12} {sub:<20} {sub_counts[sub]:>3} clips   {sample}")

    print()
    print("  ESC-50 rename complete.")
    print("  Next step: run augment_esc50.py to double each subclass 40 → 80 clips")
    if dry_run:
        print("\n  [DRY RUN — no changes made. Run without --dry-run to apply.]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename ESC-50 files and update metadata")
    parser.add_argument("--base",        required=True, help="Path to ESC-50-master folder")
    parser.add_argument("--global-meta", required=True, help="Path to global main_metadata.csv")
    parser.add_argument("--dry-run",     action="store_true")
    args = parser.parse_args()

    base        = Path(args.base)
    global_meta = Path(args.global_meta)

    if not base.exists():
        print(f"ERROR: Base path does not exist: {base}")
        exit(1)

    global_meta.parent.mkdir(parents=True, exist_ok=True)
    run(base, global_meta, args.dry_run)
