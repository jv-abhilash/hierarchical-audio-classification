"""
ESC-50 Filtering Script
------------------------
Keeps only the 10 categories needed for the 3-class acoustic project.
Deletes all other WAV files from audio/ folder.
Updates meta/esc50.csv to contain only kept rows.
Adds a 'subclass' column mapping ESC-50 categories to project subclass names.

Safe to run multiple times — checks before deleting.
Original CSV is backed up as meta/esc50_original_backup.csv before any changes.

Usage:
    python filter_esc50.py --base /path/to/ESC-50-master
    python filter_esc50.py --base /path/to/ESC-50-master --dry-run
"""

import os
import shutil
import csv
import argparse
from pathlib import Path
from collections import defaultdict

# ── CONFIG ──────────────────────────────────────────────────────────────────

# Categories to KEEP and their project subclass mapping
KEEP_CATEGORIES = {
    # Wildlife
    "crow":          "crow",
    "frog":          "frog",
    "insects":       "insects",
    "crickets":      "insects",    # merged into insects

    # Nature
    "rain":          "rain",
    "sea_waves":     "sea_waves",
    "wind":          "wind",
    "crackling_fire":"crackling_fire",

    # Urban
    "car_horn":      "car_horn",
    "siren":         "siren",
}

MAIN_CLASS_MAP = {
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

# ── MAIN ────────────────────────────────────────────────────────────────────

def run(base_path: Path, dry_run: bool):
    audio_dir = base_path / "audio"
    meta_dir  = base_path / "meta"
    csv_path  = meta_dir / "esc50.csv"
    backup_path = meta_dir / "esc50_original_backup.csv"
    output_csv  = meta_dir / "esc50.csv"

    if not audio_dir.exists():
        print(f"ERROR: audio/ folder not found at {audio_dir}")
        return
    if not csv_path.exists():
        print(f"ERROR: esc50.csv not found at {csv_path}")
        return

    # ── STEP 1: Read CSV ────────────────────────────────────────────────────
    print("\n── Step 1: Reading esc50.csv ──")
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    print(f"  Total rows in CSV: {len(rows)}")
    all_cats = defaultdict(int)
    for r in rows:
        all_cats[r["category"]] += 1
    print(f"  Total categories: {len(all_cats)}")

    # ── STEP 2: Separate keep vs remove ─────────────────────────────────────
    print("\n── Step 2: Classifying rows ──")
    keep_rows   = []
    remove_rows = []
    for row in rows:
        if row["category"] in KEEP_CATEGORIES:
            keep_rows.append(row)
        else:
            remove_rows.append(row)

    keep_cats   = defaultdict(int)
    remove_cats = defaultdict(int)
    for r in keep_rows:
        keep_cats[r["category"]] += 1
    for r in remove_rows:
        remove_cats[r["category"]] += 1

    print(f"\n  KEEP ({len(keep_rows)} files across {len(keep_cats)} categories):")
    for cat in sorted(keep_cats):
        subclass   = KEEP_CATEGORIES[cat]
        main_class = MAIN_CLASS_MAP[subclass]
        print(f"    {cat:<20} → subclass: {subclass:<20} main_class: {main_class}  ({keep_cats[cat]} clips)")

    print(f"\n  REMOVE ({len(remove_rows)} files across {len(remove_cats)} categories):")
    for cat in sorted(remove_cats):
        print(f"    {cat:<20} ({remove_cats[cat]} clips)")

    # ── STEP 3: Delete audio files ───────────────────────────────────────────
    print(f"\n── Step 3: {'[DRY RUN] ' if dry_run else ''}Deleting {len(remove_rows)} WAV files ──")
    deleted   = 0
    not_found = 0
    for row in remove_rows:
        wav_path = audio_dir / row["filename"]
        if wav_path.exists():
            if not dry_run:
                wav_path.unlink()
            deleted += 1
        else:
            not_found += 1

    if dry_run:
        print(f"  [DRY RUN] Would delete: {deleted} files")
        print(f"  [DRY RUN] Not found:    {not_found} files (already deleted)")
    else:
        print(f"  Deleted:   {deleted} files")
        print(f"  Not found: {not_found} files (already deleted)")

    # Verify audio folder
    remaining_wavs = list(audio_dir.glob("*.wav"))
    print(f"  WAV files remaining in audio/: {len(remaining_wavs)}")

    # ── STEP 4: Backup + update CSV ──────────────────────────────────────────
    print(f"\n── Step 4: {'[DRY RUN] ' if dry_run else ''}Updating esc50.csv ──")

    if not dry_run:
        # Backup original if not already done
        if not backup_path.exists():
            shutil.copy(csv_path, backup_path)
            print(f"  Backed up original CSV to: {backup_path.name}")
        else:
            print(f"  Backup already exists at: {backup_path.name} — skipping backup")

        # Write updated CSV with new columns
        new_fieldnames = list(fieldnames) + ["subclass", "main_class"]
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=new_fieldnames)
            writer.writeheader()
            for row in keep_rows:
                subclass   = KEEP_CATEGORIES[row["category"]]
                main_class = MAIN_CLASS_MAP[subclass]
                row["subclass"]   = subclass
                row["main_class"] = main_class
                writer.writerow(row)

        print(f"  Updated esc50.csv with {len(keep_rows)} rows")
        print(f"  New columns added: subclass, main_class")
    else:
        print(f"  [DRY RUN] Would write {len(keep_rows)} rows to esc50.csv")
        print(f"  [DRY RUN] Would add columns: subclass, main_class")

    # ── STEP 5: Final summary ────────────────────────────────────────────────
    print("\n── Summary ──")
    print(f"  Categories kept:   {len(keep_cats)}")
    print(f"  Categories removed:{len(remove_cats)}")
    print(f"  Files kept:        {len(keep_rows)}")
    print(f"  Files removed:     {deleted}")
    print()
    print("  Subclass breakdown after filtering:")
    subclass_counts = defaultdict(int)
    for row in keep_rows:
        subclass_counts[KEEP_CATEGORIES[row["category"]]] += 1
    for sub in sorted(subclass_counts):
        mc = MAIN_CLASS_MAP[sub]
        print(f"    {mc:<12} {sub:<20} {subclass_counts[sub]} clips")

    print()
    print("  Note: 'insects' has 80 clips (40 from 'insects' + 40 from 'crickets')")
    print("        All other subclasses have 40 clips each")
    print("        Next step: augment each to 80 clips before combining with other sources")
    if dry_run:
        print("\n  [DRY RUN complete — no files were modified]")
        print("  Run without --dry-run to apply changes")
    else:
        print("\n  ESC-50 filtering complete.")
        print("  Next: run augmentation script to double each subclass from 40 → 80 clips")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter ESC-50 dataset for acoustic project")
    parser.add_argument("--base", required=True, help="Path to ESC-50-master folder")
    parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")
    args = parser.parse_args()

    base = Path(args.base)
    if not base.exists():
        print(f"ERROR: Path does not exist: {base}")
        exit(1)

    run(base, args.dry_run)
