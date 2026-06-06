"""
Dataset Split Script
---------------------
Reads main_metadata.csv (which already has split column filled),
then organises final_dataset/ into:

  final_dataset/
      wildlife/crow/        ← TRAIN only (original names)
      wildlife/owl/         ← TRAIN only
      wildlife/frog/        ← TRAIN only
      wildlife/insects/     ← TRAIN only
      nature/rain/          ← TRAIN only
      nature/sea_waves/     ← TRAIN only
      nature/wind/          ← TRAIN only
      nature/crackling_fire/← TRAIN only
      urban/car_horn/       ← TRAIN only
      urban/engine_idling/  ← TRAIN only
      urban/siren/          ← TRAIN only
      urban/jackhammer/     ← TRAIN only
      validation/           ← VAL clips, renamed val_0001.wav ...
      test/                 ← TEST clips, renamed test_0001.wav ...
      train_metadata.csv    ← train rows only
      val_metadata.csv      ← val rows with new filenames
      test_metadata.csv     ← test rows with new filenames
      main_metadata.csv     ← UNTOUCHED — full reference

What this script does:
  1. Read main_metadata.csv — get all 5905 rows with split column
  2. Separate into train / val / test rows
  3. Verify source balance across splits (no source concentrated in one split)
  4. Create validation/ and test/ folders
  5. MOVE val clips from class folders → validation/ with new random names
  6. MOVE test clips from class folders → test/ with new random names
  7. Class folders now contain ONLY train clips
  8. Write train_metadata.csv, val_metadata.csv, test_metadata.csv

ESC-50 pair protection:
  aug pairs always in same split as their original (already done in create_final_dataset.py)

Usage:
    python split_dataset.py --final /path/to/final_dataset --dry-run
    python split_dataset.py --final /path/to/final_dataset
"""

import csv
import random
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────

MAIN_CLASS = {
    "crow":           "wildlife",
    "owl":            "wildlife",
    "frog":           "wildlife",
    "insects":        "wildlife",
    "rain":           "nature",
    "sea_waves":      "nature",
    "wind":           "nature",
    "crackling_fire": "nature",
    "car_horn":       "urban",
    "engine_idling":  "urban",
    "siren":          "urban",
    "jackhammer":     "urban",
}

SUBCLASS_ORDER = [
    "crow", "owl", "frog", "insects",
    "rain", "sea_waves", "wind", "crackling_fire",
    "car_horn", "engine_idling", "siren", "jackhammer"
]

CSV_FIELDS = [
    "filename", "subclass", "main_class", "source",
    "original_filename", "sample_rate", "duration_s",
    "augmented", "split"
]

RANDOM_SEED = 42

# ── HELPERS ───────────────────────────────────────────────────────────────────

def find_wav(filename: str, final_dir: Path) -> Path:
    """Search for a WAV file across all class subfolders."""
    for mc in ["wildlife", "nature", "urban"]:
        for sub in SUBCLASS_ORDER:
            p = final_dir / mc / sub / filename
            if p.exists():
                return p
    return None

# ── MAIN ──────────────────────────────────────────────────────────────────────

def run(final_dir: Path, dry_run: bool):
    metadata_csv = final_dir / "main_metadata.csv"

    if not metadata_csv.exists():
        print(f"ERROR: main_metadata.csv not found at {metadata_csv}")
        return

    random.seed(RANDOM_SEED)

    # ── STEP 1: Read main_metadata.csv ───────────────────────────────────────
    print("\n── Step 1: Reading main_metadata.csv ──")
    all_rows = []
    fieldnames = None

    with open(metadata_csv, newline="", encoding="utf-8") as f:
        reader    = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            all_rows.append(row)

    print(f"  Total rows: {len(all_rows)}")

    # Separate by split
    train_rows = [r for r in all_rows if r.get("split") == "train"]
    val_rows   = [r for r in all_rows if r.get("split") == "val"]
    test_rows  = [r for r in all_rows if r.get("split") == "test"]
    other_rows = [r for r in all_rows if r.get("split") not in ("train","val","test")]

    print(f"  Train: {len(train_rows)}")
    print(f"  Val:   {len(val_rows)}")
    print(f"  Test:  {len(test_rows)}")
    if other_rows:
        print(f"  WARNING: {len(other_rows)} rows with unknown split value")

    # ── STEP 2: Source balance verification ───────────────────────────────────
    print("\n── Step 2: Source balance verification ──")
    print(f"  {'Source':<25} {'Train':>6}  {'Val':>5}  {'Test':>5}  {'Train%':>7}")
    print(f"  {'-'*25} {'-'*6}  {'-'*5}  {'-'*5}  {'-'*7}")

    src_train = defaultdict(int)
    src_val   = defaultdict(int)
    src_test  = defaultdict(int)
    for r in train_rows: src_train[r["source"]] += 1
    for r in val_rows:   src_val[r["source"]]   += 1
    for r in test_rows:  src_test[r["source"]]  += 1

    all_sources = sorted(set(list(src_train) + list(src_val) + list(src_test)))
    for src in all_sources:
        total = src_train[src] + src_val[src] + src_test[src]
        pct   = src_train[src] / total * 100 if total > 0 else 0
        print(f"  {src:<25} {src_train[src]:>6}  {src_val[src]:>5}  "
              f"{src_test[src]:>5}  {pct:>6.1f}%")

    # ── STEP 3: Subclass balance verification ─────────────────────────────────
    print("\n── Step 3: Subclass balance per split ──")
    print(f"  {'Subclass':<20} {'Train':>6}  {'Val':>5}  {'Test':>5}  {'Total':>6}")
    print(f"  {'-'*20} {'-'*6}  {'-'*5}  {'-'*5}  {'-'*6}")

    sub_train = defaultdict(int)
    sub_val   = defaultdict(int)
    sub_test  = defaultdict(int)
    for r in train_rows: sub_train[r["subclass"]] += 1
    for r in val_rows:   sub_val[r["subclass"]]   += 1
    for r in test_rows:  sub_test[r["subclass"]]  += 1

    for sub in SUBCLASS_ORDER:
        total = sub_train[sub] + sub_val[sub] + sub_test[sub]
        print(f"  {sub:<20} {sub_train[sub]:>6}  {sub_val[sub]:>5}  "
              f"{sub_test[sub]:>5}  {total:>6}")

    # ── STEP 4: Create validation/ and test/ folders ──────────────────────────
    print(f"\n── Step 4: {'[DRY RUN] ' if dry_run else ''}Creating folders ──")
    val_dir  = final_dir / "validation"
    test_dir = final_dir / "test"

    if not dry_run:
        val_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)
        print(f"  Created: validation/")
        print(f"  Created: test/")
    else:
        print(f"  [DRY RUN] Would create: validation/")
        print(f"  [DRY RUN] Would create: test/")

    # ── STEP 5: Move val clips → validation/ with new names ──────────────────
    print(f"\n── Step 5: {'[DRY RUN] ' if dry_run else ''}Moving val clips → validation/ ──")

    # Shuffle val rows for random naming
    val_rows_shuffled = val_rows.copy()
    random.shuffle(val_rows_shuffled)

    val_moved     = 0
    val_not_found = 0
    val_updated   = []  # rows with new filenames

    for i, row in enumerate(val_rows_shuffled):
        orig_fn  = row["filename"]
        new_fn   = f"val_{i+1:04d}.wav"
        src_path = find_wav(orig_fn, final_dir)
        dst_path = val_dir / new_fn

        if src_path is None:
            val_not_found += 1
            print(f"  WARNING: not found — {orig_fn}")
            continue

        if not dry_run:
            shutil.move(str(src_path), str(dst_path))
        val_moved += 1

        # Update row with new filename
        updated_row = dict(row)
        updated_row["filename"]          = new_fn
        updated_row["original_filename"] = orig_fn
        updated_row["split"]             = "val"
        val_updated.append(updated_row)

    tag = "[DRY RUN] " if dry_run else ""
    print(f"  {tag}Moved:     {val_moved}")
    print(f"  {tag}Not found: {val_not_found}")
    print(f"  {tag}Sample: {val_rows_shuffled[0]['filename']} → val_0001.wav")
    print(f"  {tag}Sample: {val_rows_shuffled[1]['filename']} → val_0002.wav")

    # ── STEP 6: Move test clips → test/ with new names ────────────────────────
    print(f"\n── Step 6: {'[DRY RUN] ' if dry_run else ''}Moving test clips → test/ ──")

    test_rows_shuffled = test_rows.copy()
    random.shuffle(test_rows_shuffled)

    test_moved     = 0
    test_not_found = 0
    test_updated   = []

    for i, row in enumerate(test_rows_shuffled):
        orig_fn  = row["filename"]
        new_fn   = f"test_{i+1:04d}.wav"
        src_path = find_wav(orig_fn, final_dir)
        dst_path = test_dir / new_fn

        if src_path is None:
            test_not_found += 1
            print(f"  WARNING: not found — {orig_fn}")
            continue

        if not dry_run:
            shutil.move(str(src_path), str(dst_path))
        test_moved += 1

        updated_row = dict(row)
        updated_row["filename"]          = new_fn
        updated_row["original_filename"] = orig_fn
        updated_row["split"]             = "test"
        test_updated.append(updated_row)

    print(f"  {tag}Moved:     {test_moved}")
    print(f"  {tag}Not found: {test_not_found}")
    print(f"  {tag}Sample: {test_rows_shuffled[0]['filename']} → test_0001.wav")

    # ── STEP 7: Write train_metadata.csv ──────────────────────────────────────
    print(f"\n── Step 7: {'[DRY RUN] ' if dry_run else ''}Writing train_metadata.csv ──")
    train_csv = final_dir / "train_metadata.csv"

    if not dry_run:
        with open(train_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(train_rows)
        print(f"  Written {len(train_rows)} rows → train_metadata.csv")
    else:
        print(f"  [DRY RUN] Would write {len(train_rows)} rows → train_metadata.csv")

    # ── STEP 8: Write val_metadata.csv ────────────────────────────────────────
    print(f"\n── Step 8: {'[DRY RUN] ' if dry_run else ''}Writing val_metadata.csv ──")
    val_csv = final_dir / "val_metadata.csv"

    if not dry_run:
        with open(val_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(val_updated)
        print(f"  Written {len(val_updated)} rows → val_metadata.csv")
    else:
        print(f"  [DRY RUN] Would write {len(val_updated)} rows → val_metadata.csv")

    # ── STEP 9: Write test_metadata.csv ───────────────────────────────────────
    print(f"\n── Step 9: {'[DRY RUN] ' if dry_run else ''}Writing test_metadata.csv ──")
    test_csv = final_dir / "test_metadata.csv"

    if not dry_run:
        with open(test_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(test_updated)
        print(f"  Written {len(test_updated)} rows → test_metadata.csv")
    else:
        print(f"  [DRY RUN] Would write {len(test_updated)} rows → test_metadata.csv")

    # ── STEP 10: Verify class folders contain ONLY train clips ────────────────
    if not dry_run:
        print(f"\n── Step 10: Verifying class folders contain only train clips ──")
        print(f"\n  {'Subclass':<20} {'Files on disk':>14}  {'Train in CSV':>12}  {'Match?'}")
        print(f"  {'-'*20} {'-'*14}  {'-'*12}  {'-'*6}")

        all_match = True
        for mc in ["wildlife", "nature", "urban"]:
            for sub in SUBCLASS_ORDER:
                folder = final_dir / mc / sub
                if not folder.exists():
                    continue
                on_disk = len(list(folder.glob("*.wav")))
                in_csv  = sub_train[sub]
                match   = "✓" if on_disk == in_csv else "✗ MISMATCH"
                if on_disk != in_csv:
                    all_match = False
                print(f"  {sub:<20} {on_disk:>14}  {in_csv:>12}  {match}")

        if all_match:
            print(f"\n  ✓ All class folders contain exactly the train clips")
        else:
            print(f"\n  ✗ Some folders have mismatches — check above")

        # Verify validation and test folders
        val_on_disk  = len(list(val_dir.glob("*.wav")))
        test_on_disk = len(list(test_dir.glob("*.wav")))
        print(f"\n  validation/ files on disk: {val_on_disk}  (CSV: {len(val_updated)})")
        print(f"  test/       files on disk: {test_on_disk}  (CSV: {len(test_updated)})")

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    print(f"\n══ Final Summary ══════════════════════════════════════════")
    print(f"  Train clips:      {len(train_rows):>5}  → wildlife/nature/urban class folders")
    print(f"  Val clips:        {len(val_updated):>5}  → validation/ (val_0001.wav ...)")
    print(f"  Test clips:       {len(test_updated):>5}  → test/ (test_0001.wav ...)")
    print(f"  Total:            {len(train_rows)+len(val_updated)+len(test_updated):>5}")
    print()
    print(f"  Files written:")
    print(f"    train_metadata.csv  ← {len(train_rows)} rows, original filenames")
    print(f"    val_metadata.csv    ← {len(val_updated)} rows, val_XXXX.wav names")
    print(f"    test_metadata.csv   ← {len(test_updated)} rows, test_XXXX.wav names")
    print(f"    main_metadata.csv   ← UNTOUCHED (5905 rows, full reference)")
    print()
    print(f"  Folder structure:")
    print(f"    final_dataset/")
    print(f"      wildlife/ nature/ urban/  ← train clips only (nested)")
    print(f"      validation/               ← {val_moved} val clips (flat, random names)")
    print(f"      test/                     ← {test_moved} test clips (flat, random names)")
    print(f"      normalisation_log.csv     ← reference (untouched)")
    print(f"      main_metadata.csv         ← reference (untouched)")
    print(f"      train_metadata.csv        ← DataLoader reads this for training")
    print(f"      val_metadata.csv          ← DataLoader reads this for validation")
    print(f"      test_metadata.csv         ← DataLoader reads this for testing")

    if dry_run:
        print(f"\n  [DRY RUN — no files modified. Run without --dry-run to apply.]")
    else:
        print(f"\n  Dataset split complete.")
        print(f"  Next: implement DataLoader in pipeline/dataset.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split final_dataset into train/val/test")
    parser.add_argument("--final",   required=True,
                        help="Path to final_dataset/ folder")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    final = Path(args.final)
    if not final.exists():
        print(f"ERROR: {final} does not exist")
        exit(1)

    run(final, args.dry_run)
