"""
Final Dataset Creation Script
-------------------------------
Creates the final balanced dataset folder structure:
  final_dataset/
      wildlife/crow/owl/frog/insects/
      nature/rain/sea_waves/wind/crackling_fire/
      urban/car_horn/engine_idling/siren/jackhammer/
      main_metadata.csv  ← final 5,925 rows with split column filled

Sampling targets (Option A):
  car_horn    → 425 (keep all — below 500)
  all others  → 500 each
  Total       → 5,925 clips

Sampling strategy:
  - Stratified by source within each subclass
    (preserves source diversity proportionally)
  - ESC-50 originals and their aug pairs always
    placed in the same split (prevents data leakage)
  - Random seed 42 for reproducibility

Split allocation:
  train → 70%   (350 per subclass, 297 for car_horn)
  val   → 15%   (75 per subclass, 64 for car_horn)
  test  → 15%   (75 per subclass, 64 for car_horn)

Audio source folders searched:
  ESC-50-master/audio/
  fsd50k/sliced_audio/
  birdclef_2021/sliced_audio/
  anuraset/sliced_audio/
  urbansound8k/sliced_audio/
  freefield1010/sliced_audio/
  forest_wild_fire_sound_dataset/sliced_audio/

Usage:
    python create_final_dataset.py --base /path/to/datasets --dry-run
    python create_final_dataset.py --base /path/to/datasets
"""

import csv
import math
import random
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────

TARGETS = {
    "crow":           500,
    "owl":            500,
    "frog":           500,
    "insects":        500,
    "rain":           500,
    "sea_waves":      500,
    "wind":           500,
    "crackling_fire": 500,
    "car_horn":       425,   # keep all — below 500
    "engine_idling":  500,
    "siren":          500,
    "jackhammer":     500,
}

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

# Source audio folders (relative to base)
AUDIO_FOLDERS = [
    "ESC-50-master/audio",
    "fsd50k/sliced_audio",
    "birdclef_2021/sliced_audio",
    "anuraset/sliced_audio",
    "urbansound8k/sliced_audio",
    "freefield1010/sliced_audio",
    "forest_wild_fire_sound_dataset/sliced_audio",
]

FINAL_CSV_FIELDS = [
    "filename", "subclass", "main_class", "source",
    "original_filename", "sample_rate", "duration_s",
    "augmented", "split"
]

RANDOM_SEED = 42

# ── HELPERS ───────────────────────────────────────────────────────────────────

def stratified_sample(rows, n, seed=42):
    """
    Sample n rows stratified by source column.
    Preserves source diversity proportionally.
    ESC-50 aug pairs are kept with their originals.
    """
    if len(rows) <= n:
        return rows.copy()

    rng = random.Random(seed)

    # Separate ESC-50 original+aug pairs
    esc50_orig = {r["filename"]: r for r in rows
                  if r["source"] == "ESC-50"}
    esc50_aug  = {r["filename"].replace("_aug.wav", ".wav"): r
                  for r in rows if r["source"] == "ESC-50-aug"}

    # Build pairs: orig_fn → (orig_row, aug_row or None)
    esc50_pairs = []
    for orig_fn, orig_row in esc50_orig.items():
        aug_row = esc50_aug.get(orig_fn)
        esc50_pairs.append((orig_row, aug_row))

    # Non-ESC-50 rows
    other_rows = [r for r in rows
                  if r["source"] not in ("ESC-50", "ESC-50-aug")]

    # Group other rows by source
    by_source = defaultdict(list)
    for r in other_rows:
        by_source[r["source"]].append(r)

    # How many slots for ESC-50 pairs vs others
    n_esc50_pairs = min(len(esc50_pairs), n // 10)  # up to 10% from ESC-50
    n_others      = n - (n_esc50_pairs * 2 if esc50_pairs else 0)

    # If not enough non-ESC50 to fill, take more ESC-50
    if len(other_rows) < n_others:
        n_others      = len(other_rows)
        n_esc50_pairs = min(len(esc50_pairs), (n - n_others) // 2)

    # Sample ESC-50 pairs
    sampled_esc50_pairs = rng.sample(esc50_pairs,
                                      min(n_esc50_pairs, len(esc50_pairs)))
    sampled = []
    for orig_row, aug_row in sampled_esc50_pairs:
        sampled.append(orig_row)
        if aug_row:
            sampled.append(aug_row)

    # Stratified sample from other sources
    remaining = n - len(sampled)
    total_other = len(other_rows)

    if total_other > 0 and remaining > 0:
        for source, src_rows in sorted(by_source.items()):
            proportion = len(src_rows) / total_other
            take       = max(1, round(remaining * proportion))
            take       = min(take, len(src_rows))
            sampled.extend(rng.sample(src_rows, take))

        # Trim or top up to exact n
        rng.shuffle(sampled)
        if len(sampled) > n:
            sampled = sampled[:n]
        elif len(sampled) < n:
            # Top up from remaining other rows
            sampled_fns = {r["filename"] for r in sampled}
            leftover    = [r for r in other_rows
                           if r["filename"] not in sampled_fns]
            rng.shuffle(leftover)
            sampled.extend(leftover[:n - len(sampled)])

    return sampled[:n]


def assign_splits(rows, train_pct=0.70, val_pct=0.15, seed=42):
    """
    Assign train/val/test splits.
    ESC-50 orig+aug pairs always go to same split.
    Returns rows with split field filled.
    """
    rng = random.Random(seed)
    n   = len(rows)

    n_train = math.floor(n * train_pct)
    n_val   = math.floor(n * val_pct)
    n_test  = n - n_train - n_val

    # Separate ESC-50 pairs
    esc50_orig_rows = [r for r in rows if r["source"] == "ESC-50"]
    esc50_aug_rows  = {r["filename"].replace("_aug.wav", ".wav"): r
                       for r in rows if r["source"] == "ESC-50-aug"}
    other_rows      = [r for r in rows
                       if r["source"] not in ("ESC-50", "ESC-50-aug")]

    # Shuffle
    rng.shuffle(esc50_orig_rows)
    rng.shuffle(other_rows)

    # Assign splits to other rows first
    splits = (["train"] * math.floor(len(other_rows) * train_pct) +
              ["val"]   * math.floor(len(other_rows) * val_pct))
    splits += ["test"] * (len(other_rows) - len(splits))
    rng.shuffle(splits)

    result = []
    for row, split in zip(other_rows, splits):
        row = dict(row)
        row["split"] = split
        result.append(row)

    # Assign ESC-50 pairs to same split
    esc50_splits = (["train"] * math.floor(len(esc50_orig_rows) * train_pct) +
                    ["val"]   * math.floor(len(esc50_orig_rows) * val_pct))
    esc50_splits += ["test"] * (len(esc50_orig_rows) - len(esc50_splits))
    rng.shuffle(esc50_splits)

    for orig_row, split in zip(esc50_orig_rows, esc50_splits):
        orig_fn  = orig_row["filename"]
        aug_row  = esc50_aug_rows.get(orig_fn)

        o = dict(orig_row)
        o["split"] = split
        result.append(o)

        if aug_row:
            a = dict(aug_row)
            a["split"] = split  # same split as original
            result.append(a)

    return result


def find_wav(filename: str, search_dirs: list) -> Path:
    """Find a WAV file across multiple search directories."""
    for d in search_dirs:
        p = d / filename
        if p.exists():
            return p
    return None

# ── MAIN ──────────────────────────────────────────────────────────────────────

def run(base_path: Path, dry_run: bool):
    metadata_csv = base_path / "main_metadata.csv"
    final_dir    = base_path / "final_dataset"

    if not metadata_csv.exists():
        print(f"ERROR: {metadata_csv} not found")
        return

    random.seed(RANDOM_SEED)

    # Build search dirs
    search_dirs = [base_path / f for f in AUDIO_FOLDERS]

    # ── STEP 1: Read metadata ─────────────────────────────────────────────────
    print("\n── Step 1: Reading main_metadata.csv ──")
    all_rows = []
    with open(metadata_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_rows.append(row)
    print(f"  Total rows: {len(all_rows)}")

    # Group by subclass
    by_subclass = defaultdict(list)
    for row in all_rows:
        by_subclass[row["subclass"]].append(row)

    # ── STEP 2: Sample per subclass ───────────────────────────────────────────
    print("\n── Step 2: Stratified sampling per subclass ──")
    sampled = {}

    for sub in SUBCLASS_ORDER:
        rows      = by_subclass[sub]
        target    = TARGETS[sub]
        selected  = stratified_sample(rows, target, seed=RANDOM_SEED)
        sampled[sub] = selected

        # Source breakdown of selected
        src_counts = defaultdict(int)
        for r in selected:
            src_counts[r["source"]] += 1
        src_str = "  ".join(f"{s}:{c}" for s, c in
                            sorted(src_counts.items(),
                                   key=lambda x: x[1], reverse=True))
        print(f"  {sub:<20} {len(rows):>5} → {len(selected):>4}  | {src_str}")

    total_sampled = sum(len(v) for v in sampled.values())
    print(f"\n  Total sampled: {total_sampled}")

    # ── STEP 3: Assign train/val/test splits ──────────────────────────────────
    print("\n── Step 3: Assigning train/val/test splits ──")
    final_rows = []

    for sub in SUBCLASS_ORDER:
        rows_with_splits = assign_splits(sampled[sub], seed=RANDOM_SEED)
        final_rows.extend(rows_with_splits)

        split_counts = defaultdict(int)
        for r in rows_with_splits:
            split_counts[r["split"]] += 1
        print(f"  {sub:<20} "
              f"train:{split_counts['train']:>4}  "
              f"val:{split_counts['val']:>3}  "
              f"test:{split_counts['test']:>3}")

    # ── STEP 4: Create folder structure ───────────────────────────────────────
    print(f"\n── Step 4: {'[DRY RUN] ' if dry_run else ''}Creating folder structure ──")

    folders_to_create = []
    for mc in ["wildlife", "nature", "urban"]:
        folders_to_create.append(final_dir / mc)
    for sub, mc in MAIN_CLASS.items():
        folders_to_create.append(final_dir / mc / sub)

    for folder in folders_to_create:
        if not dry_run:
            folder.mkdir(parents=True, exist_ok=True)
        print(f"  {'[DRY RUN] ' if dry_run else ''}mkdir: {folder.relative_to(base_path)}")

    # ── STEP 5: Copy WAV files ─────────────────────────────────────────────────
    print(f"\n── Step 5: {'[DRY RUN] ' if dry_run else ''}Copying WAV files ──")

    copied    = 0
    not_found = 0
    errors    = 0
    not_found_list = []

    for row in final_rows:
        fn      = row["filename"]
        sub     = row["subclass"]
        mc      = MAIN_CLASS[sub]
        dst_dir = final_dir / mc / sub
        dst     = dst_dir / fn

        src = find_wav(fn, search_dirs)

        if src is None:
            not_found += 1
            not_found_list.append(fn)
            continue

        if dst.exists():
            copied += 1
            continue

        if not dry_run:
            try:
                shutil.copy2(str(src), str(dst))
                copied += 1
            except Exception as e:
                print(f"  ERROR copying {fn}: {e}")
                errors += 1
        else:
            copied += 1

        if copied % 500 == 0 and copied > 0 and not dry_run:
            print(f"  Progress: {copied}/{len(final_rows)} copied...")

    tag = "[DRY RUN] " if dry_run else ""
    print(f"  {tag}Copied:    {copied}")
    print(f"  {tag}Not found: {not_found}")
    print(f"  {tag}Errors:    {errors}")

    if not_found_list[:10]:
        print(f"  Sample not found: {not_found_list[:10]}")

    # ── STEP 6: Write final main_metadata.csv ─────────────────────────────────
    print(f"\n── Step 6: {'[DRY RUN] ' if dry_run else ''}Writing final_dataset/main_metadata.csv ──")
    final_csv = final_dir / "main_metadata.csv"

    if not dry_run:
        with open(final_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FINAL_CSV_FIELDS,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(final_rows)
        verify = sum(1 for _ in open(final_csv)) - 1
        print(f"  Written {verify} rows → final_dataset/main_metadata.csv")
    else:
        print(f"  [DRY RUN] Would write {len(final_rows)} rows")

    # ── STEP 7: Final summary ──────────────────────────────────────────────────
    print("\n── Final Summary ──")
    print(f"\n  {'Subclass':<20} {'Main Class':<12} {'Total':>6}  "
          f"{'Train':>6}  {'Val':>5}  {'Test':>5}")
    print(f"  {'-'*20} {'-'*12} {'-'*6}  {'-'*6}  {'-'*5}  {'-'*5}")

    split_totals = defaultdict(int)
    for sub in SUBCLASS_ORDER:
        rows_sub  = [r for r in final_rows if r["subclass"] == sub]
        n_train   = sum(1 for r in rows_sub if r["split"] == "train")
        n_val     = sum(1 for r in rows_sub if r["split"] == "val")
        n_test    = sum(1 for r in rows_sub if r["split"] == "test")
        mc        = MAIN_CLASS[sub]
        print(f"  {sub:<20} {mc:<12} {len(rows_sub):>6}  "
              f"{n_train:>6}  {n_val:>5}  {n_test:>5}")
        split_totals["train"] += n_train
        split_totals["val"]   += n_val
        split_totals["test"]  += n_test

    total = sum(split_totals.values())
    print(f"  {'-'*20} {'-'*12} {'-'*6}  {'-'*6}  {'-'*5}  {'-'*5}")
    print(f"  {'TOTAL':<20} {'':<12} {total:>6}  "
          f"{split_totals['train']:>6}  "
          f"{split_totals['val']:>5}  "
          f"{split_totals['test']:>5}")

    print(f"\n  Folder structure created at: final_dataset/")
    print(f"  final_dataset/")
    for mc in ["wildlife", "nature", "urban"]:
        subs = [s for s, m in MAIN_CLASS.items() if m == mc]
        print(f"      {mc}/")
        for sub in subs:
            n = len([r for r in final_rows if r["subclass"] == sub])
            print(f"          {sub}/  ({n} WAV files)")
    print(f"      main_metadata.csv  ({total} rows, split column filled)")

    if dry_run:
        print(f"\n  [DRY RUN — no files created. Run without --dry-run to apply.]")
    else:
        print(f"\n  Final dataset created successfully.")
        print(f"  Next step: normalise audio clips to uniform sample rate (22050Hz)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create final balanced dataset folder structure")
    parser.add_argument("--base",    required=True,
                        help="Path to datasets root folder")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = Path(args.base)
    if not base.exists():
        print(f"ERROR: {base} does not exist")
        exit(1)

    run(base, args.dry_run)
