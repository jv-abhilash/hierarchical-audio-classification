"""
Remove Flagged Clips Script
-----------------------------
Reads eda/08_flagged_clips.csv, deletes flagged WAV files from disk,
removes those rows from main_metadata.csv, and prints final counts.

Steps:
  1. Read 08_flagged_clips.csv (silent + clipped clips)
  2. Delete WAV files from disk (searches across all sliced_audio folders)
  3. Remove flagged rows from main_metadata.csv
  4. Verify source column exists (already present — dataset name per clip)
  5. Print final counts per subclass
  6. Save updated main_metadata.csv

Usage:
    python remove_flagged.py --base /path/to/datasets
    python remove_flagged.py --base /path/to/datasets --dry-run
"""

import csv
import argparse
from pathlib import Path
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────

# All folders that contain sliced_audio/ subfolders
AUDIO_FOLDERS = [
    "ESC-50-master/audio",
    "fsd50k/sliced_audio",
    "birdclef_2021/sliced_audio",
    "anuraset/sliced_audio",
    "urbansound8k/sliced_audio",
    "freefield1010/sliced_audio",
    "forest_wild_fire_sound_dataset/sliced_audio",
]

# ── MAIN ──────────────────────────────────────────────────────────────────────

def run(base_path: Path, dry_run: bool):
    flagged_csv  = base_path / "eda" / "08_flagged_clips.csv"
    metadata_csv = base_path / "main_metadata.csv"
    backup_csv   = base_path / "main_metadata_before_removal.csv"

    if not flagged_csv.exists():
        print(f"ERROR: {flagged_csv} not found")
        return
    if not metadata_csv.exists():
        print(f"ERROR: {metadata_csv} not found")
        return

    # ── STEP 1: Read flagged filenames ────────────────────────────────────────
    print("\n── Step 1: Reading flagged clips ──")
    flagged_fns  = set()
    flag_reasons = {}
    silent_count = 0
    clipped_count = 0

    with open(flagged_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        print(f"  Columns in flagged CSV: {reader.fieldnames}")
        for row in reader:
            # Try common column names for filename and reason
            fn = (row.get("filename") or row.get("file") or
                  row.get("clip") or list(row.values())[0]).strip()
            reason = (row.get("reason") or row.get("flag") or
                      row.get("issue") or "flagged")
            flagged_fns.add(fn)
            flag_reasons[fn] = reason
            if "silent" in reason.lower():
                silent_count += 1
            elif "clip" in reason.lower():
                clipped_count += 1

    print(f"  Total flagged: {len(flagged_fns)}")
    print(f"  Silent: {silent_count}")
    print(f"  Clipped: {clipped_count}")

    # ── STEP 2: Build filename → path index ──────────────────────────────────
    print("\n── Step 2: Building audio file index ──")
    file_index = {}  # filename → full path
    total_indexed = 0

    for folder_rel in AUDIO_FOLDERS:
        folder = base_path / folder_rel
        if not folder.exists():
            print(f"  SKIP (not found): {folder_rel}")
            continue
        wavs = list(folder.glob("*.wav"))
        for wav in wavs:
            file_index[wav.name] = wav
        total_indexed += len(wavs)
        print(f"  {folder_rel:<50} {len(wavs):>6} WAV files indexed")

    print(f"\n  Total files indexed: {total_indexed}")

    # ── STEP 3: Find flagged files on disk ────────────────────────────────────
    print("\n── Step 3: Locating flagged files on disk ──")
    found_on_disk   = []
    not_found_disk  = []

    for fn in flagged_fns:
        if fn in file_index:
            found_on_disk.append((fn, file_index[fn]))
        else:
            not_found_disk.append(fn)

    print(f"  Found on disk:     {len(found_on_disk)}")
    print(f"  Not found on disk: {len(not_found_disk)}")
    if not_found_disk[:5]:
        print(f"  Sample not found: {not_found_disk[:5]}")

    # ── STEP 4: Delete files from disk ───────────────────────────────────────
    print(f"\n── Step 4: {'[DRY RUN] ' if dry_run else ''}Deleting {len(found_on_disk)} files ──")
    deleted   = 0
    del_errors = 0

    for fn, path in found_on_disk:
        if not dry_run:
            try:
                path.unlink()
                deleted += 1
            except Exception as e:
                print(f"  ERROR deleting {fn}: {e}")
                del_errors += 1
        else:
            deleted += 1

    tag = "[DRY RUN] " if dry_run else ""
    print(f"  {tag}Deleted:  {deleted}")
    print(f"  {tag}Errors:   {del_errors}")

    # ── STEP 5: Read main_metadata.csv ────────────────────────────────────────
    print("\n── Step 5: Reading main_metadata.csv ──")
    all_rows = []
    fieldnames = None

    with open(metadata_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            all_rows.append(row)

    print(f"  Total rows: {len(all_rows)}")
    print(f"  Columns: {fieldnames}")

    # Verify source column exists
    if "source" in fieldnames:
        print(f"  ✓ source column present")
        # Show unique sources
        sources = set(r["source"] for r in all_rows)
        print(f"  Unique sources: {sorted(sources)}")
    else:
        print(f"  ⚠ source column NOT found — will add it")

    # ── STEP 6: Filter out flagged rows ───────────────────────────────────────
    print("\n── Step 6: Removing flagged rows from metadata ──")
    kept_rows    = []
    removed_rows = []
    removed_by_subclass = defaultdict(int)
    removed_by_reason   = defaultdict(int)

    for row in all_rows:
        fn = row.get("filename", "").strip()
        if fn in flagged_fns:
            removed_rows.append(row)
            removed_by_subclass[row.get("subclass", "unknown")] += 1
            removed_by_reason[flag_reasons.get(fn, "unknown")] += 1
        else:
            kept_rows.append(row)

    print(f"  Rows removed: {len(removed_rows)}")
    print(f"  Rows kept:    {len(kept_rows)}")
    print(f"\n  Removed by subclass:")
    for sub in sorted(removed_by_subclass):
        print(f"    {sub:<20} {removed_by_subclass[sub]:>4} removed")

    # ── STEP 7: Count remaining per subclass ──────────────────────────────────
    print("\n── Step 7: Remaining counts per subclass ──")
    remaining = defaultdict(int)
    for row in kept_rows:
        remaining[row.get("subclass", "unknown")] += 1

    subclass_order = [
        "crow", "owl", "frog", "insects",
        "rain", "sea_waves", "wind", "crackling_fire",
        "car_horn", "engine_idling", "siren", "jackhammer"
    ]

    print(f"\n  {'Subclass':<20} {'Before':>7}  {'Removed':>7}  {'After':>7}  {'Status'}")
    print(f"  {'-'*20} {'-'*7}  {'-'*7}  {'-'*7}  {'-'*15}")

    before_counts = defaultdict(int)
    for row in all_rows:
        before_counts[row.get("subclass", "unknown")] += 1

    all_ok = True
    for sub in subclass_order:
        before  = before_counts[sub]
        removed = removed_by_subclass[sub]
        after   = remaining[sub]
        target  = 472 if sub == "car_horn" else 500
        status  = "✓ above target" if after >= target else f"⚠ BELOW {target}"
        if after < target:
            all_ok = False
        print(f"  {sub:<20} {before:>7}  {removed:>7}  {after:>7}  {status}")

    total_after = sum(remaining.values())
    print(f"\n  Total clips remaining: {total_after}")

    # ── STEP 8: Backup + save updated metadata ───────────────────────────────
    print(f"\n── Step 8: {'[DRY RUN] ' if dry_run else ''}Saving updated main_metadata.csv ──")

    if not dry_run:
        # Backup original
        import shutil
        shutil.copy(metadata_csv, backup_csv)
        print(f"  Backed up to: {backup_csv.name}")

        # Write updated CSV
        with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept_rows)

        print(f"  Saved {len(kept_rows)} rows to main_metadata.csv")

        # Verify
        with open(metadata_csv) as f:
            verify_count = sum(1 for _ in f) - 1
        print(f"  Verified: {verify_count} rows in updated CSV")
    else:
        print(f"  [DRY RUN] Would save {len(kept_rows)} rows")
        print(f"  [DRY RUN] Would backup original to main_metadata_before_removal.csv")

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    print("\n══ Final Summary ══════════════════════════════════════")
    print(f"  Flagged clips in CSV:       {len(flagged_fns)}")
    print(f"  Found and deleted on disk:  {deleted}")
    print(f"  Not found on disk:          {len(not_found_disk)}")
    print(f"  Rows removed from metadata: {len(removed_rows)}")
    print(f"  Rows remaining:             {len(kept_rows)}")
    print(f"  All subclasses above target: {'YES' if all_ok else 'NO — check above'}")
    print()
    print("  source column: PRESENT — tracks which dataset each clip came from")
    print("  Values:", sorted(set(r.get('source','') for r in kept_rows)))
    print()
    print("  Next steps:")
    print("  1. Take a copy of the datasets folder (backup)")
    print("  2. Run the balancing + final folder creation script")
    print("  3. Move to model implementation")

    if dry_run:
        print("\n  [DRY RUN — no files modified. Run without --dry-run to apply.]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove flagged clips from disk and main_metadata.csv")
    parser.add_argument("--base",    required=True,
                        help="Path to datasets root folder")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = Path(args.base)
    if not base.exists():
        print(f"ERROR: {base} does not exist")
        exit(1)

    run(base, args.dry_run)
