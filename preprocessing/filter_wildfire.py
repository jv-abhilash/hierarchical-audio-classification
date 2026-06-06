"""
Forest Wild Fire Sound Dataset — Slice + Metadata Script
----------------------------------------------------------
No filtering needed — every file is crackling fire.
Slices each ~50s WAV into 10 non-overlapping 5s clips.
Takes only enough clips to reach the target (300 clips buffer).

Strategy:
  - Scan both subfolders recursively
  - Each ~50s file → 10 × 5s slices (samples 0-220499, 220500-440999, ...)
  - Discard any remainder under 1s
  - Take files until we have enough slices (stop early)
  - Target: 300 clips (crackling_fire needs ~280 more, 300 gives small buffer)

No resampling needed — already 44100Hz.
Naming: crackling_fire_124_s01.wav (continuing from registry counter 123)

Usage:
    python filter_wildfire.py --base ./forest_wild_fire_sound_dataset --global-meta ./main_metadata.csv --registry ./filename_registry.csv
    python filter_wildfire.py --base ./forest_wild_fire_sound_dataset --global-meta ./main_metadata.csv --registry ./filename_registry.csv --dry-run

Requirements:
    pip install soundfile numpy
"""

import csv
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────

SUBCLASS      = "crackling_fire"
MAIN_CLASS    = "nature"
TARGET_CLIPS  = 300             # how many clips to generate (buffer above 280 needed)
TARGET_SR     = 44100           # native SR — no resampling
TARGET_DUR    = 5.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DUR)   # 220,500
MIN_SAMPLES   = int(TARGET_SR * 1.0)           # discard remainder < 1s

ALL_SUBCLASSES = [
    "crow", "owl", "frog", "insects",
    "rain", "sea_waves", "wind", "crackling_fire",
    "car_horn", "engine_idling", "siren", "jackhammer",
]

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

# ── MAIN ──────────────────────────────────────────────────────────────────────

def run(base_path: Path, global_meta_path: Path, registry_path: Path, dry_run: bool):
    output_dir = base_path / "sliced_audio"
    local_meta = base_path / "wildfire_metadata.csv"

    if not dry_run:
        output_dir.mkdir(exist_ok=True)

    # ── STEP 1: Load registry ─────────────────────────────────────────────────
    print("\n── Step 1: Loading registry ──")
    registry  = load_registry(registry_path)
    r         = registry[SUBCLASS]
    print(f"  crackling_fire  counter: {r['last_counter']}  "
          f"last file: {r['last_filename']}  "
          f"originals so far: {r['total_originals']}")

    # ── STEP 2: Scan all WAV files ────────────────────────────────────────────
    print("\n── Step 2: Scanning WAV files ──")
    all_wavs = sorted(base_path.rglob("*.wav"))
    print(f"  Total WAV files found: {len(all_wavs)}")

    for subfolder in sorted(base_path.iterdir()):
        if subfolder.is_dir():
            count = len(list(subfolder.rglob("*.wav")))
            print(f"    {subfolder.name[:60]:<60} {count} files")

    # Estimate clips available
    est_clips = len(all_wavs) * 10  # ~50s each / 5s = 10 slices
    print(f"\n  Estimated clips available: ~{est_clips}")
    print(f"  Target clips to generate:   {TARGET_CLIPS}")
    print(f"  Files needed:               ~{TARGET_CLIPS // 10 + 1}")

    # ── STEP 3: Build slice plan ──────────────────────────────────────────────
    print("\n── Step 3: Building slice plan (stop after target reached) ──")
    counter = registry[SUBCLASS]["last_counter"]
    plan    = []   # (wav_path, slice_idx, sample_start, new_fn)
    clips_planned = 0

    for wav_path in all_wavs:
        if clips_planned >= TARGET_CLIPS:
            break

        try:
            info     = sf.info(str(wav_path))
            n_frames = info.frames
            sr       = info.samplerate
        except Exception as e:
            print(f"  WARNING: cannot read {wav_path.name}: {e}")
            continue

        # Calculate slices for this file
        start     = 0
        slice_idx = 1
        while start + TARGET_SAMPLES <= n_frames:
            if clips_planned >= TARGET_CLIPS:
                break
            counter += 1
            new_fn   = f"{SUBCLASS}_{counter:03d}_s{slice_idx:02d}.wav"
            plan.append((wav_path, slice_idx, start, new_fn, wav_path.name))
            clips_planned += 1
            start     += TARGET_SAMPLES
            slice_idx += 1

    print(f"  Files to process: "
          f"{len(set(p[0] for p in plan))}")
    print(f"  Slices planned:   {len(plan)}")
    print(f"  Counter range:    "
          f"{SUBCLASS}_{registry[SUBCLASS]['last_counter']+1:03d}_s01.wav "
          f"→ {SUBCLASS}_{counter:03d}_s{plan[-1][1]:02d}.wav")
    print(f"\n  Sample plan:")
    for wav_path, s_idx, start, new_fn, orig in plan[:4]:
        print(f"    {orig:<40} slice {s_idx}  "
              f"(samples {start}–{start+TARGET_SAMPLES})  → {new_fn}")

    # ── STEP 4: Slice and write ───────────────────────────────────────────────
    print(f"\n── Step 4: {'[DRY RUN] ' if dry_run else ''}Slicing and writing ──")
    written     = 0
    skipped     = 0
    errors      = 0
    local_rows  = []
    global_rows = []

    for wav_path, s_idx, sample_start, new_fn, orig_fn in plan:
        dst_path = output_dir / new_fn

        if dst_path.exists():
            skipped += 1
            local_rows.append({
                "filename":          new_fn,
                "subclass":          SUBCLASS,
                "main_class":        MAIN_CLASS,
                "original_filename": orig_fn,
                "slice_index":       s_idx,
                "sample_start":      sample_start,
                "sample_rate":       TARGET_SR,
                "duration_s":        TARGET_DUR,
            })
            continue

        if not dry_run:
            try:
                # Read only the slice we need — efficient for large files
                y, sr = sf.read(str(wav_path),
                                frames=TARGET_SAMPLES,
                                start=sample_start,
                                dtype="float32",
                                always_2d=False)

                # Ensure mono
                if y.ndim == 2:
                    y = y.mean(axis=1)

                # Should be exactly TARGET_SAMPLES but pad just in case
                if len(y) < TARGET_SAMPLES:
                    if len(y) < MIN_SAMPLES:
                        errors += 1
                        continue
                    y = np.pad(y, (0, TARGET_SAMPLES - len(y)))

                sf.write(str(dst_path), y, TARGET_SR, subtype="PCM_16")
                written += 1

            except Exception as e:
                print(f"  ERROR: {new_fn}: {e}")
                errors += 1
                continue
        else:
            written += 1

        local_rows.append({
            "filename":          new_fn,
            "subclass":          SUBCLASS,
            "main_class":        MAIN_CLASS,
            "original_filename": orig_fn,
            "slice_index":       s_idx,
            "sample_start":      sample_start,
            "sample_rate":       TARGET_SR,
            "duration_s":        TARGET_DUR,
        })
        global_rows.append({
            "filename":          new_fn,
            "subclass":          SUBCLASS,
            "main_class":        MAIN_CLASS,
            "source":            "forest_wild_fire",
            "original_filename": orig_fn,
            "sample_rate":       TARGET_SR,
            "duration_s":        TARGET_DUR,
            "augmented":         "no",
            "split":             "",
        })

        if written % 100 == 0 and written > 0 and not dry_run:
            print(f"  Progress: {written}/{len(plan)} written...")

    tag = "[DRY RUN] " if dry_run else ""
    print(f"  {tag}Written:  {written}")
    print(f"  {tag}Skipped:  {skipped}")
    print(f"  {tag}Errors:   {errors}")

    if not dry_run:
        print(f"  Files in sliced_audio/: "
              f"{len(list(output_dir.glob('*.wav')))}")

    # ── STEP 5: Write local metadata ──────────────────────────────────────────
    print(f"\n── Step 5: {'[DRY RUN] ' if dry_run else ''}Writing wildfire_metadata.csv ──")
    local_fields = ["filename", "subclass", "main_class", "original_filename",
                    "slice_index", "sample_start", "sample_rate", "duration_s"]

    if not dry_run:
        with open(local_meta, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=local_fields)
            writer.writeheader()
            writer.writerows(local_rows)
        print(f"  Written {len(local_rows)} rows → {local_meta.name}")
    else:
        print(f"  [DRY RUN] Would write {len(local_rows)} rows")

    # ── STEP 6: Append to global main_metadata.csv ────────────────────────────
    print(f"\n── Step 6: "
          f"{'[DRY RUN] ' if dry_run else ''}Updating global main_metadata.csv ──")

    if not dry_run:
        existing_fns = set()
        if global_meta_path.exists():
            with open(global_meta_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    existing_fns.add(row["filename"])

        new_rows = [r for r in global_rows
                    if r["filename"] not in existing_fns]
        write_header = not global_meta_path.exists()
        with open(global_meta_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=GLOBAL_CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerows(new_rows)

        total = sum(1 for _ in open(global_meta_path)) - 1
        print(f"  Appended {len(new_rows)} rows — global total now: {total}")
    else:
        print(f"  [DRY RUN] Would append {len(global_rows)} rows")

    # ── STEP 7: Update registry ────────────────────────────────────────────────
    print(f"\n── Step 7: "
          f"{'[DRY RUN] ' if dry_run else ''}Updating registry ──")

    if not dry_run:
        registry[SUBCLASS]["last_counter"]    = counter
        registry[SUBCLASS]["last_filename"]   = \
            f"{SUBCLASS}_{counter:03d}_s{plan[-1][1]:02d}.wav"
        registry[SUBCLASS]["total_originals"] = (
            registry[SUBCLASS]["total_originals"] + written)
        registry[SUBCLASS]["last_source"]     = "forest_wild_fire"
        registry[SUBCLASS]["last_updated"]    = \
            datetime.now().strftime("%Y-%m-%d %H:%M")
        save_registry(registry, registry_path)
        print(f"  Registry saved — "
              f"crackling_fire counter: {counter}")
    else:
        print(f"  [DRY RUN] crackling_fire counter would be: {counter}")

    # ── STEP 8: Final summary ──────────────────────────────────────────────────
    print("\n── Final Summary ──")
    prior = 220   # from ESC-50 + FSD50K + freefield1010
    total = prior + written
    status = "✅" if total >= 500 else "⚠️ "

    print(f"  crackling_fire clips generated: {written}")
    print(f"  {status} crackling_fire total:  "
          f"Prior {prior} + wildfire {written} = {total} / 500 target")
    print()

    files_used = len(set(p[0] for p in plan))
    files_remaining = len(all_wavs) - files_used
    print(f"  Source files used:      {files_used} of {len(all_wavs)}")
    print(f"  Source files remaining: {files_remaining} "
          f"(available if you need more)")
    print()
    print(f"  forest_wild_fire processing complete.")

    if total < 500:
        still_needed = 500 - total
        print(f"  ⚠️  Still need {still_needed} more clips.")
        print(f"  Run FSC22_forest script next to top up.")
    else:
        print(f"  crackling_fire is above 500 — FSC22_forest optional.")
        print(f"  Next: final count check across all 12 subclasses.")

    if dry_run:
        print("\n  [DRY RUN — no files modified. "
              "Run without --dry-run to apply.]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Slice forest wildfire recordings for crackling_fire subclass")
    parser.add_argument("--base",        required=True,
                        help="Path to forest_wild_fire_sound_dataset folder")
    parser.add_argument("--global-meta", required=True,
                        help="Path to main_metadata.csv")
    parser.add_argument("--registry",    required=True,
                        help="Path to filename_registry.csv")
    parser.add_argument("--dry-run",     action="store_true")
    args = parser.parse_args()

    base        = Path(args.base)
    global_meta = Path(args.global_meta)
    registry    = Path(args.registry)

    if not base.exists():
        print(f"ERROR: Path does not exist: {base}")
        exit(1)

    run(base, global_meta, registry, args.dry_run)
