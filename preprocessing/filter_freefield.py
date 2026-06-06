"""
freefield1010 Filter + Slice + Metadata Script
------------------------------------------------
Filters freefield1010 by JSON sidecar tags, slices each 10s WAV
into exactly two 5s clips, and updates local + global metadata
and filename_registry.csv.

Tag filter strategy:
  rain      → tags contain any of: rain, rainfall, raining, rainy, rainstorm
  sea_waves → tags contain any of: ocean, waves, wave, sea, surf, beach,
                                    shore, coast, seaside
  wind      → tags contain any of: wind, windy, breeze, gust, gale, windstorm
  crackling_fire → tags contain any of: fire, bonfire, campfire, fireplace,
                                         firewood, crackling
                   AND must NOT contain: fireworks, explosion, gunshot

Exclusive assignment (priority order to avoid duplicates):
  rain > sea_waves > wind > crackling_fire
  A file tagged rain+wind is assigned to rain only.

Slicing:
  Each 10s file → slice_1 = samples 0–110249 (0s–5s)
                  slice_2 = samples 110250–220499 (5s–10s)
  Both slices named: rain_537_s01.wav, rain_537_s02.wav
  (continuing from registry counter)

No resampling needed — already 44100Hz mono PCM_16.
Note: pipeline target SR is 22050Hz — files will be resampled
during final standardisation pass. Stored as-is for now.

Usage:
    python filter_freefield.py --base ./freefield1010 --global-meta ./main_metadata.csv --registry ./filename_registry.csv
    python filter_freefield.py --base ./freefield1010 --global-meta ./main_metadata.csv --registry ./filename_registry.csv --dry-run

Requirements:
    pip install soundfile numpy
"""

import csv
import json
import random
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────

# Tag sets per subclass (all lowercase)
TAG_FILTERS = {
    "rain": {
        "include": {"rain", "rainfall", "raining", "rainy", "rainstorm"},
        "exclude": {"fireworks", "explosion", "gunshot", "music", "speech"},
    },
    "sea_waves": {
        "include": {"ocean", "waves", "wave", "sea", "surf", "beach",
                    "shore", "coast", "seaside"},
        "exclude": {"fireworks", "explosion", "music", "speech"},
    },
    "wind": {
        "include": {"wind", "windy", "breeze", "gust", "gale", "windstorm"},
        "exclude": {"fireworks", "explosion", "music", "speech"},
    },
    "crackling_fire": {
        "include": {"fire", "bonfire", "campfire", "fireplace",
                    "firewood", "crackling"},
        "exclude": {"fireworks", "explosion", "gunshot", "music", "speech"},
    },
}

# Priority order for exclusive assignment
PRIORITY_ORDER = ["rain", "sea_waves", "wind", "crackling_fire"]

# How many SOURCE files to take per subclass (each gives 2 clips)
MAX_FILES = {
    "rain":           200,   # → 400 clips
    "sea_waves":      200,   # → 400 clips
    "wind":           200,   # → 400 clips
    "crackling_fire": None,  # → take all available (~56 files = 112 clips)
}

SUBCLASS_TO_MAIN = {
    "rain":           "nature",
    "sea_waves":      "nature",
    "wind":           "nature",
    "crackling_fire": "nature",
}

ALL_SUBCLASSES = [
    "crow", "owl", "frog", "insects",
    "rain", "sea_waves", "wind", "crackling_fire",
    "car_horn", "engine_idling", "siren", "jackhammer",
]

TARGET_SR      = 44100          # freefield1010 native — keep as-is
TARGET_DUR     = 5.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DUR)   # 220,500

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

# ── TAG MATCHING ──────────────────────────────────────────────────────────────

def assign_subclass(tags: list) -> str:
    """
    Assign a single subclass using priority order.
    Returns subclass name or None if no match.
    """
    tag_set = set(t.lower().strip() for t in tags)

    for subclass in PRIORITY_ORDER:
        rule = TAG_FILTERS[subclass]
        # Check include — any tag matches
        if not tag_set.intersection(rule["include"]):
            continue
        # Check exclude — none must match
        if tag_set.intersection(rule["exclude"]):
            continue
        return subclass

    return None

# ── MAIN ──────────────────────────────────────────────────────────────────────

def run(base_path: Path, global_meta_path: Path, registry_path: Path, dry_run: bool):
    output_dir = base_path / "sliced_audio"
    local_meta = base_path / "freefield_metadata.csv"

    if not dry_run:
        output_dir.mkdir(exist_ok=True)

    # ── STEP 1: Load registry ─────────────────────────────────────────────────
    print("\n── Step 1: Loading registry ──")
    registry = load_registry(registry_path)
    print(f"  {'Subclass':<20} {'Counter':>7}  {'Last file':<30}  {'Originals':>9}")
    print(f"  {'-'*20} {'-'*7}  {'-'*30}  {'-'*9}")
    for sub in PRIORITY_ORDER:
        r = registry[sub]
        print(f"  {sub:<20} {r['last_counter']:>7}  {r['last_filename']:<30}  "
              f"{r['total_originals']:>9}")

    # ── STEP 2: Scan and tag all JSON files ───────────────────────────────────
    print("\n── Step 2: Scanning JSON sidecars and assigning subclasses ──")

    subclass_files = defaultdict(list)  # subclass → list of wav paths
    unmatched      = 0
    total_json     = 0

    for folder in sorted(base_path.iterdir()):
        if not folder.is_dir() or folder.name == "metadataonly":
            continue

        json_files = sorted(folder.glob("*.json"))
        for json_path in json_files:
            total_json += 1
            wav_path = json_path.with_suffix(".wav")
            if not wav_path.exists():
                continue

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                tags = meta.get("tags", [])
            except Exception:
                continue

            subclass = assign_subclass(tags)
            if subclass:
                subclass_files[subclass].append(wav_path)
            else:
                unmatched += 1

    print(f"  Total JSON files scanned: {total_json}")
    print(f"  Unmatched (no target subclass): {unmatched}")
    print()
    for sub in PRIORITY_ORDER:
        files = subclass_files[sub]
        clips = len(files) * 2
        take  = MAX_FILES[sub] if MAX_FILES[sub] else len(files)
        print(f"  {sub:<20} {len(files):>4} files matched  "
              f"→ {clips:>4} clips available  "
              f"taking: {min(take, len(files))} files "
              f"→ {min(take, len(files))*2} clips")

    # ── STEP 3: Sample files per subclass ─────────────────────────────────────
    print("\n── Step 3: Sampling files per subclass ──")
    random.seed(42)
    selected = {}

    for sub in PRIORITY_ORDER:
        files    = subclass_files[sub]
        max_take = MAX_FILES[sub]
        n        = min(max_take, len(files)) if max_take else len(files)
        sampled  = random.sample(files, n)
        selected[sub] = sampled
        print(f"  {sub:<20} selected {len(sampled)} files → {len(sampled)*2} clips")

    total_clips = sum(len(v) * 2 for v in selected.values())
    print(f"\n  Total clips to generate: {total_clips}")

    # ── STEP 4: Build output plan ──────────────────────────────────────────────
    print("\n── Step 4: Building output plan ──")
    counters = {sub: registry[sub]["last_counter"] for sub in ALL_SUBCLASSES}
    plan     = []  # (wav_path, sub, slice_idx, new_fn, sample_start)

    for sub in PRIORITY_ORDER:
        start = counters[sub]
        for wav_path in selected[sub]:
            counters[sub] += 1
            base_counter = counters[sub]
            # Two slices per 10s file
            for s_idx, sample_start in enumerate([0, TARGET_SAMPLES]):
                new_fn = f"{sub}_{base_counter:03d}_s{s_idx+1:02d}.wav"
                plan.append((wav_path, sub, s_idx+1, new_fn,
                             sample_start, base_counter))

        end = counters[sub]
        if end > start:
            print(f"  {sub:<20} counter {start:>3} → {end:>3}  "
                  f"({(end-start)*2} clips)  "
                  f"{sub}_{start+1:03d}_s01.wav → {sub}_{end:03d}_s02.wav")

    print(f"\n  Total output files: {len(plan)}")

    # ── STEP 5: Slice and write ────────────────────────────────────────────────
    print(f"\n── Step 5: {'[DRY RUN] ' if dry_run else ''}Slicing and writing ──")

    written    = 0
    skipped    = 0
    errors     = 0
    local_rows  = []
    global_rows = []
    stats       = defaultdict(lambda: {"written": 0, "skipped": 0, "errors": 0})

    for wav_path, sub, s_idx, new_fn, sample_start, base_counter in plan:
        dst_path = output_dir / new_fn

        if dst_path.exists():
            skipped += 1
            stats[sub]["skipped"] += 1
            local_rows.append({
                "filename":          new_fn,
                "subclass":          sub,
                "main_class":        SUBCLASS_TO_MAIN[sub],
                "original_filename": wav_path.name,
                "slice_index":       s_idx,
                "sample_start":      sample_start,
                "sample_rate":       TARGET_SR,
                "duration_s":        TARGET_DUR,
            })
            continue

        if not dry_run:
            try:
                y, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)

                # Ensure mono
                if y.ndim == 2:
                    y = y.mean(axis=1)

                # Extract slice
                slice_end = sample_start + TARGET_SAMPLES
                if slice_end > len(y):
                    # Should not happen with 10s files but handle gracefully
                    segment = y[sample_start:]
                    if len(segment) < TARGET_SAMPLES // 2:
                        stats[sub]["errors"] += 1
                        errors += 1
                        continue
                    segment = np.pad(segment,
                                    (0, TARGET_SAMPLES - len(segment)))
                else:
                    segment = y[sample_start:slice_end]

                sf.write(str(dst_path), segment, TARGET_SR, subtype="PCM_16")
                written += 1
                stats[sub]["written"] += 1

            except Exception as e:
                print(f"  ERROR: {new_fn}: {e}")
                errors += 1
                stats[sub]["errors"] += 1
                continue
        else:
            written += 1
            stats[sub]["written"] += 1

        local_rows.append({
            "filename":          new_fn,
            "subclass":          sub,
            "main_class":        SUBCLASS_TO_MAIN[sub],
            "original_filename": wav_path.name,
            "slice_index":       s_idx,
            "sample_start":      sample_start,
            "sample_rate":       TARGET_SR,
            "duration_s":        TARGET_DUR,
        })
        global_rows.append({
            "filename":          new_fn,
            "subclass":          sub,
            "main_class":        SUBCLASS_TO_MAIN[sub],
            "source":            "freefield1010",
            "original_filename": wav_path.name,
            "sample_rate":       TARGET_SR,
            "duration_s":        TARGET_DUR,
            "augmented":         "no",
            "split":             "",
        })

        if written % 200 == 0 and written > 0 and not dry_run:
            print(f"  Progress: {written}/{len(plan)} written...")

    tag = "[DRY RUN] " if dry_run else ""
    print(f"  {tag}Written:  {written}")
    print(f"  {tag}Skipped:  {skipped}")
    print(f"  {tag}Errors:   {errors}")

    if not dry_run:
        print(f"  Files in sliced_audio/: {len(list(output_dir.glob('*.wav')))}")

    # ── STEP 6: Write local metadata ──────────────────────────────────────────
    print(f"\n── Step 6: {'[DRY RUN] ' if dry_run else ''}Writing freefield_metadata.csv ──")
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

    # ── STEP 7: Append to global main_metadata.csv ────────────────────────────
    print(f"\n── Step 7: {'[DRY RUN] ' if dry_run else ''}Updating global main_metadata.csv ──")

    if not dry_run:
        existing_fns = set()
        if global_meta_path.exists():
            with open(global_meta_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    existing_fns.add(row["filename"])

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
        print(f"  [DRY RUN] Would append {len(global_rows)} rows")

    # ── STEP 8: Update registry ────────────────────────────────────────────────
    print(f"\n── Step 8: {'[DRY RUN] ' if dry_run else ''}Updating registry ──")

    for sub in PRIORITY_ORDER:
        added = stats[sub]["written"]
        if added > 0:
            registry[sub]["last_counter"]    = counters[sub]
            registry[sub]["last_filename"]   = \
                f"{sub}_{counters[sub]:03d}_s02.wav"
            registry[sub]["total_originals"] = (
                registry[sub]["total_originals"] + added)
            registry[sub]["last_source"]     = "freefield1010"
            registry[sub]["last_updated"]    = \
                datetime.now().strftime("%Y-%m-%d %H:%M")

    if not dry_run:
        save_registry(registry, registry_path)
        print(f"  Registry saved")
    else:
        print(f"  [DRY RUN] Registry would be updated for: {PRIORITY_ORDER}")

    # ── STEP 9: Final summary ──────────────────────────────────────────────────
    print("\n── Final Summary ──")
    print(f"  {'Subclass':<20} {'Files':>6}  {'Clips':>6}  "
          f"{'Written':>7}  {'Counter':>7}")
    print(f"  {'-'*20} {'-'*6}  {'-'*6}  {'-'*7}  {'-'*7}")

    for sub in PRIORITY_ORDER:
        n_files = len(selected[sub])
        s       = stats[sub]
        print(f"  {sub:<20} {n_files:>6}  {n_files*2:>6}  "
              f"{s['written']:>7}  {counters[sub]:>7}")

    print()
    print("  Nature subclass pipeline totals (freefield1010 contribution):")
    nature_prior = {
        "rain":           80 + 496,   # ESC-50 aug + FSD50K staged
        "sea_waves":      80 + 238,
        "wind":           80 + 291,
        "crackling_fire": 80 + 26,
    }
    for sub in PRIORITY_ORDER:
        prior  = nature_prior[sub]
        added  = stats[sub]["written"]
        total  = prior + added
        status = "✅" if total >= 500 else "⚠️ "
        print(f"  {status} {sub:<20} Prior: {prior:>4}  "
              f"+freefield: {added:>4}  Total: {total:>5} / 500 target")

    print()
    print("  freefield1010 processing complete.")
    print("  Next: FSC22_forest (rain, wind, crackling_fire top-up)")

    if dry_run:
        print("\n  [DRY RUN — no files modified. Run without --dry-run to apply.]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter freefield1010 by JSON tags for nature subclasses")
    parser.add_argument("--base",        required=True,
                        help="Path to freefield1010 folder")
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
