"""
AnuraSet Filter + Concatenate + Metadata Script
-------------------------------------------------
Finds consecutive clip pairs from the same recording,
concatenates them (3s + 3s = 6s), trims to exactly 5s,
and samples 700 clips for the frog subclass.

Why 700: target is 420 (to reach 500 with 80 from ESC-50),
  but we sample 700 to give buffer for EDA-based discarding.
  You trim to 420 after EDA.

Pairing logic (Option C — strict consecutive):
  Filename pattern: {site}_{date}_{time}_{startSec}_{endSec}.wav
  e.g. INCT4_20191005_173000_0_3.wav and INCT4_20191005_173000_3_6.wav
  A valid pair: same site+date+time, clip B startSec == clip A endSec

Concatenation:
  clip_A (3s, 66150 samples) + clip_B (3s, 66150 samples) = 6s
  Trim first 5s = 110250 samples → exactly 5s at 22050Hz

Output: anuraset/sliced_audio/frog_XXX.wav
Local:  anuraset/anuraset_metadata.csv
Global: main_metadata.csv
Registry: filename_registry.csv

Usage:
    python filter_anuraset.py --base ./anuraset --global-meta ./main_metadata.csv --registry ./filename_registry.csv
    python filter_anuraset.py --base ./anuraset --global-meta ./main_metadata.csv --registry ./filename_registry.csv --dry-run
    python filter_anuraset.py --base ./anuraset --global-meta ./main_metadata.csv --registry ./filename_registry.csv --sample-size 700

Requirements:
    pip install soundfile numpy
"""

import csv
import re
import random
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────

DEFAULT_SAMPLE_SIZE = 700
TARGET_SR           = 22050
TARGET_DUR          = 5.0
TARGET_SAMPLES      = int(TARGET_SR * TARGET_DUR)   # 110,250
SOURCE_SAMPLES      = int(TARGET_SR * 3.0)          # 66,150 per clip
CONCAT_SAMPLES      = SOURCE_SAMPLES * 2            # 132,300 = 6s

SUBCLASS   = "frog"
MAIN_CLASS = "wildlife"

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

# ── PAIRING LOGIC ─────────────────────────────────────────────────────────────

def parse_clip_info(filename: str):
    """
    Parse INCT4_20191005_173000_0_3.wav
    Returns (site_date_time_key, start_sec, end_sec) or None if unmatched.
    """
    stem = Path(filename).stem
    # Pattern: {site}_{date}_{time}_{start}_{end}
    # site can have letters and digits, date is 8 digits, time is 6 digits
    m = re.match(r'^(.+_\d{8}_\d{6})_(\d+)_(\d+)$', stem)
    if m:
        key       = m.group(1)   # e.g. INCT4_20191005_173000
        start_sec = int(m.group(2))
        end_sec   = int(m.group(3))
        return key, start_sec, end_sec
    return None

def find_consecutive_pairs(site_clips: dict) -> list:
    """
    Find all valid consecutive pairs across all sites.
    Returns list of (clip_a_path, clip_b_path, site) tuples.
    """
    # Build lookup: key → {start_sec → path}
    pairs = []

    for site, clips in site_clips.items():
        # Build index: recording_key → dict of start_sec → path
        recording_index = defaultdict(dict)
        for clip_path in clips:
            info = parse_clip_info(clip_path.name)
            if info is None:
                continue
            key, start_sec, end_sec = info
            recording_index[key][start_sec] = (clip_path, end_sec)

        # Find consecutive pairs: clip ending at T pairs with clip starting at T
        for key, clips_by_start in recording_index.items():
            for start_a, (path_a, end_a) in clips_by_start.items():
                # Look for clip starting exactly where this one ends
                if end_a in clips_by_start:
                    path_b, end_b = clips_by_start[end_a]
                    pairs.append((path_a, path_b, site))

    return pairs

# ── MAIN ──────────────────────────────────────────────────────────────────────

def run(base_path: Path, global_meta_path: Path, registry_path: Path,
        sample_size: int, dry_run: bool):

    # Locate audio folder
    audio_base = base_path / "anuraset" / "audio"
    if not audio_base.exists():
        audio_base = base_path / "audio"
    if not audio_base.exists():
        print(f"ERROR: audio folder not found under {base_path}")
        return

    output_dir = base_path / "sliced_audio"
    local_meta = base_path / "anuraset_metadata.csv"

    if not dry_run:
        output_dir.mkdir(exist_ok=True)

    # ── STEP 1: Load registry ─────────────────────────────────────────────────
    print("\n── Step 1: Loading registry ──")
    registry = load_registry(registry_path)
    r = registry[SUBCLASS]
    print(f"  frog  counter: {r['last_counter']}  "
          f"last file: {r['last_filename']}  "
          f"originals so far: {r['total_originals']}")

    # ── STEP 2: Scan clips per site ───────────────────────────────────────────
    print("\n── Step 2: Scanning anuraset/audio/ ──")
    site_clips = defaultdict(list)
    sites      = sorted([d for d in audio_base.iterdir() if d.is_dir()])

    for site_dir in sites:
        wavs = sorted(site_dir.glob("*.wav"))
        site_clips[site_dir.name] = wavs
        print(f"  {site_dir.name:<12} {len(wavs):>6} clips")

    total_clips = sum(len(v) for v in site_clips.values())
    print(f"\n  Total clips available: {total_clips}")

    # ── STEP 3: Find all consecutive pairs ────────────────────────────────────
    print("\n── Step 3: Finding consecutive pairs ──")
    all_pairs = find_consecutive_pairs(site_clips)

    pairs_by_site = defaultdict(int)
    for _, _, site in all_pairs:
        pairs_by_site[site] += 1

    print(f"  Total valid pairs found: {len(all_pairs)}")
    for site in sorted(pairs_by_site):
        print(f"    {site:<12} {pairs_by_site[site]:>6} pairs")

    if len(all_pairs) < sample_size:
        print(f"  WARNING: only {len(all_pairs)} pairs available, "
              f"less than requested {sample_size}")
        sample_size = len(all_pairs)
        print(f"  Adjusting sample size to: {sample_size}")

    # ── STEP 4: Stratified sample ─────────────────────────────────────────────
    print(f"\n── Step 4: Stratified sampling {sample_size} pairs ──")
    random.seed(42)

    # Group pairs by site for stratified sampling
    site_pairs = defaultdict(list)
    for pair in all_pairs:
        site_pairs[pair[2]].append(pair)

    selected = []
    for site, site_pair_list in site_pairs.items():
        proportion  = len(site_pair_list) / len(all_pairs)
        n_from_site = max(1, round(sample_size * proportion))
        sampled     = random.sample(site_pair_list,
                                    min(n_from_site, len(site_pair_list)))
        selected.extend(sampled)
        print(f"  {site:<12} {len(site_pair_list):>6} pairs  "
              f"proportion: {proportion:.2%}  sampled: {len(sampled)}")

    # Trim or top-up to exact sample_size
    random.shuffle(selected)
    if len(selected) > sample_size:
        selected = selected[:sample_size]
    elif len(selected) < sample_size:
        selected_set = set((a.name, b.name) for a, b, _ in selected)
        remaining    = [(a, b, s) for a, b, s in all_pairs
                        if (a.name, b.name) not in selected_set]
        random.shuffle(remaining)
        selected.extend(remaining[:sample_size - len(selected)])

    print(f"\n  Final sample count: {len(selected)}")

    # ── STEP 5: Build output plan ──────────────────────────────────────────────
    print("\n── Step 5: Building output plan ──")
    counter = registry[SUBCLASS]["last_counter"]
    plan    = []

    for path_a, path_b, site in selected:
        counter += 1
        new_fn   = f"{SUBCLASS}_{counter:03d}.wav"
        dst_path = output_dir / new_fn
        plan.append((path_a, path_b, dst_path, new_fn, site))

    print(f"  Counter range: frog_{registry[SUBCLASS]['last_counter']+1:03d}.wav "
          f"→ frog_{counter:03d}.wav")
    print(f"  Sample pairs:")
    for a, b, dst, fn, site in plan[:3]:
        print(f"    {a.name} + {b.name}")
        print(f"    → {fn}  (site: {site})")

    # ── STEP 6: Concatenate and write ─────────────────────────────────────────
    print(f"\n── Step 6: {'[DRY RUN] ' if dry_run else ''}Concatenating + trimming to 5s ──")
    written     = 0
    skipped     = 0
    errors      = 0
    local_rows  = []
    global_rows = []

    for path_a, path_b, dst_path, new_fn, site in plan:
        if dst_path.exists():
            skipped += 1
            local_rows.append({
                "filename":          new_fn,
                "subclass":          SUBCLASS,
                "main_class":        MAIN_CLASS,
                "monitoring_site":   site,
                "source_clip_a":     path_a.name,
                "source_clip_b":     path_b.name,
                "concat_method":     "A+B trimmed to 5s",
                "sample_rate":       TARGET_SR,
                "duration_s":        TARGET_DUR,
            })
            continue

        if not dry_run:
            try:
                y_a, _ = sf.read(str(path_a), dtype="float32", always_2d=False)
                y_b, _ = sf.read(str(path_b), dtype="float32", always_2d=False)

                # Ensure mono
                if y_a.ndim == 2: y_a = y_a.mean(axis=1)
                if y_b.ndim == 2: y_b = y_b.mean(axis=1)

                # Trim each to SOURCE_SAMPLES if slightly off
                y_a = y_a[:SOURCE_SAMPLES]
                y_b = y_b[:SOURCE_SAMPLES]

                # Concatenate → 6s → trim to 5s
                y_concat = np.concatenate([y_a, y_b])
                y_out    = y_concat[:TARGET_SAMPLES]

                sf.write(str(dst_path), y_out, TARGET_SR, subtype="PCM_16")
                written += 1

            except Exception as e:
                print(f"  ERROR: {new_fn}: {e}")
                errors += 1
                continue
        else:
            written += 1

        local_rows.append({
            "filename":        new_fn,
            "subclass":        SUBCLASS,
            "main_class":      MAIN_CLASS,
            "monitoring_site": site,
            "source_clip_a":   path_a.name,
            "source_clip_b":   path_b.name,
            "concat_method":   "A+B trimmed to 5s",
            "sample_rate":     TARGET_SR,
            "duration_s":      TARGET_DUR,
        })
        global_rows.append({
            "filename":          new_fn,
            "subclass":          SUBCLASS,
            "main_class":        MAIN_CLASS,
            "source":            "AnuraSet",
            "original_filename": f"{path_a.name}+{path_b.name}",
            "sample_rate":       TARGET_SR,
            "duration_s":        TARGET_DUR,
            "augmented":         "no",
            "split":             "",
        })

        if written % 100 == 0 and written > 0 and not dry_run:
            print(f"  Progress: {written}/{len(plan)} written...")

    tag = "[DRY RUN] " if dry_run else ""
    print(f"  {tag}Written:  {written}")
    print(f"  {tag}Skipped:  {skipped} (already exist)")
    print(f"  {tag}Errors:   {errors}")

    if not dry_run:
        print(f"  Files in sliced_audio/: {len(list(output_dir.glob('*.wav')))}")

    # ── STEP 7: Write local metadata ──────────────────────────────────────────
    print(f"\n── Step 7: {'[DRY RUN] ' if dry_run else ''}Writing anuraset_metadata.csv ──")
    local_fields = ["filename", "subclass", "main_class", "monitoring_site",
                    "source_clip_a", "source_clip_b", "concat_method",
                    "sample_rate", "duration_s"]

    if not dry_run:
        with open(local_meta, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=local_fields)
            writer.writeheader()
            writer.writerows(local_rows)
        print(f"  Written {len(local_rows)} rows → {local_meta.name}")
    else:
        print(f"  [DRY RUN] Would write {len(local_rows)} rows")

    # ── STEP 8: Append to global main_metadata.csv ────────────────────────────
    print(f"\n── Step 8: {'[DRY RUN] ' if dry_run else ''}Updating global main_metadata.csv ──")

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

    # ── STEP 9: Update registry ────────────────────────────────────────────────
    print(f"\n── Step 9: {'[DRY RUN] ' if dry_run else ''}Updating registry ──")
    new_counter = registry[SUBCLASS]["last_counter"] + len(plan)

    if not dry_run:
        registry[SUBCLASS]["last_counter"]    = new_counter
        registry[SUBCLASS]["last_filename"]   = f"frog_{new_counter:03d}.wav"
        registry[SUBCLASS]["total_originals"] = (registry[SUBCLASS]["total_originals"]
                                                  + written)
        registry[SUBCLASS]["last_source"]     = "AnuraSet"
        registry[SUBCLASS]["last_updated"]    = datetime.now().strftime("%Y-%m-%d %H:%M")
        save_registry(registry, registry_path)
        print(f"  Registry saved — frog counter: {new_counter}")
    else:
        print(f"  [DRY RUN] frog counter would be: {new_counter}")

    # ── STEP 10: Final summary ─────────────────────────────────────────────────
    print("\n── Final Summary ──")
    site_counts = defaultdict(int)
    for _, _, _, _, site in plan:
        site_counts[site] += 1

    print(f"  Clips generated per monitoring site:")
    for site in sorted(site_counts):
        print(f"    {site:<12} {site_counts[site]:>3} clips")

    print(f"\n  Total AnuraSet frog clips: {len(plan)} (buffer — trim to 420 after EDA)")
    print(f"  Format: 22050Hz mono PCM_16, 5.0s (3s+3s concatenated, trimmed)")
    print(f"  Method: consecutive clip pairs from same recording")
    print()
    print(f"  Frog subclass total so far:")
    print(f"    ESC-50 (orig + aug):  80")
    print(f"    AnuraSet:             {len(plan)}")
    print(f"    Total:                {80 + len(plan)} (trim AnuraSet to 420 → final 500)")
    print()
    print(f"  AnuraSet processing complete.")
    print(f"  Next: urbansound8k (car_horn, engine_idling, siren, jackhammer)")

    if dry_run:
        print("\n  [DRY RUN — no files modified. Run without --dry-run to apply.]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Concatenate consecutive AnuraSet clip pairs for frog subclass")
    parser.add_argument("--base",        required=True, help="Path to anuraset folder")
    parser.add_argument("--global-meta", required=True, help="Path to main_metadata.csv")
    parser.add_argument("--registry",    required=True, help="Path to filename_registry.csv")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--dry-run",     action="store_true")
    args = parser.parse_args()

    base        = Path(args.base)
    global_meta = Path(args.global_meta)
    registry    = Path(args.registry)

    if not base.exists():
        print(f"ERROR: Path does not exist: {base}")
        exit(1)

    run(base, global_meta, registry, args.sample_size, args.dry_run)
