"""
UrbanSound8K Filter + Pair + Metadata Script
---------------------------------------------
Filters UrbanSound8K for 4 urban subclasses and creates 5s clips
by pairing clips using bidirectional A+B / B+A strategy —
exactly the same approach used for AnuraSet.

Pairing rules:
  - Pairs are within the same class only
  - A+B forward  → concat, trim to exactly 5s
  - B+A reverse  → concat, trim to exactly 5s
  - Only keep pairs where combined duration >= 5s
  - No zero-padding at all
  - Pairs shuffled with seed 42 for reproducibility
  - Same pair (A,B) produces two clips: forward and reverse

Target clips per class:
  car_horn      → ALL valid pairs (small class, take everything)
  engine_idling → 600 pairs × 2 = 1200 clips (trim to 500 at EDA)
  siren         → 600 pairs × 2 = 1200 clips (trim to 500 at EDA)
  jackhammer    → 600 pairs × 2 = 1200 clips (trim to 500 at EDA)

Audio handling:
  - Load via librosa: resample to 22050Hz, force mono
  - Concatenate A+B → trim to TARGET_SAMPLES (5s)
  - Concatenate B+A → trim to TARGET_SAMPLES (5s)
  - Save as PCM_16 WAV

Naming:
  car_horn_156.wav   (A+B forward, continuing from registry)
  car_horn_157.wav   (B+A reverse, next counter)
  ...

Usage:
    python filter_urbansound.py --base ./urbansound8k --global-meta ./main_metadata.csv --registry ./filename_registry.csv
    python filter_urbansound.py --base ./urbansound8k --global-meta ./main_metadata.csv --registry ./filename_registry.csv --dry-run

Requirements:
    pip install librosa soundfile numpy pandas
"""

import csv
import random
import argparse
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────

# classID → (subclass, max_pairs_to_take)
# max_pairs × 2 = total clips (forward + reverse)
CLASS_CONFIG = {
    1: ("car_horn",      None),   # None = take all valid pairs
    5: ("engine_idling", 600),    # 600 pairs × 2 = 1200 clips
    8: ("siren",         600),
    7: ("jackhammer",    600),
}

SUBCLASS_TO_MAIN = {
    "car_horn":      "urban",
    "engine_idling": "urban",
    "siren":         "urban",
    "jackhammer":    "urban",
}

ALL_SUBCLASSES = [
    "crow", "owl", "frog", "insects",
    "rain", "sea_waves", "wind", "crackling_fire",
    "car_horn", "engine_idling", "siren", "jackhammer",
]

TARGET_SR      = 22050
TARGET_DUR     = 5.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DUR)   # 110,250
MIN_PAIR_DUR   = 5.0    # discard pair if A+B combined < 5s

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

# ── PAIRING ───────────────────────────────────────────────────────────────────

def build_pairs(file_list: list, max_pairs: int, min_combined_dur: float) -> list:
    """
    Build A+B pairs from file_list.
    Each file appears in multiple pairs but same (A,B) combo appears only once.
    Returns list of (path_a, path_b, dur_a, dur_b) tuples.

    Strategy:
      - Shuffle file list with seed 42
      - Pair consecutive files: (0,1), (2,3), (4,5) ...
      - Keep pair only if dur_a + dur_b >= min_combined_dur
      - If max_pairs set, stop after collecting that many valid pairs
    """
    random.seed(42)
    shuffled = file_list.copy()
    random.shuffle(shuffled)

    valid_pairs = []
    used        = set()

    # First pass — consecutive pairing
    i = 0
    while i < len(shuffled) - 1:
        path_a, dur_a = shuffled[i]
        path_b, dur_b = shuffled[i + 1]

        pair_key = tuple(sorted([path_a.name, path_b.name]))
        if pair_key not in used and dur_a + dur_b >= min_combined_dur:
            valid_pairs.append((path_a, path_b, dur_a, dur_b))
            used.add(pair_key)

        i += 2
        if max_pairs and len(valid_pairs) >= max_pairs:
            break

    # Second pass — fill remaining if needed
    if max_pairs and len(valid_pairs) < max_pairs:
        random.shuffle(shuffled)
        for i in range(len(shuffled)):
            for j in range(i + 1, len(shuffled)):
                path_a, dur_a = shuffled[i]
                path_b, dur_b = shuffled[j]
                pair_key = tuple(sorted([path_a.name, path_b.name]))
                if pair_key not in used and dur_a + dur_b >= min_combined_dur:
                    valid_pairs.append((path_a, path_b, dur_a, dur_b))
                    used.add(pair_key)
                if max_pairs and len(valid_pairs) >= max_pairs:
                    break
            if max_pairs and len(valid_pairs) >= max_pairs:
                break

    return valid_pairs[:max_pairs] if max_pairs else valid_pairs

# ── MAIN ──────────────────────────────────────────────────────────────────────

def run(base_path: Path, global_meta_path: Path, registry_path: Path, dry_run: bool):
    csv_path   = base_path / "UrbanSound8K.csv"
    output_dir = base_path / "sliced_audio"
    local_meta = base_path / "urbansound_metadata.csv"

    if not csv_path.exists():
        print(f"ERROR: UrbanSound8K.csv not found at {csv_path}")
        return

    if not dry_run:
        output_dir.mkdir(exist_ok=True)

    # ── STEP 1: Load registry ─────────────────────────────────────────────────
    print("\n── Step 1: Loading registry ──")
    registry = load_registry(registry_path)
    print(f"  {'Subclass':<20} {'Counter':>7}  {'Last file':<35}  {'Originals':>9}")
    print(f"  {'-'*20} {'-'*7}  {'-'*35}  {'-'*9}")
    for sub in ["car_horn", "engine_idling", "siren", "jackhammer"]:
        r = registry[sub]
        print(f"  {sub:<20} {r['last_counter']:>7}  {r['last_filename']:<35}  "
              f"{r['total_originals']:>9}")

    # ── STEP 2: Load UrbanSound8K.csv ─────────────────────────────────────────
    print("\n── Step 2: Loading UrbanSound8K.csv ──")
    df = pd.read_csv(str(csv_path))
    print(f"  Total rows: {len(df)}")

    # ── STEP 3: Build file lists per class ────────────────────────────────────
    print("\n── Step 3: Scanning files per class ──")
    class_files = {}

    for class_id, (subclass, max_pairs) in CLASS_CONFIG.items():
        class_df  = df[df["classID"] == class_id].copy()
        file_list = []

        for _, row in class_df.iterrows():
            fold     = int(row["fold"])
            orig_fn  = row["slice_file_name"]
            src_path = base_path / f"fold{fold}" / orig_fn
            if not src_path.exists():
                continue
            dur = float(row["end"]) - float(row["start"])
            file_list.append((src_path, dur))

        class_files[class_id] = file_list
        print(f"  {subclass:<20} {len(file_list):>4} files on disk  "
              f"avg dur: {sum(d for _,d in file_list)/len(file_list):.2f}s  "
              f"max pairs target: {max_pairs if max_pairs else 'all'}")

    # ── STEP 4: Build pairs per class ─────────────────────────────────────────
    print("\n── Step 4: Building A+B pairs per class ──")
    class_pairs = {}

    for class_id, (subclass, max_pairs) in CLASS_CONFIG.items():
        file_list   = class_files[class_id]
        pairs       = build_pairs(file_list, max_pairs, MIN_PAIR_DUR)
        class_pairs[class_id] = pairs

        discarded   = len(file_list) // 2 - len(pairs)
        print(f"  {subclass:<20} {len(pairs):>4} valid pairs  "
              f"→ {len(pairs)*2:>4} clips (forward + reverse)  "
              f"discarded short pairs: {max(0, discarded)}")

    total_clips = sum(len(p) * 2 for p in class_pairs.values())
    print(f"\n  Total clips to generate: {total_clips}")

    # ── STEP 5: Generate clips ─────────────────────────────────────────────────
    print(f"\n── Step 5: {'[DRY RUN] ' if dry_run else ''}Generating paired clips ──")

    counters    = {sub: registry[sub]["last_counter"] for sub in ALL_SUBCLASSES}
    local_rows  = []
    global_rows = []
    stats       = defaultdict(lambda: {"written": 0, "skipped": 0,
                                        "errors": 0, "discarded": 0})

    for class_id, (subclass, _) in CLASS_CONFIG.items():
        pairs = class_pairs[class_id]

        for path_a, path_b, dur_a, dur_b in pairs:
            # Generate both directions
            directions = [
                (path_a, path_b, "forward", f"{path_a.name}+{path_b.name}"),
                (path_b, path_a, "reverse", f"{path_b.name}+{path_a.name}"),
            ]

            for first, second, direction, orig_desc in directions:
                counters[subclass] += 1
                new_fn   = f"{subclass}_{counters[subclass]:03d}.wav"
                dst_path = output_dir / new_fn

                if dst_path.exists():
                    stats[subclass]["skipped"] += 1
                    local_rows.append({
                        "filename":          new_fn,
                        "subclass":          subclass,
                        "main_class":        "urban",
                        "original_filename": orig_desc,
                        "direction":         direction,
                        "sample_rate":       TARGET_SR,
                        "duration_s":        TARGET_DUR,
                    })
                    continue

                if not dry_run:
                    try:
                        y_first,  _ = librosa.load(str(first),
                                                   sr=TARGET_SR, mono=True)
                        y_second, _ = librosa.load(str(second),
                                                   sr=TARGET_SR, mono=True)

                        # Concatenate
                        y_concat = np.concatenate([y_first, y_second])

                        # Must be >= TARGET_SAMPLES — safety check
                        if len(y_concat) < TARGET_SAMPLES:
                            stats[subclass]["discarded"] += 1
                            counters[subclass] -= 1  # roll back counter
                            continue

                        # Trim to exactly 5s
                        y_out = y_concat[:TARGET_SAMPLES].astype(np.float32)

                        sf.write(str(dst_path), y_out, TARGET_SR, subtype="PCM_16")
                        stats[subclass]["written"] += 1

                    except Exception as e:
                        print(f"  ERROR: {new_fn}: {e}")
                        stats[subclass]["errors"] += 1
                        counters[subclass] -= 1  # roll back counter
                        continue
                else:
                    stats[subclass]["written"] += 1

                local_rows.append({
                    "filename":          new_fn,
                    "subclass":          subclass,
                    "main_class":        "urban",
                    "original_filename": orig_desc,
                    "direction":         direction,
                    "sample_rate":       TARGET_SR,
                    "duration_s":        TARGET_DUR,
                })
                global_rows.append({
                    "filename":          new_fn,
                    "subclass":          subclass,
                    "main_class":        SUBCLASS_TO_MAIN[subclass],
                    "source":            "UrbanSound8K",
                    "original_filename": orig_desc,
                    "sample_rate":       TARGET_SR,
                    "duration_s":        TARGET_DUR,
                    "augmented":         "no",
                    "split":             "",
                })

        written_sub = stats[subclass]["written"]
        if written_sub % 100 == 0 and not dry_run and written_sub > 0:
            print(f"  {subclass}: {written_sub} clips written...")

    tag = "[DRY RUN] " if dry_run else ""
    total_written = sum(s["written"] for s in stats.values())
    total_errors  = sum(s["errors"]  for s in stats.values())
    print(f"  {tag}Total written:   {total_written}")
    print(f"  {tag}Total skipped:   {sum(s['skipped'] for s in stats.values())}")
    print(f"  {tag}Total discarded: {sum(s['discarded'] for s in stats.values())}")
    print(f"  {tag}Total errors:    {total_errors}")

    if not dry_run:
        print(f"  Files in sliced_audio/: {len(list(output_dir.glob('*.wav')))}")

    # ── STEP 6: Write local metadata ──────────────────────────────────────────
    print(f"\n── Step 6: {'[DRY RUN] ' if dry_run else ''}Writing urbansound_metadata.csv ──")
    local_fields = ["filename", "subclass", "main_class", "original_filename",
                    "direction", "sample_rate", "duration_s"]

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

    for subclass in ["car_horn", "engine_idling", "siren", "jackhammer"]:
        added = stats[subclass]["written"]
        if added > 0:
            registry[subclass]["last_counter"]    = counters[subclass]
            registry[subclass]["last_filename"]   = \
                f"{subclass}_{counters[subclass]:03d}.wav"
            registry[subclass]["total_originals"] = (
                registry[subclass]["total_originals"] + added)
            registry[subclass]["last_source"]     = "UrbanSound8K"
            registry[subclass]["last_updated"]    = \
                datetime.now().strftime("%Y-%m-%d %H:%M")

    if not dry_run:
        save_registry(registry, registry_path)
        print(f"  Registry saved")

    # ── STEP 9: Final summary ──────────────────────────────────────────────────
    print("\n── Final Summary ──")
    print(f"  {'Subclass':<20} {'Pairs':>6}  {'Clips (×2)':>10}  "
          f"{'Written':>7}  {'Errors':>6}  {'Counter':>7}")
    print(f"  {'-'*20} {'-'*6}  {'-'*10}  {'-'*7}  {'-'*6}  {'-'*7}")

    for class_id, (subclass, _) in CLASS_CONFIG.items():
        pairs  = len(class_pairs[class_id])
        s      = stats[subclass]
        print(f"  {subclass:<20} {pairs:>6}  {pairs*2:>10}  "
              f"{s['written']:>7}  {s['errors']:>6}  "
              f"{counters[subclass]:>7}")

    print()
    print("  Pairing method: A+B forward + B+A reverse (same as AnuraSet)")
    print("  No zero-padding — all clips exactly 5s from real audio")
    print()
    print("  Urban subclass pipeline totals:")
    for subclass in ["car_horn", "engine_idling", "siren", "jackhammer"]:
        esc50  = 80 if subclass in ["car_horn", "siren"] else 0
        fsd50k = {"car_horn": 115, "engine_idling": 103,
                  "siren": 77, "jackhammer": 0}.get(subclass, 0)
        urban  = stats[subclass]["written"]
        total  = esc50 + fsd50k + urban
        status = "✅" if total >= 500 else "⚠️ "
        print(f"  {status} {subclass:<20} ESC-50:{esc50:>4}  "
              f"FSD50K:{fsd50k:>4}  UrbanSound8K:{urban:>5}  "
              f"Total:{total:>5} / 500")

    print()
    print("  UrbanSound8K processing complete.")
    print("  Next: freefield1010 (rain, sea_waves, wind, crackling_fire)")

    if dry_run:
        print("\n  [DRY RUN — no files modified. Run without --dry-run to apply.]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter UrbanSound8K using A+B/B+A pairing — no padding")
    parser.add_argument("--base",        required=True,
                        help="Path to urbansound8k folder")
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
