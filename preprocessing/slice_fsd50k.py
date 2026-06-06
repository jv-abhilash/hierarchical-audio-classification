"""
FSD50K Slice to 5s Script
---------------------------
Takes all files in fsd50k/filtered_audio/ and slices them into
uniform 5-second WAV clips, writing output to fsd50k/sliced_audio/.

Slicing rules (based on duration audit):
  < 1.0s   → DISCARD — too short, padding would be >80% silence
  1.0–5.0s → KEEP as single clip, zero-pad end to exactly 5s
  ~5.0s    → KEEP as single clip, trim/pad to exact 5s
  > 5.0s   → Non-overlapping 5s windows:
                 slice_1 = 0s–5s
                 slice_2 = 5s–10s
                 ...
                 final remainder < 1s → discard
                 final remainder 1s–5s → zero-pad to 5s and keep

Naming convention:
  Source file:  rain_041.wav  (variable length)
  Sliced clips: rain_041_s01.wav, rain_041_s02.wav, ...
  Single clip:  rain_041_s01.wav  (even if only one slice)

Registry: counters are NOT updated here because sliced clips
  keep the same subclass prefix number as their source.
  The slicer adds a _sXX suffix to distinguish slices.
  main_metadata.csv gets the sliced filenames replacing the originals.

Usage:
    python slice_fsd50k.py --base ./fsd50k --global-meta ./main_metadata.csv --registry ./filename_registry.csv
    python slice_fsd50k.py --base ./fsd50k --global-meta ./main_metadata.csv --registry ./filename_registry.csv --dry-run

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

TARGET_SR      = 44100          # FSD50K native — no resampling needed
TARGET_DUR     = 5.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DUR)   # 220,500 samples
MIN_DUR        = 1.0            # clips shorter than this are discarded
MIN_SAMPLES    = int(TARGET_SR * MIN_DUR)

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
    "engine_idling":  "urban",
}

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

# ── REGISTRY ─────────────────────────────────────────────────────────────────

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
    """
    Slice a 1D numpy audio array into TARGET_SAMPLES-length chunks.
    Returns list of numpy arrays, each exactly TARGET_SAMPLES long.
    Discards any remainder shorter than MIN_SAMPLES.
    """
    chunks = []
    total  = len(y)
    start  = 0

    while start < total:
        end     = start + TARGET_SAMPLES
        segment = y[start:end]

        if len(segment) < MIN_SAMPLES:
            break  # remainder too short — discard

        if len(segment) < TARGET_SAMPLES:
            # Pad with zeros to reach exactly TARGET_SAMPLES
            segment = np.pad(segment, (0, TARGET_SAMPLES - len(segment)))

        chunks.append(segment)
        start = end

    return chunks

# ── MAIN ──────────────────────────────────────────────────────────────────────

def run(base_path: Path, global_meta_path: Path, registry_path: Path, dry_run: bool):
    input_dir  = base_path / "filtered_audio"
    output_dir = base_path / "sliced_audio"
    local_meta = base_path / "sliced_metadata.csv"

    if not input_dir.exists():
        print(f"ERROR: filtered_audio/ not found — run filter_fsd50k.py first")
        return

    if not dry_run:
        output_dir.mkdir(exist_ok=True)

    # ── STEP 1: Load registry ─────────────────────────────────────────────────
    print("\n── Step 1: Loading registry ──")
    registry = load_registry(registry_path)
    print(f"  Registry loaded: {len(registry)} subclasses tracked")

    # ── STEP 2: Scan input files ──────────────────────────────────────────────
    print("\n── Step 2: Scanning filtered_audio/ ──")
    all_files = sorted(input_dir.glob("*.wav"))
    print(f"  Files found: {len(all_files)}")

    # Parse subclass from filename
    by_subclass = defaultdict(list)
    for f in all_files:
        # crow_041.wav → subclass = crow
        parts = f.stem.split("_")
        # handle sea_waves_ prefix (two-word subclass)
        for sub in SUBCLASS_TO_MAIN:
            if f.stem.startswith(sub + "_"):
                by_subclass[sub].append(f)
                break

    print(f"  Files per subclass:")
    for sub in sorted(by_subclass):
        print(f"    {sub:<20} {len(by_subclass[sub]):>4} source files")

    # ── STEP 3: Build slice plan ──────────────────────────────────────────────
    print("\n── Step 3: Building slice plan ──")
    slice_plan    = []  # (src_path, slices_info, subclass)
    stats         = defaultdict(lambda: {"files": 0, "discarded": 0, "slices": 0})

    for sub in sorted(by_subclass):
        for src_path in by_subclass[sub]:
            try:
                info = sf.info(str(src_path))
                dur  = info.duration
                sr   = info.samplerate
            except Exception as e:
                print(f"  ERROR reading {src_path.name}: {e}")
                continue

            stats[sub]["files"] += 1

            if dur < MIN_DUR:
                stats[sub]["discarded"] += 1
                continue

            # Predict number of slices without loading audio yet
            n_slices   = int(dur / TARGET_DUR)
            remainder  = dur - (n_slices * TARGET_DUR)
            if remainder >= MIN_DUR:
                n_slices += 1  # remainder is keepable with padding

            if n_slices == 0:
                stats[sub]["discarded"] += 1
                continue

            stats[sub]["slices"] += n_slices

            # Build output filenames: rain_041.wav → rain_041_s01.wav, rain_041_s02.wav ...
            stem = src_path.stem  # rain_041
            slice_filenames = [f"{stem}_s{i+1:02d}.wav" for i in range(n_slices)]
            slice_plan.append((src_path, slice_filenames, sub, dur))

    print(f"\n  Slice plan summary:")
    print(f"  {'Subclass':<20} {'Src files':>9}  {'Discarded':>9}  {'Slices est':>10}")
    print(f"  {'-'*20} {'-'*9}  {'-'*9}  {'-'*10}")
    total_slices = 0
    total_disc   = 0
    for sub in sorted(stats):
        s = stats[sub]
        total_slices += s["slices"]
        total_disc   += s["discarded"]
        print(f"  {sub:<20} {s['files']:>9}  {s['discarded']:>9}  {s['slices']:>10}")
    print(f"  {'TOTAL':<20} {sum(s['files'] for s in stats.values()):>9}  "
          f"{total_disc:>9}  {total_slices:>10}")

    # ── STEP 4: Execute slicing ───────────────────────────────────────────────
    print(f"\n── Step 4: {'[DRY RUN] ' if dry_run else ''}Slicing files ──")
    sliced_rows  = []   # for local + global CSV
    done         = 0
    skipped      = 0
    errors       = 0
    actual_slices_per_sub = defaultdict(int)

    for src_path, slice_fns, sub, dur in slice_plan:
        # Check if all slices already exist
        all_exist = all((output_dir / fn).exists() for fn in slice_fns)
        if all_exist:
            skipped += len(slice_fns)
            for fn in slice_fns:
                actual_slices_per_sub[sub] += 1
                sliced_rows.append((fn, sub, src_path.name))
            continue

        if not dry_run:
            try:
                y, sr = sf.read(str(src_path), dtype="float32", always_2d=False)
                # Convert stereo to mono if needed
                if y.ndim == 2:
                    y = y.mean(axis=1)

                chunks = slice_audio(y)

                for i, chunk in enumerate(chunks):
                    out_fn   = slice_fns[i] if i < len(slice_fns) else f"{src_path.stem}_s{i+1:02d}.wav"
                    out_path = output_dir / out_fn
                    sf.write(str(out_path), chunk, TARGET_SR, subtype="PCM_16")
                    done += 1
                    actual_slices_per_sub[sub] += 1
                    sliced_rows.append((out_fn, sub, src_path.name))

            except Exception as e:
                print(f"  ERROR slicing {src_path.name}: {e}")
                errors += 1
        else:
            for fn in slice_fns:
                done += 1
                actual_slices_per_sub[sub] += 1
                sliced_rows.append((fn, sub, src_path.name))

        if done % 500 == 0 and done > 0 and not dry_run:
            print(f"  Progress: {done} slices written...")

    tag = "[DRY RUN] " if dry_run else ""
    print(f"  {tag}Slices written: {done}")
    print(f"  {tag}Skipped (exist): {skipped}")
    print(f"  {tag}Errors: {errors}")

    if not dry_run:
        total_out = len(list(output_dir.glob("*.wav")))
        print(f"  Files in sliced_audio/: {total_out}")

    # ── STEP 5: Write local sliced_metadata.csv ───────────────────────────────
    print(f"\n── Step 5: {'[DRY RUN] ' if dry_run else ''}Writing sliced_metadata.csv ──")
    local_fields = ["filename", "subclass", "main_class",
                    "source_file", "sample_rate", "duration_s"]
    local_rows   = []

    for fn, sub, src_fn in sliced_rows:
        local_rows.append({
            "filename":   fn,
            "subclass":   sub,
            "main_class": SUBCLASS_TO_MAIN[sub],
            "source_file":src_fn,
            "sample_rate":TARGET_SR,
            "duration_s": TARGET_DUR,
        })

    if not dry_run:
        with open(local_meta, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=local_fields)
            writer.writeheader()
            writer.writerows(local_rows)
        print(f"  Written {len(local_rows)} rows → {local_meta.name}")
    else:
        print(f"  [DRY RUN] Would write {len(local_rows)} rows to sliced_metadata.csv")

    # ── STEP 6: Update global main_metadata.csv ───────────────────────────────
    print(f"\n── Step 6: {'[DRY RUN] ' if dry_run else ''}Updating global main_metadata.csv ──")

    # Read existing to find FSD50K source entries and replace with sliced names
    if global_meta_path.exists():
        existing_rows = []
        with open(global_meta_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_rows.append(row)

        # Remove old FSD50K filtered entries (pre-slice) and replace with sliced
        fsd50k_orig_fns = {fn for _, _, fn in sliced_rows}  # source filenames
        # Build mapping: source_file → list of slice filenames
        src_to_slices = defaultdict(list)
        for slice_fn, sub, src_fn in sliced_rows:
            src_to_slices[src_fn].append((slice_fn, sub))

        new_global = []
        replaced   = 0
        for row in existing_rows:
            if row["source"] == "FSD50K" and row["filename"] in fsd50k_orig_fns:
                # Replace with sliced versions
                src_fn = row["filename"]
                for slice_fn, sub in src_to_slices.get(src_fn, []):
                    new_global.append({
                        "filename":          slice_fn,
                        "subclass":          sub,
                        "main_class":        SUBCLASS_TO_MAIN[sub],
                        "source":            "FSD50K",
                        "original_filename": row["original_filename"],
                        "sample_rate":       TARGET_SR,
                        "duration_s":        TARGET_DUR,
                        "augmented":         "no",
                        "split":             "",
                    })
                replaced += 1
            else:
                new_global.append(row)

        if not dry_run:
            with open(global_meta_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=GLOBAL_CSV_FIELDS)
                writer.writeheader()
                writer.writerows(new_global)
            total = len(new_global)
            print(f"  Replaced {replaced} FSD50K source entries with {done} sliced entries")
            print(f"  Global CSV total rows now: {total}")
        else:
            total_new = len(existing_rows) - replaced + done
            print(f"  [DRY RUN] Would replace {replaced} source entries with ~{done} slice entries")
            print(f"  [DRY RUN] Global CSV would have ~{total_new} rows")
    else:
        print(f"  WARNING: main_metadata.csv not found — skipping global update")

    # ── STEP 7: Update registry with slice counts ─────────────────────────────
    print(f"\n── Step 7: {'[DRY RUN] ' if dry_run else ''}Updating registry ──")

    for sub, count in actual_slices_per_sub.items():
        if sub in registry:
            registry[sub]["total_originals"] = count
            registry[sub]["last_source"]     = "FSD50K-sliced"
            registry[sub]["last_updated"]    = datetime.now().strftime("%Y-%m-%d %H:%M")

    if not dry_run:
        save_registry(registry, registry_path)
        print(f"  Registry updated: {registry_path.name}")

    # ── STEP 8: Final summary ──────────────────────────────────────────────────
    print("\n── Final Summary ──")
    print(f"  {'Subclass':<20} {'Source files':>12}  {'Slices':>7}  {'Per file avg':>12}")
    print(f"  {'-'*20} {'-'*12}  {'-'*7}  {'-'*12}")
    for sub in sorted(actual_slices_per_sub):
        mc      = SUBCLASS_TO_MAIN[sub]
        n_src   = stats[sub]["files"] - stats[sub]["discarded"]
        n_slc   = actual_slices_per_sub[sub]
        avg     = f"{n_slc/n_src:.1f}" if n_src > 0 else "—"
        print(f"  {sub:<20} {n_src:>12}  {n_slc:>7}  {avg:>12}")

    total_sliced = sum(actual_slices_per_sub.values())
    print(f"\n  Total sliced clips: {total_sliced}")
    print(f"  Output folder: {output_dir}")
    print()
    print("  FSD50K slicing complete.")
    print("  Next: process birdclef_2021 (crow + owl)")

    if dry_run:
        print("\n  [DRY RUN — no files modified. Run without --dry-run to apply.]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slice FSD50K filtered clips to 5s")
    parser.add_argument("--base",        required=True, help="Path to fsd50k folder")
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
