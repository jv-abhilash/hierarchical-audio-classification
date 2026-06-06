"""
FSD50K Filter + Rename + Metadata Script
------------------------------------------
Filters FSD50K dev set by AudioSet label tokens,
copies matching files to a staging folder with clean subclass naming,
and appends to global main_metadata.csv.

Also reads and writes filename_registry.csv — a single source of truth
for the highest counter used per subclass across ALL datasets. Every
script in this pipeline reads the registry at start and updates it at
end so filenames never collide regardless of processing order.

Label filter strategy (confirmed from audit):
  crow          → 'Crow' in labels                              (~75 clips)
  frog          → 'Frog' in labels                             (~72 clips)
  insects       → 'Insect' in labels                           (~383 clips)
  rain          → 'Rain' in labels                             (~500 clips)
  sea_waves     → 'Ocean' OR 'Waves_and_surf' in labels        (~250 clips dedup)
  wind          → 'Wind' in labels                             (~294 clips)
  crackling_fire→ 'Fire' AND 'Crackle' BOTH in labels          (~100-150 clips)
  car_horn      → 'Vehicle_horn_and_car_horn_and_honking'      (~115 clips)
  siren         → 'Siren' in labels                            (~77 clips)
  engine_idling → 'Engine' AND 'Idling' BOTH in labels         (~107 clips)
  owl           → NOT PRESENT — skipped
  jackhammer    → NOT PRESENT — skipped

Output staging folder: fsd50k/filtered_audio/
Registry file:         datasets/filename_registry.csv

Usage:
    python filter_fsd50k.py --base ./fsd50k --global-meta ./main_metadata.csv --registry ./filename_registry.csv
    python filter_fsd50k.py --base ./fsd50k --global-meta ./main_metadata.csv --registry ./filename_registry.csv --dry-run

Requirements:
    pip install pandas soundfile
"""

import csv
import shutil
import argparse
import pandas as pd
import soundfile as sf
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ── CONFIG ───────────────────────────────────────────────────────────────────

LABEL_FILTERS = {
    "crow": {
        "include":     ["Crow"],
        "require_all": False,
        "exclude":     [],
    },
    "frog": {
        "include":     ["Frog"],
        "require_all": False,
        "exclude":     [],
    },
    "insects": {
        "include":     ["Insect"],
        "require_all": False,
        "exclude":     ["Fireworks", "Explosion", "Gunshot"],
    },
    "rain": {
        "include":     ["Rain"],
        "require_all": False,
        "exclude":     ["Fireworks", "Explosion", "Music", "Speech"],
    },
    "sea_waves": {
        "include":     ["Ocean", "Waves_and_surf"],
        "require_all": False,
        "exclude":     ["Music", "Speech", "Fireworks"],
    },
    "wind": {
        "include":     ["Wind"],
        "require_all": False,
        "exclude":     ["Fireworks", "Explosion", "Music", "Speech"],
    },
    "crackling_fire": {
        "include":     ["Fire", "Crackle"],
        "require_all": True,
        "exclude":     ["Fireworks", "Explosion", "Gunshot"],
    },
    "car_horn": {
        "include":     ["Vehicle_horn_and_car_horn_and_honking"],
        "require_all": False,
        "exclude":     [],
    },
    "siren": {
        "include":     ["Siren"],
        "require_all": False,
        "exclude":     [],
    },
    "engine_idling": {
        "include":     ["Engine", "Idling"],
        "require_all": True,
        "exclude":     ["Music", "Speech"],
    },
}

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

# All 12 subclasses in the project — registry tracks all of them
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

TARGET_SR = 22050

# ── REGISTRY FUNCTIONS ────────────────────────────────────────────────────────

def load_registry(registry_path: Path) -> dict:
    """
    Load filename_registry.csv into a dict keyed by subclass.
    If file does not exist, build it fresh from main_metadata.csv or return zeros.
    Returns: {subclass: {last_counter, last_filename, total_originals, total_augmented, ...}}
    """
    registry = {}

    # Initialise all subclasses to zero
    for sub in ALL_SUBCLASSES:
        registry[sub] = {
            "subclass":        sub,
            "last_counter":    0,
            "last_filename":   "—",
            "total_originals": 0,
            "total_augmented": 0,
            "last_source":     "—",
            "last_updated":    "—",
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
        print(f"  Loaded registry: {registry_path}")
    else:
        print(f"  Registry not found — starting from zero: {registry_path}")

    return registry


def save_registry(registry: dict, registry_path: Path):
    """Write registry dict back to CSV."""
    with open(registry_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        for sub in ALL_SUBCLASSES:
            writer.writerow(registry[sub])


def build_registry_from_metadata(global_meta_path: Path, registry_path: Path):
    """
    One-time utility: build registry from existing main_metadata.csv.
    Called automatically if registry does not exist but metadata does.
    """
    if not global_meta_path.exists():
        return

    print(f"  Building registry from existing main_metadata.csv...")
    counters   = defaultdict(int)
    originals  = defaultdict(int)
    augmented  = defaultdict(int)
    last_src   = defaultdict(str)
    last_fn    = defaultdict(str)

    with open(global_meta_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fn  = row["filename"]
            sub = row["subclass"]
            src = row["source"]
            aug = row.get("augmented", "no")

            # Parse counter from filename e.g. crow_041.wav → 41
            base = fn.replace(f"{sub}_", "").replace("_aug.wav", "").replace(".wav", "")
            try:
                num = int(base)
                if aug == "no" and num > counters[sub]:
                    counters[sub]  = num
                    last_fn[sub]   = fn
                    last_src[sub]  = src
            except ValueError:
                pass

            if aug == "yes":
                augmented[sub] += 1
            else:
                originals[sub] += 1

    registry = {}
    for sub in ALL_SUBCLASSES:
        registry[sub] = {
            "subclass":        sub,
            "last_counter":    counters[sub],
            "last_filename":   last_fn[sub] if last_fn[sub] else "—",
            "total_originals": originals[sub],
            "total_augmented": augmented[sub],
            "last_source":     last_src[sub] if last_src[sub] else "—",
            "last_updated":    datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

    save_registry(registry, registry_path)
    print(f"  Registry built and saved: {registry_path}")
    for sub in ALL_SUBCLASSES:
        r = registry[sub]
        print(f"    {sub:<20} counter: {r['last_counter']:>3}  "
              f"originals: {r['total_originals']:>3}  "
              f"augmented: {r['total_augmented']:>3}  "
              f"last: {r['last_filename']}")

    return registry


# ── HELPERS ───────────────────────────────────────────────────────────────────

def matches_filter(label_str: str, rule: dict) -> bool:
    tokens = set(t.strip() for t in label_str.split(","))
    if rule["require_all"]:
        if not all(inc in tokens for inc in rule["include"]):
            return False
    else:
        if not any(inc in tokens for inc in rule["include"]):
            return False
    if any(exc in tokens for exc in rule["exclude"]):
        return False
    return True


def get_audio_info(path: Path):
    try:
        info = sf.info(str(path))
        return int(info.samplerate), round(info.duration, 3)
    except Exception:
        return None, None


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run(base_path: Path, global_meta_path: Path, registry_path: Path, dry_run: bool):
    dev_csv    = base_path / "FSD50K.ground_truth" / "dev.csv"
    audio_dir  = base_path / "FSD50K.dev_audio"
    output_dir = base_path / "filtered_audio"

    if not dev_csv.exists():
        print(f"ERROR: dev.csv not found at {dev_csv}")
        return
    if not audio_dir.exists():
        print(f"ERROR: FSD50K.dev_audio not found at {audio_dir}")
        return

    if not dry_run:
        output_dir.mkdir(exist_ok=True)

    # ── STEP 1: Load / build registry ────────────────────────────────────────
    print("\n── Step 1: Loading filename registry ──")
    if not registry_path.exists() and global_meta_path.exists():
        print("  Registry missing — building from main_metadata.csv")
        registry = build_registry_from_metadata(global_meta_path, registry_path
                       if not dry_run else Path("/tmp/registry_temp.csv"))
        if registry is None:
            registry = load_registry(registry_path)
    else:
        registry = load_registry(registry_path)

    print(f"\n  Current registry state:")
    print(f"  {'Subclass':<20} {'Last counter':>12}  {'Last file':<30}  {'Originals':>9}  {'Augmented':>9}")
    print(f"  {'-'*20} {'-'*12}  {'-'*30}  {'-'*9}  {'-'*9}")
    for sub in ALL_SUBCLASSES:
        r = registry[sub]
        print(f"  {sub:<20} {r['last_counter']:>12}  {r['last_filename']:<30}  "
              f"{r['total_originals']:>9}  {r['total_augmented']:>9}")

    # ── STEP 2: Load FSD50K dev.csv ───────────────────────────────────────────
    print("\n── Step 2: Loading FSD50K dev.csv ──")
    df = pd.read_csv(str(dev_csv))
    print(f"  Total rows: {len(df)}")

    # ── STEP 3: Filter per subclass ───────────────────────────────────────────
    print("\n── Step 3: Filtering by label rules ──")
    subclass_matches = {}
    for subclass, rule in LABEL_FILTERS.items():
        matched = df[df["labels"].apply(lambda x: matches_filter(str(x), rule))]
        matched = matched.drop_duplicates(subset=["fname"])
        subclass_matches[subclass] = matched
        inc_str = " AND ".join(rule["include"]) if rule["require_all"] else " OR ".join(rule["include"])
        exc_str = f" | exclude: {rule['exclude']}" if rule["exclude"] else ""
        print(f"  {subclass:<20} {inc_str:<55}{exc_str}  → {len(matched)} label matches")

    # ── STEP 4: Build copy plan using registry counters ───────────────────────
    print("\n── Step 4: Building copy plan (counters from registry) ──")
    copy_plan    = []
    new_counters = {sub: registry[sub]["last_counter"] for sub in ALL_SUBCLASSES}

    for subclass, matched_df in subclass_matches.items():
        start = new_counters[subclass]
        found = 0

        for _, row in matched_df.iterrows():
            fname    = str(int(row["fname"]))
            orig_fn  = f"{fname}.wav"
            src_path = audio_dir / orig_fn

            if not src_path.exists():
                continue

            new_counters[subclass] += 1
            new_fn   = f"{subclass}_{new_counters[subclass]:03d}.wav"
            dst_path = output_dir / new_fn
            copy_plan.append((src_path, dst_path, new_fn, orig_fn, subclass, row))
            found += 1

        end = new_counters[subclass]
        if found > 0:
            print(f"  {subclass:<20} registry start: {start:>3} → end: {end:>3}  "
                  f"({found} files)  "
                  f"{subclass}_{start+1:03d}.wav → {subclass}_{end:03d}.wav")
        else:
            print(f"  {subclass:<20} registry start: {start:>3}  no files found on disk")

    print(f"\n  Total files to copy: {len(copy_plan)}")

    # ── STEP 5: Copy files ────────────────────────────────────────────────────
    print(f"\n── Step 5: {'[DRY RUN] ' if dry_run else ''}Copying files ──")
    copied    = 0
    skipped   = 0
    not_found = 0

    for src_path, dst_path, new_fn, orig_fn, subclass, row in copy_plan:
        if not src_path.exists():
            not_found += 1
            continue
        if dst_path.exists():
            skipped += 1
            continue
        if not dry_run:
            shutil.copy2(str(src_path), str(dst_path))
        copied += 1
        if copied % 100 == 0 and not dry_run:
            print(f"  Progress: {copied}/{len(copy_plan)} copied...")

    tag = "[DRY RUN] " if dry_run else ""
    print(f"  {tag}Copied:    {copied}")
    print(f"  {tag}Skipped:   {skipped} (already exist)")
    print(f"  {tag}Not found: {not_found}")

    if not dry_run:
        print(f"  Files in filtered_audio/: {len(list(output_dir.glob('*.wav')))}")

    # ── STEP 6: Local filtered_metadata.csv ──────────────────────────────────
    print(f"\n── Step 6: {'[DRY RUN] ' if dry_run else ''}Writing local filtered_metadata.csv ──")
    local_meta  = base_path / "filtered_metadata.csv"
    local_fields = ["filename", "subclass", "main_class", "original_fname",
                    "fsd50k_labels", "sample_rate", "duration_s"]
    local_rows  = []

    for src_path, dst_path, new_fn, orig_fn, subclass, row in copy_plan:
        sr, dur = get_audio_info(dst_path) if (not dry_run and dst_path.exists()) else (None, None)
        local_rows.append({
            "filename":       new_fn,
            "subclass":       subclass,
            "main_class":     SUBCLASS_TO_MAIN[subclass],
            "original_fname": orig_fn,
            "fsd50k_labels":  row.get("labels", ""),
            "sample_rate":    sr if sr else "unknown",
            "duration_s":     dur if dur else "unknown",
        })

    if not dry_run:
        with open(local_meta, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=local_fields)
            writer.writeheader()
            writer.writerows(local_rows)
        print(f"  Written {len(local_rows)} rows → {local_meta.name}")
    else:
        print(f"  [DRY RUN] Would write {len(local_rows)} rows to filtered_metadata.csv")

    # ── STEP 7: Append to global main_metadata.csv ───────────────────────────
    print(f"\n── Step 7: {'[DRY RUN] ' if dry_run else ''}Updating global main_metadata.csv ──")

    existing_fns = set()
    if global_meta_path.exists():
        with open(global_meta_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                existing_fns.add(r["filename"])

    global_rows = []
    for src_path, dst_path, new_fn, orig_fn, subclass, row in copy_plan:
        if new_fn in existing_fns:
            continue
        sr, dur = get_audio_info(dst_path) if (not dry_run and dst_path.exists()) else (TARGET_SR, None)
        global_rows.append({
            "filename":          new_fn,
            "subclass":          subclass,
            "main_class":        SUBCLASS_TO_MAIN[subclass],
            "source":            "FSD50K",
            "original_filename": orig_fn,
            "sample_rate":       sr if sr else TARGET_SR,
            "duration_s":        dur if dur else "variable",
            "augmented":         "no",
            "split":             "",
        })

    if not dry_run:
        write_header = not global_meta_path.exists()
        with open(global_meta_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=GLOBAL_CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerows(global_rows)
        total = sum(1 for _ in open(global_meta_path)) - 1
        print(f"  Appended {len(global_rows)} rows — global total now: {total}")
    else:
        print(f"  [DRY RUN] Would append {len(global_rows)} rows to main_metadata.csv")

    # ── STEP 8: Update registry ───────────────────────────────────────────────
    print(f"\n── Step 8: {'[DRY RUN] ' if dry_run else ''}Updating filename registry ──")

    sub_counts = defaultdict(int)
    for _, _, new_fn, _, subclass, _ in copy_plan:
        sub_counts[subclass] += 1

    for sub in ALL_SUBCLASSES:
        if sub in new_counters and new_counters[sub] != registry[sub]["last_counter"]:
            added = new_counters[sub] - registry[sub]["last_counter"]
            registry[sub]["last_counter"]    = new_counters[sub]
            registry[sub]["last_filename"]   = f"{sub}_{new_counters[sub]:03d}.wav"
            registry[sub]["total_originals"] = registry[sub]["total_originals"] + added
            registry[sub]["last_source"]     = "FSD50K"
            registry[sub]["last_updated"]    = datetime.now().strftime("%Y-%m-%d %H:%M")

    if not dry_run:
        save_registry(registry, registry_path)
        print(f"  Registry saved: {registry_path}")
        print(f"\n  Updated registry state:")
        print(f"  {'Subclass':<20} {'Counter':>7}  {'Last file':<30}  {'Originals':>9}")
        print(f"  {'-'*20} {'-'*7}  {'-'*30}  {'-'*9}")
        for sub in ALL_SUBCLASSES:
            r = registry[sub]
            marker = " ◄ updated" if sub in sub_counts else ""
            print(f"  {sub:<20} {r['last_counter']:>7}  {r['last_filename']:<30}  "
                  f"{r['total_originals']:>9}{marker}")
    else:
        print(f"  [DRY RUN] Would update registry for: {sorted(sub_counts.keys())}")

    # ── STEP 9: Final summary ─────────────────────────────────────────────────
    print("\n── Final Summary ──")
    total = 0
    for sub in sorted(sub_counts):
        mc = SUBCLASS_TO_MAIN[sub]
        n  = sub_counts[sub]
        total += n
        print(f"  {mc:<12} {sub:<20} {n:>4} clips staged")

    print(f"\n  Total FSD50K clips staged: {total}")
    print(f"  Output: {output_dir}")
    print()
    print("  IMPORTANT: FSD50K clips are variable length (0.3s–30s)")
    print("  These will be sliced to 5s in a later unified slicing step")
    print("  Next dataset: birdclef_2021 (crow + owl subclasses)")

    if dry_run:
        print("\n  [DRY RUN — no files modified. Run without --dry-run to apply.]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter FSD50K and update registry")
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
