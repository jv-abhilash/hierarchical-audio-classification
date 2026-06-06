"""
Trim Crow and Owl to 1,000 Clips Each
---------------------------------------
Crow allocation (1,000 total):
  ESC-50 + ESC-50-aug  →   80 clips  (fixed, highest quality)
  FSD50K               →   75 clips  (fixed, take all available)
  fiscro               →   60 clips  (fixed, take all — limited supply)
  comrav               →   50 clips  (5% cap — raven kept minimal)
  amecro               →  735 clips  (fill remainder to reach 1,000)

Owl allocation (1,000 total):
  RMS filter first: remove clips with mean RMS < 0.01
  (stronger than EDA's 0.001 threshold — removes low-energy silent clips)
  grhowl               →  800 clips  (80% — iconic Great Horned Owl hoot)
  brdowl               →  200 clips  (20% — Barred Owl, acoustic diversity)

What this script does:
  1. Reads main_metadata.csv
  2. Identifies crow and owl clips with their source/species
  3. Applies RMS filter to owl clips
  4. Samples per allocation plan
  5. Deletes unneeded WAV files from birdclef_2021/sliced_audio/
  6. Removes unneeded rows from main_metadata.csv
  7. Prints final counts and saves updated CSV

Usage:
    python trim_crow_owl.py --base /path/to/datasets
    python trim_crow_owl.py --base /path/to/datasets --dry-run

Requirements:
    pip install soundfile numpy
"""

import csv
import random
import shutil
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────

CROW_TARGET  = 1000
OWL_TARGET   = 1000
OWL_RMS_MIN  = 0.01    # clips below this RMS are too silent for owl
RANDOM_SEED  = 42

# Crow allocation
CROW_ALLOC = {
    "ESC-50":     80,   # fixed — keep all ESC-50 crow originals
    "ESC-50-aug": 0,    # handled together with ESC-50 (80 includes both)
    "FSD50K":     75,   # fixed — take all available
    "fiscro":     60,   # fixed — take all (limited supply ~60)
    "comrav":     50,   # 5% cap
    "amecro":    735,   # fill to reach 1,000
}

# Owl allocation (after RMS filter)
OWL_ALLOC = {
    "grhowl": 800,  # 80%
    "brdowl": 200,  # 20%
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def get_rms(wav_path: Path) -> float:
    """Compute mean RMS of a WAV file."""
    try:
        y, _ = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        return float(np.sqrt(np.mean(y ** 2)))
    except Exception:
        return 0.0

def get_species_from_filename(filename: str) -> str:
    """
    Infer BirdCLEF species from filename counter range.
    BirdCLEF crow files are named crow_NNN_sXX.wav
    We track which counter ranges belong to which ebird_code
    by reading birdclef_metadata.csv.
    """
    return None  # handled via metadata CSV

# ── MAIN ──────────────────────────────────────────────────────────────────────

def run(base_path: Path, dry_run: bool):
    metadata_csv  = base_path / "main_metadata.csv"
    backup_csv    = base_path / "main_metadata_before_crow_owl_trim.csv"
    birdclef_meta = base_path / "birdclef_2021" / "birdclef_metadata.csv"
    sliced_dir    = base_path / "birdclef_2021" / "sliced_audio"

    if not metadata_csv.exists():
        print(f"ERROR: {metadata_csv} not found")
        return

    random.seed(RANDOM_SEED)

    # ── STEP 1: Load main_metadata.csv ───────────────────────────────────────
    print("\n── Step 1: Loading main_metadata.csv ──")
    all_rows   = []
    fieldnames = None

    with open(metadata_csv, newline="", encoding="utf-8") as f:
        reader     = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            all_rows.append(row)

    print(f"  Total rows: {len(all_rows)}")

    # Separate crow and owl rows
    crow_rows  = [r for r in all_rows if r["subclass"] == "crow"]
    owl_rows   = [r for r in all_rows if r["subclass"] == "owl"]
    other_rows = [r for r in all_rows
                  if r["subclass"] not in ("crow", "owl")]

    print(f"  Crow rows: {len(crow_rows)}")
    print(f"  Owl rows:  {len(owl_rows)}")
    print(f"  Other rows: {len(other_rows)}")

    # ── STEP 2: Load BirdCLEF metadata for species mapping ───────────────────
    print("\n── Step 2: Loading BirdCLEF species mapping ──")
    # Map filename → ebird_code
    fn_to_species = {}

    if birdclef_meta.exists():
        with open(birdclef_meta, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fn  = row.get("filename", "").strip()
                code = row.get("ebird_code", "").strip()
                if fn and code:
                    fn_to_species[fn] = code
        print(f"  Loaded {len(fn_to_species)} BirdCLEF filename→species mappings")
    else:
        print(f"  WARNING: birdclef_metadata.csv not found at {birdclef_meta}")
        print(f"  Will infer species from original_filename column")

    def get_species(row):
        """Get ebird_code for a BirdCLEF row."""
        fn   = row.get("filename", "")
        orig = row.get("original_filename", "")

        # Try direct lookup
        if fn in fn_to_species:
            return fn_to_species[fn]

        # Try from original_filename (e.g. XC110263.mp3)
        if orig in fn_to_species:
            return fn_to_species[orig]

        # Infer from original_filename pattern
        # birdclef_metadata has source_file = XC110263.mp3
        # Check if orig matches any known species
        for key, code in fn_to_species.items():
            if orig and orig in key:
                return code

        return "unknown"

    # ── STEP 3: Classify crow rows by species/source ─────────────────────────
    print("\n── Step 3: Classifying crow rows by species/source ──")
    crow_by_bucket = defaultdict(list)

    for row in crow_rows:
        source = row.get("source", "")
        if source in ("ESC-50", "ESC-50-aug"):
            crow_by_bucket[source].append(row)
        elif source == "FSD50K":
            crow_by_bucket["FSD50K"].append(row)
        elif source == "BirdCLEF2021":
            species = get_species(row)
            if species in ("amecro", "fiscro", "comrav"):
                crow_by_bucket[species].append(row)
            else:
                # Fallback — count uncategorised as amecro
                crow_by_bucket["amecro"].append(row)
        else:
            crow_by_bucket["other"].append(row)

    print(f"  Crow by bucket:")
    for bucket in ["ESC-50", "ESC-50-aug", "FSD50K",
                   "amecro", "fiscro", "comrav", "other"]:
        print(f"    {bucket:<12} {len(crow_by_bucket[bucket]):>5} clips")

    # ── STEP 4: Sample crow per allocation ───────────────────────────────────
    print("\n── Step 4: Sampling crow clips per allocation ──")
    crow_kept = []

    # ESC-50 originals + aug together = 80
    esc50_all = crow_by_bucket["ESC-50"] + crow_by_bucket["ESC-50-aug"]
    esc50_take = min(80, len(esc50_all))
    crow_kept.extend(random.sample(esc50_all, esc50_take))
    print(f"  ESC-50 (orig+aug):  kept {esc50_take}/{len(esc50_all)}")

    # FSD50K — take all up to 75
    fsd_take = min(75, len(crow_by_bucket["FSD50K"]))
    crow_kept.extend(random.sample(crow_by_bucket["FSD50K"], fsd_take)
                     if fsd_take > 0 else [])
    print(f"  FSD50K:             kept {fsd_take}/{len(crow_by_bucket['FSD50K'])}")

    # fiscro — take all up to 60
    fiscro_take = min(60, len(crow_by_bucket["fiscro"]))
    crow_kept.extend(random.sample(crow_by_bucket["fiscro"], fiscro_take)
                     if fiscro_take > 0 else [])
    print(f"  fiscro (Fish Crow): kept {fiscro_take}/{len(crow_by_bucket['fiscro'])}")

    # comrav — max 50 (5% of 1000)
    comrav_take = min(50, len(crow_by_bucket["comrav"]))
    crow_kept.extend(random.sample(crow_by_bucket["comrav"], comrav_take)
                     if comrav_take > 0 else [])
    print(f"  comrav (Raven):     kept {comrav_take}/{len(crow_by_bucket['comrav'])}")

    # amecro — fill remainder to reach CROW_TARGET
    current     = len(crow_kept)
    amecro_need = CROW_TARGET - current
    amecro_take = min(amecro_need, len(crow_by_bucket["amecro"]))
    crow_kept.extend(random.sample(crow_by_bucket["amecro"], amecro_take)
                     if amecro_take > 0 else [])
    print(f"  amecro (Am. Crow):  kept {amecro_take}/{len(crow_by_bucket['amecro'])}")

    print(f"\n  Total crow kept: {len(crow_kept)} / target {CROW_TARGET}")

    # Identify crow rows to discard
    kept_crow_fns   = set(r["filename"] for r in crow_kept)
    crow_discarded  = [r for r in crow_rows
                       if r["filename"] not in kept_crow_fns]
    print(f"  Crow to discard: {len(crow_discarded)}")

    # ── STEP 5: RMS filter + sample owl clips ─────────────────────────────────
    print(f"\n── Step 5: RMS filtering owl clips (threshold={OWL_RMS_MIN}) ──")

    # Classify owl by species
    owl_by_species = defaultdict(list)
    for row in owl_rows:
        if row.get("source") == "BirdCLEF2021":
            species = get_species(row)
            if species in ("grhowl", "brdowl"):
                owl_by_species[species].append(row)
            else:
                owl_by_species["grhowl"].append(row)  # default to grhowl
        else:
            owl_by_species["grhowl"].append(row)

    print(f"  grhowl pool: {len(owl_by_species['grhowl'])} clips")
    print(f"  brdowl pool: {len(owl_by_species['brdowl'])} clips")

    def rms_filter(rows, min_rms, sliced_dir):
        """Filter rows keeping only clips above min_rms threshold."""
        passed = []
        failed = 0
        checked = 0
        for row in rows:
            wav_path = sliced_dir / row["filename"]
            if not wav_path.exists():
                # Try other audio directories
                passed.append(row)  # keep if can't check
                continue
            rms = get_rms(wav_path)
            if rms >= min_rms:
                passed.append(row)
            else:
                failed += 1
            checked += 1
            if checked % 500 == 0:
                print(f"    RMS checked: {checked}...")
        print(f"    Passed RMS filter: {len(passed)}, Failed: {failed}")
        return passed

    print(f"  Filtering grhowl...")
    grhowl_filtered = rms_filter(
        owl_by_species["grhowl"], OWL_RMS_MIN, sliced_dir)

    print(f"  Filtering brdowl...")
    brdowl_filtered = rms_filter(
        owl_by_species["brdowl"], OWL_RMS_MIN, sliced_dir)

    # Sample per allocation
    print(f"\n  Sampling owl per 80/20 allocation...")
    owl_kept = []

    grhowl_take = min(OWL_ALLOC["grhowl"], len(grhowl_filtered))
    owl_kept.extend(random.sample(grhowl_filtered, grhowl_take))
    print(f"  grhowl: kept {grhowl_take}/{len(grhowl_filtered)} (post-RMS)")

    brdowl_take = min(OWL_ALLOC["brdowl"], len(brdowl_filtered))
    owl_kept.extend(random.sample(brdowl_filtered, brdowl_take))
    print(f"  brdowl: kept {brdowl_take}/{len(brdowl_filtered)} (post-RMS)")

    total_owl = len(owl_kept)
    print(f"\n  Total owl kept: {total_owl} / target {OWL_TARGET}")

    # Identify owl rows to discard
    kept_owl_fns  = set(r["filename"] for r in owl_kept)
    owl_discarded = [r for r in owl_rows
                     if r["filename"] not in kept_owl_fns]
    print(f"  Owl to discard: {len(owl_discarded)}")

    # ── STEP 6: Delete discarded WAV files ────────────────────────────────────
    all_discarded = crow_discarded + owl_discarded
    print(f"\n── Step 6: {'[DRY RUN] ' if dry_run else ''}Deleting {len(all_discarded)} WAV files ──")

    deleted    = 0
    not_found  = 0
    errors     = 0

    # Build search paths
    search_dirs = [
        base_path / "birdclef_2021" / "sliced_audio",
        base_path / "ESC-50-master" / "audio",
        base_path / "fsd50k" / "sliced_audio",
    ]

    for row in all_discarded:
        fn = row["filename"]
        found = False
        for d in search_dirs:
            wav = d / fn
            if wav.exists():
                found = True
                if not dry_run:
                    try:
                        wav.unlink()
                        deleted += 1
                    except Exception as e:
                        print(f"  ERROR: {fn}: {e}")
                        errors += 1
                else:
                    deleted += 1
                break
        if not found:
            not_found += 1

    tag = "[DRY RUN] " if dry_run else ""
    print(f"  {tag}Deleted:   {deleted}")
    print(f"  {tag}Not found: {not_found}")
    print(f"  {tag}Errors:    {errors}")

    # ── STEP 7: Update main_metadata.csv ──────────────────────────────────────
    print(f"\n── Step 7: {'[DRY RUN] ' if dry_run else ''}Updating main_metadata.csv ──")

    new_rows = other_rows + crow_kept + owl_kept
    print(f"  New total rows: {len(new_rows)}")

    # Verify counts
    final_counts = defaultdict(int)
    for r in new_rows:
        final_counts[r["subclass"]] += 1

    print(f"\n  Final counts after trim:")
    for sub in ["crow", "owl"]:
        print(f"    {sub:<10} {final_counts[sub]:>6}")

    if not dry_run:
        shutil.copy(metadata_csv, backup_csv)
        print(f"\n  Backed up to: {backup_csv.name}")

        with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(new_rows)

        verify = sum(1 for _ in open(metadata_csv)) - 1
        print(f"  Saved {verify} rows to main_metadata.csv")

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    print("\n══ Final Summary ══════════════════════════════════════════")
    print(f"  Crow: {len(crow_rows)} → {len(crow_kept)} clips")
    print(f"    ESC-50+aug:  {esc50_take}")
    print(f"    FSD50K:      {fsd_take}")
    print(f"    fiscro:      {fiscro_take}")
    print(f"    comrav (5%): {comrav_take}")
    print(f"    amecro:      {amecro_take}")
    print()
    print(f"  Owl:  {len(owl_rows)} → {len(owl_kept)} clips")
    print(f"    grhowl (80%): {grhowl_take}")
    print(f"    brdowl (20%): {brdowl_take}")
    print(f"    RMS filter:   removed low-energy clips < {OWL_RMS_MIN}")
    print()
    print(f"  WAV files deleted: {deleted}")
    print(f"  main_metadata.csv: {len(new_rows)} rows")
    print()
    print("  Next: run remove_flagged.py to remove silent/clipped clips")

    if dry_run:
        print("\n  [DRY RUN — no files modified. Run without --dry-run to apply.]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trim crow and owl to 1,000 clips each")
    parser.add_argument("--base",    required=True,
                        help="Path to datasets root folder")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = Path(args.base)
    if not base.exists():
        print(f"ERROR: {base} does not exist")
        exit(1)

    run(base, args.dry_run)
