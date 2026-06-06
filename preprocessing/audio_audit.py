#!/usr/bin/env python3
"""
Comprehensive Audio Dataset Audit Script
Scans dataset folders for audio file counts, quality metrics, and subclass relevance.
"""

import os
import sys
import random
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import soundfile as sf

warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────────────
BASE = Path("/home/abhi-ubuntu-pc/Datascience, ML and DL/data_science/dataset")

FOLDERS = [
    "ESC-50-master",
    "birdclef_2021",
    "anuraset",
    "fsd50k",
    "freefield1010",
    "FSC22_forest",
    "forest_wild_fire_sound_dataset",
    "99Sounds Nature Sounds",
    "urbansound8k",
    "voice_of_birds",
]

SKIP_QUALITY = {"voice_of_birds"}  # count only, no quality check

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff"}

SAMPLE_N = 200

# Subclass mapping — which subclasses each folder can feed
SUBCLASS_MAP = {
    "ESC-50-master": {
        "crow": ["crow"],
        "owl": ["owl"],
        "frog": ["frog"],
        "insects": ["insects"],
        "rain": ["rain"],
        "sea_waves": ["sea_waves"],
        "wind": ["wind"],
        "crackling_fire": ["crackling_fire"],
        "car_horn": ["car_horn"],
        "engine_idling": ["engine_idling"],
        "siren": ["siren"],
        "jackhammer": ["jackhammer"],
    },
    "birdclef_2021": {
        "crow": ["brcow1", "amecro"],
        "owl": ["brnowl", "grhowl", "brdowl", "easowl1", "snoowl1"],
    },
    "anuraset": {
        "frog": ["ALL"],  # entire dataset is frog/anuran calls
    },
    "fsd50k": {
        "crow": ["Crow"],
        "owl": ["Owl"],
        "frog": ["Frog"],
        "insects": ["Insect", "Cricket", "Mosquito", "Fly_housefly"],
        "rain": ["Rain", "Raindrop"],
        "sea_waves": ["Ocean", "Waves_surf"],
        "wind": ["Wind", "Rustling_leaves"],
        "crackling_fire": ["Fire", "Crackle"],
        "car_horn": ["Car_horn", "Vehicle_horn"],
        "engine_idling": ["Engine", "Idling"],
        "siren": ["Siren", "Emergency_vehicle", "Ambulance_siren", "Fire_engine_siren", "Police_car_siren"],
        "jackhammer": ["Jackhammer"],
    },
    "freefield1010": {
        "crow": ["bird-related"],
        "owl": ["bird-related"],
        "rain": ["rain-related"],
        "wind": ["wind-related"],
    },
    "FSC22_forest": {
        "insects": ["insects/cicadas"],
        "rain": ["rain"],
        "wind": ["wind"],
        "crackling_fire": ["fire"],
    },
    "forest_wild_fire_sound_dataset": {
        "crackling_fire": ["fire_sounds"],
    },
    "99Sounds Nature Sounds": {
        "rain": ["rain"],
        "sea_waves": ["ocean/sea"],
        "wind": ["wind"],
        "insects": ["nature_ambience"],
    },
    "urbansound8k": {
        "car_horn": ["car_horn"],
        "engine_idling": ["engine_idling"],
        "siren": ["siren"],
        "jackhammer": ["jackhammer"],
    },
    "voice_of_birds": {
        "crow": ["crow-species"],
        "owl": ["owl-species"],
    },
}

# Estimated clip counts per subclass per folder (based on known metadata/structure)
ESTIMATED_CLIPS = {
    "ESC-50-master": {
        "crow": 40, "owl": 40, "frog": 40, "insects": 40,
        "rain": 40, "sea_waves": 40, "wind": 40, "crackling_fire": 40,
        "car_horn": 40, "engine_idling": 40, "siren": 40, "jackhammer": 40,
    },
    "birdclef_2021": {
        "crow": 150, "owl": 250,
    },
    "anuraset": {
        "frog": 10000,
    },
    "fsd50k": {
        "crow": 80, "owl": 50, "frog": 120, "insects": 350,
        "rain": 300, "sea_waves": 150, "wind": 400,
        "crackling_fire": 90, "car_horn": 200, "engine_idling": 280,
        "siren": 350, "jackhammer": 120,
    },
    "freefield1010": {
        "crow": 200, "owl": 100, "rain": 150, "wind": 180,
    },
    "FSC22_forest": {
        "insects": 500, "rain": 300, "wind": 400, "crackling_fire": 200,
    },
    "forest_wild_fire_sound_dataset": {
        "crackling_fire": 500,
    },
    "99Sounds Nature Sounds": {
        "rain": 30, "sea_waves": 20, "wind": 25, "insects": 15,
    },
    "urbansound8k": {
        "car_horn": 429, "engine_idling": 930, "siren": 929, "jackhammer": 1000,
    },
    "voice_of_birds": {
        "crow": 100, "owl": 80,
    },
}


# ─── Helper Functions ────────────────────────────────────────────────────────

def find_audio_files(folder_path):
    """Recursively find all audio files and return paths + extension breakdown."""
    audio_files = []
    ext_counts = Counter()
    total_bytes = 0

    for root, _, files in os.walk(folder_path):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in AUDIO_EXTS:
                fpath = os.path.join(root, fname)
                audio_files.append(fpath)
                ext_counts[ext] += 1
                try:
                    total_bytes += os.path.getsize(fpath)
                except OSError:
                    pass

    return audio_files, ext_counts, total_bytes


def format_size(nbytes):
    """Format byte count to human-readable."""
    if nbytes >= 1 << 30:
        return f"{nbytes / (1 << 30):.2f} GB"
    elif nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.1f} MB"
    else:
        return f"{nbytes / (1 << 10):.1f} KB"


def sample_audio_quality(audio_files, n=200):
    """Sample up to n files and extract sample rate, channels, duration, bit depth."""
    if not audio_files:
        return {}, {}, [], []

    # Filter to soundfile-readable formats (wav, flac, ogg, aiff — not mp3/m4a natively)
    sf_readable = [f for f in audio_files if os.path.splitext(f)[1].lower() in {".wav", ".flac", ".ogg", ".aiff"}]
    mp3_files = [f for f in audio_files if os.path.splitext(f)[1].lower() == ".mp3"]
    m4a_files = [f for f in audio_files if os.path.splitext(f)[1].lower() == ".m4a"]

    sample_pool = sf_readable.copy()
    random.seed(42)

    if len(sample_pool) > n:
        sample_pool = random.sample(sample_pool, n)
    elif len(sample_pool) < n and mp3_files:
        # Fill remaining with mp3 files using mutagen
        remaining = n - len(sample_pool)
        mp3_sample = random.sample(mp3_files, min(remaining, len(mp3_files)))
        sample_pool_mp3 = mp3_sample
    else:
        sample_pool_mp3 = []

    if len(sample_pool) < n and mp3_files:
        remaining = n - len(sample_pool)
        sample_pool_mp3 = random.sample(mp3_files, min(remaining, len(mp3_files)))
    else:
        sample_pool_mp3 = []

    sr_counts = Counter()
    ch_counts = Counter()
    durations = []
    bit_depths = []

    # Process soundfile-readable files
    for fpath in sample_pool:
        try:
            info = sf.info(fpath)
            sr_counts[info.samplerate] += 1
            ch_label = "mono" if info.channels == 1 else "stereo" if info.channels == 2 else f"{info.channels}ch"
            ch_counts[ch_label] += 1
            durations.append(info.duration)
            # Extract bit depth from subtype
            subtype = info.subtype
            if "16" in subtype:
                bit_depths.append(16)
            elif "24" in subtype:
                bit_depths.append(24)
            elif "32" in subtype:
                bit_depths.append(32)
            elif "FLOAT" in subtype.upper():
                bit_depths.append(32)
            elif "8" in subtype:
                bit_depths.append(8)
            else:
                bit_depths.append(16)  # default assumption
        except Exception:
            pass

    # Process MP3 files with mutagen
    if sample_pool_mp3:
        try:
            from mutagen.mp3 import MP3
            for fpath in sample_pool_mp3:
                try:
                    audio = MP3(fpath)
                    sr = audio.info.sample_rate
                    sr_counts[sr] += 1
                    ch_label = "mono" if audio.info.channels == 1 else "stereo" if audio.info.channels == 2 else f"{audio.info.channels}ch"
                    ch_counts[ch_label] += 1
                    durations.append(audio.info.length)
                    bit_depths.append(audio.info.bitrate // 1000)  # kbps for mp3
                except Exception:
                    pass
        except ImportError:
            pass

    return sr_counts, ch_counts, durations, bit_depths


def assess_quality_tier(sr_counts, ext_counts, bit_depths):
    """Assign HIGH / MEDIUM / LOW quality tier."""
    if not sr_counts:
        return "LOW"

    dominant_sr = sr_counts.most_common(1)[0][0] if sr_counts else 0
    has_wav = ext_counts.get(".wav", 0) > 0
    has_flac = ext_counts.get(".flac", 0) > 0
    has_mp3 = ext_counts.get(".mp3", 0) > 0
    has_m4a = ext_counts.get(".m4a", 0) > 0
    has_webm = ext_counts.get(".webm", 0) > 0
    mixed_sr = len(sr_counts) > 2

    total_files = sum(ext_counts.values())
    wav_flac_ratio = (ext_counts.get(".wav", 0) + ext_counts.get(".flac", 0)) / max(total_files, 1)

    if dominant_sr >= 44100 and wav_flac_ratio >= 0.8 and not mixed_sr:
        return "HIGH"
    elif dominant_sr < 22050 or has_webm or has_m4a:
        return "LOW"
    else:
        return "MEDIUM"


# ─── Main Audit ──────────────────────────────────────────────────────────────

def audit_folder(folder_name):
    """Run full audit on a single folder."""
    folder_path = BASE / folder_name
    skip_quality = folder_name in SKIP_QUALITY

    print(f"\n{'='*80}")
    print(f"  FOLDER: {folder_name}")
    print(f"{'='*80}")

    # STEP 1 — Total Count
    audio_files, ext_counts, total_bytes = find_audio_files(folder_path)
    total_files = len(audio_files)

    print(f"\n  STEP 1 — TOTAL COUNT")
    print(f"  Total audio files: {total_files:,}")
    print(f"  Breakdown by extension:")
    for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1]):
        print(f"    {ext:8s} : {count:>8,}")
    print(f"  Disk size: {format_size(total_bytes)}")

    if skip_quality:
        print(f"\n  [SKIPPED — quality check not requested for {folder_name}]")
        # Still compute basic info for summary
        dominant_ext = ext_counts.most_common(1)[0][0] if ext_counts else "N/A"
        subclasses = list(SUBCLASS_MAP.get(folder_name, {}).keys())
        return {
            "folder": folder_name,
            "total_files": total_files,
            "format": dominant_ext,
            "sample_rates": "skipped",
            "avg_duration": "skipped",
            "disk_size": format_size(total_bytes),
            "quality_tier": "skipped",
            "subclasses": subclasses,
            "ext_counts": ext_counts,
            "total_bytes": total_bytes,
        }

    # STEP 2 — Sample Rate Distribution
    sr_counts, ch_counts, durations, bit_depths = sample_audio_quality(audio_files, SAMPLE_N)

    print(f"\n  STEP 2 — SAMPLE RATE DISTRIBUTION (from {min(SAMPLE_N, len(audio_files))} random samples)")
    print(f"  Sample rates:")
    for sr, count in sorted(sr_counts.items()):
        print(f"    {sr:>6d} Hz : {count:>4d}")
    print(f"  Channels:")
    for ch, count in sorted(ch_counts.items()):
        print(f"    {ch:>8s} : {count:>4d}")
    if durations:
        print(f"  Duration (seconds):")
        print(f"    min: {min(durations):.2f}s  |  max: {max(durations):.2f}s  |  avg: {sum(durations)/len(durations):.2f}s")
    if bit_depths:
        bd_counts = Counter(bit_depths)
        print(f"  Bit depths:")
        for bd, count in sorted(bd_counts.items()):
            print(f"    {bd:>4d}-bit : {count:>4d}")

    # STEP 3 — Quality Tier
    tier = assess_quality_tier(sr_counts, ext_counts, bit_depths)
    print(f"\n  STEP 3 — QUALITY TIER: {tier}")

    # STEP 4 — Subclass Relevance
    subclass_info = SUBCLASS_MAP.get(folder_name, {})
    clip_estimates = ESTIMATED_CLIPS.get(folder_name, {})
    subclasses_fed = list(subclass_info.keys())

    print(f"\n  STEP 4 — SUBCLASS RELEVANCE")
    if subclass_info:
        for sc, labels in subclass_info.items():
            est = clip_estimates.get(sc, "?")
            print(f"    {sc:<16s} <- {labels}  (~{est} clips)")
    else:
        print(f"    No subclass mapping defined.")

    # Build summary dict
    dominant_ext = ext_counts.most_common(1)[0][0] if ext_counts else "N/A"
    all_exts = "/".join(sorted(ext_counts.keys()))
    sr_str = ", ".join(f"{sr}Hz" for sr in sorted(sr_counts.keys()))
    avg_dur = f"{sum(durations)/len(durations):.1f}s" if durations else "N/A"

    return {
        "folder": folder_name,
        "total_files": total_files,
        "format": all_exts,
        "sample_rates": sr_str,
        "avg_duration": avg_dur,
        "disk_size": format_size(total_bytes),
        "quality_tier": tier,
        "subclasses": subclasses_fed,
        "ext_counts": ext_counts,
        "total_bytes": total_bytes,
        "durations": durations,
    }


# ─── Run All ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = []
    grand_total_files = 0
    grand_total_bytes = 0
    grand_total_5s_clips = 0

    for folder_name in FOLDERS:
        folder_path = BASE / folder_name
        if not folder_path.exists():
            print(f"\n  WARNING: {folder_name} not found — skipping")
            continue
        res = audit_folder(folder_name)
        results.append(res)
        grand_total_files += res["total_files"]
        grand_total_bytes += res["total_bytes"]

        # Estimate 5s clips: total_files * (avg_duration / 5)
        if "durations" in res and res["durations"]:
            avg_d = sum(res["durations"]) / len(res["durations"])
            est_clips = int(res["total_files"] * max(avg_d / 5.0, 1.0))
        elif res["folder"] == "voice_of_birds":
            est_clips = res["total_files"]  # assume ~5s each
        else:
            est_clips = res["total_files"]
        grand_total_5s_clips += est_clips

    # ─── STEP 5 — Summary Table ──────────────────────────────────────────────
    print(f"\n\n{'='*140}")
    print(f"  STEP 5 — SUMMARY TABLE")
    print(f"{'='*140}")

    header = f"{'Folder':<35s} | {'Files':>10s} | {'Format':<14s} | {'Sample Rates':<28s} | {'Avg Dur':>8s} | {'Disk Size':>10s} | {'Tier':<8s} | Subclasses Fed"
    print(header)
    print("-" * 140)

    for r in results:
        sc_str = ", ".join(r["subclasses"]) if r["subclasses"] else "—"
        if len(sc_str) > 40:
            sc_str = sc_str[:37] + "..."
        sr_disp = r["sample_rates"] if len(str(r["sample_rates"])) <= 28 else str(r["sample_rates"])[:25] + "..."
        print(f"{r['folder']:<35s} | {r['total_files']:>10,} | {r['format']:<14s} | {sr_disp:<28s} | {r['avg_duration']:>8s} | {r['disk_size']:>10s} | {r['quality_tier']:<8s} | {sc_str}")

    print("-" * 140)
    print(f"\n  GRAND TOTAL audio files across all folders : {grand_total_files:,}")
    print(f"  GRAND TOTAL disk usage                     : {format_size(grand_total_bytes)}")
    print(f"  GRAND TOTAL estimated 5s clips after slicing: {grand_total_5s_clips:,}")
    print()
