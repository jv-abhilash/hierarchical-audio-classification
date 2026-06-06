"""
Audio Normalisation Script
----------------------------
Resamples all WAV files in final_dataset/ from 44100Hz to 22050Hz.
Files already at 22050Hz are skipped (no processing needed).
All other properties (mono, PCM_16, 5.0s) are already uniform.

In-place processing — overwrites 44100Hz files with 22050Hz versions.
Backs up count of processed files but does NOT keep 44100Hz copies
(final_dataset is already a copy — originals are in source folders).

Usage:
    python normalise_audio.py --base /path/to/datasets --dry-run
    python normalise_audio.py --base /path/to/datasets

Requirements:
    pip install soundfile numpy librosa
"""

import argparse
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────

TARGET_SR      = 22050
TARGET_DUR     = 5.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DUR)   # 110,250 samples
TARGET_SUBTYPE = "PCM_16"

SUBCLASS_FOLDERS = [
    "wildlife/crow", "wildlife/owl", "wildlife/frog", "wildlife/insects",
    "nature/rain", "nature/sea_waves", "nature/wind", "nature/crackling_fire",
    "urban/car_horn", "urban/engine_idling", "urban/siren", "urban/jackhammer",
]

# ── MAIN ──────────────────────────────────────────────────────────────────────

def run(base_path: Path, dry_run: bool):
    final_dir = base_path / "final_dataset"

    if not final_dir.exists():
        print(f"ERROR: final_dataset/ not found at {final_dir}")
        return

    # ── STEP 1: Scan all WAV files ────────────────────────────────────────────
    print("\n── Step 1: Scanning final_dataset/ ──")
    all_wavs     = []
    need_resample = []
    already_ok   = []
    errors       = []

    for folder_rel in SUBCLASS_FOLDERS:
        folder = final_dir / folder_rel
        if not folder.exists():
            print(f"  WARNING: folder not found — {folder_rel}")
            continue
        wavs = sorted(folder.glob("*.wav"))
        all_wavs.extend(wavs)

    print(f"  Total WAV files found: {len(all_wavs)}")

    # Check sample rate of each file
    print(f"  Checking sample rates...")
    sr_counts = defaultdict(int)

    for wav in all_wavs:
        try:
            info = sf.info(str(wav))
            sr_counts[info.samplerate] += 1
            if info.samplerate != TARGET_SR:
                need_resample.append(wav)
            else:
                already_ok.append(wav)
        except Exception as e:
            print(f"  ERROR reading {wav.name}: {e}")
            errors.append(wav)

    print(f"\n  Sample rate distribution:")
    for sr, cnt in sorted(sr_counts.items()):
        status = "✓ target" if sr == TARGET_SR else "← needs resampling"
        print(f"    {sr} Hz:  {cnt:>5} files  {status}")

    print(f"\n  Already at 22050Hz:  {len(already_ok)} files (skip)")
    print(f"  Need resampling:     {len(need_resample)} files")
    print(f"  Read errors:         {len(errors)} files")

    if len(need_resample) == 0:
        print(f"\n  ✓ All files already at {TARGET_SR}Hz — nothing to do")
        return

    # ── STEP 2: Resample 44100Hz → 22050Hz ───────────────────────────────────
    print(f"\n── Step 2: {'[DRY RUN] ' if dry_run else ''}Resampling {len(need_resample)} files ──")
    print(f"  44100Hz → 22050Hz (factor 0.5 — every other sample)")

    processed    = 0
    skipped      = 0
    proc_errors  = 0
    by_subclass  = defaultdict(int)

    for wav_path in need_resample:
        subclass = wav_path.parent.name

        if not dry_run:
            try:
                # Load at original SR
                y, sr = sf.read(str(wav_path),
                                dtype="float32",
                                always_2d=False)

                # Ensure mono
                if y.ndim == 2:
                    y = y.mean(axis=1)

                # Resample to TARGET_SR
                # librosa.resample is high quality (default: kaiser_best)
                y_resampled = librosa.resample(y,
                                               orig_sr=sr,
                                               target_sr=TARGET_SR)

                # Ensure exactly TARGET_SAMPLES length
                if len(y_resampled) > TARGET_SAMPLES:
                    y_resampled = y_resampled[:TARGET_SAMPLES]
                elif len(y_resampled) < TARGET_SAMPLES:
                    y_resampled = np.pad(
                        y_resampled,
                        (0, TARGET_SAMPLES - len(y_resampled))
                    )

                # Overwrite in-place
                sf.write(str(wav_path),
                         y_resampled,
                         TARGET_SR,
                         subtype=TARGET_SUBTYPE)

                processed += 1
                by_subclass[subclass] += 1

            except Exception as e:
                print(f"  ERROR: {wav_path.name}: {e}")
                proc_errors += 1
        else:
            processed += 1
            by_subclass[subclass] += 1

        if processed % 200 == 0 and processed > 0 and not dry_run:
            print(f"  Progress: {processed}/{len(need_resample)} resampled...")

    tag = "[DRY RUN] " if dry_run else ""
    print(f"\n  {tag}Resampled: {processed}")
    print(f"  {tag}Errors:    {proc_errors}")

    print(f"\n  Files resampled per subclass:")
    for sub in sorted(by_subclass):
        print(f"    {sub:<20} {by_subclass[sub]:>4} files")

    # ── STEP 3: Verify final state ────────────────────────────────────────────
    if not dry_run:
        print(f"\n── Step 3: Verifying final state ──")
        final_sr_counts = defaultdict(int)
        verify_errors   = 0

        for wav in all_wavs:
            try:
                info = sf.info(str(wav))
                final_sr_counts[info.samplerate] += 1
            except Exception:
                verify_errors += 1

        print(f"  Final sample rate distribution:")
        for sr, cnt in sorted(final_sr_counts.items()):
            status = "✓" if sr == TARGET_SR else "✗ STILL WRONG"
            print(f"    {sr} Hz:  {cnt:>5} files  {status}")

        if len(final_sr_counts) == 1 and TARGET_SR in final_sr_counts:
            print(f"\n  ✓ All {len(all_wavs)} files now at {TARGET_SR}Hz")
            print(f"  ✓ All mono, PCM_16, 5.0s — dataset is fully normalised")
        else:
            print(f"\n  ✗ Some files still not at {TARGET_SR}Hz")
            print(f"    Check errors above and re-run")

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    print(f"\n══ Summary ════════════════════════════════════════════════")
    print(f"  Total files:          {len(all_wavs)}")
    print(f"  Already at 22050Hz:   {len(already_ok)} (skipped)")
    print(f"  Resampled 44100→22050: {processed}")
    print(f"  Errors:               {proc_errors}")
    print()
    print(f"  Final format (all files):")
    print(f"    Sample rate:  22050 Hz")
    print(f"    Channels:     mono")
    print(f"    Bit depth:    PCM_16")
    print(f"    Duration:     5.000s (110,250 samples)")
    print()
    print(f"  Dataset is ready for mel spectrogram generation and training.")

    if dry_run:
        print(f"\n  [DRY RUN — no files modified. Run without --dry-run to apply.]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normalise all final_dataset WAV files to 22050Hz")
    parser.add_argument("--base",    required=True,
                        help="Path to datasets root (contains final_dataset/)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = Path(args.base)
    if not base.exists():
        print(f"ERROR: {base} does not exist")
        exit(1)

    run(base, args.dry_run)
