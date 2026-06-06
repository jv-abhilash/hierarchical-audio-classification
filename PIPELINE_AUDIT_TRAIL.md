# Pipeline Audit Trail
# 3-Class Acoustic CRNN Project
# ============================================================
# This file maps every Claude Code inspection output to every
# script run in the exact order they were executed.
# Use this if something goes wrong — trace back to which
# inspection revealed the issue and which script handled it.
# ============================================================


# ══════════════════════════════════════════════════════════════
# PHASE 0 — INITIAL DATASET DOWNLOADS
# ══════════════════════════════════════════════════════════════

# Folders downloaded and renamed:
#   4060432          → fsd50k
#   8056090          → anuraset
#   archive          → voice_of_birds       (SKIPPED — wrong species)
#   archive (1)      → toads_frogs_anuran   (SKIPPED — no metadata, M4A format)
#   archive (2)      → birdclef_2021
#   ESC-50-master    → ESC-50-master        (no rename needed)
#   freefield1010    → freefield1010
#   FSC22_forest     → FSC22_forest         (NOT USED — targets met)
#   forest_wild_fire_sound_dataset → same
#   urbansound8k     → urbansound8k
#   99Sounds Nature Sounds → same           (NOT USED — targets met)
#   rain_dataset     → same                 (SKIPPED — no audio files)
#   audio_noise_dataset → same              (SKIPPED — 10 WebM files)


# ══════════════════════════════════════════════════════════════
# PHASE 1 — ESC-50
# ══════════════════════════════════════════════════════════════

# --- Claude Code Inspection ---
# Prompt: full folder structure, all 50 category names + counts,
#         sample filenames, total WAV count
# Key findings:
#   - 2000 WAV files, 44100Hz mono PCM_16, 5s fixed
#   - 50 categories × 40 clips each — perfectly balanced
#   - Flat audio/ folder, labels in meta/esc50.csv → category column
#   - 10 of 50 categories match our 12 subclasses
#   - engine category skipped (generic, not engine_idling)
#   - owl and jackhammer not present in ESC-50
# Categories kept (10):
#   crow, frog, insects, crickets → insects (merged)
#   rain, sea_waves, wind, crackling_fire
#   car_horn, siren

# --- Scripts Run ---

# STEP 1: Filter — remove 40 unwanted categories
# Input:  ESC-50-master/audio/ (2000 WAV)
# Output: ESC-50-master/audio/ (400 WAV kept), meta/esc50.csv updated
# Script: filter_esc50.py
#   python filter_esc50.py --base ./ESC-50-master --dry-run
#   python filter_esc50.py --base ./ESC-50-master
# Result: 1600 files deleted, 400 kept, CSV updated with subclass+main_class cols
# Backups created: esc50_original_backup.csv

# STEP 2: Rename — numeric filenames → subclass_NNN.wav
# Input:  ESC-50-master/audio/ (400 WAV, numeric names)
# Output: ESC-50-master/audio/ (400 WAV, renamed)
#         main_metadata.csv CREATED (400 rows)
#         filename_registry.csv CREATED (12 rows)
# Script: rename_esc50.py
#   python rename_esc50.py --base ./ESC-50-master \
#          --global-meta ./main_metadata.csv --dry-run
#   python rename_esc50.py --base ./ESC-50-master \
#          --global-meta ./main_metadata.csv
# Result: 400 files renamed (e.g. 1-103298-A-9.wav → crow_001.wav)
#         main_metadata.csv: 400 rows
# Backups created: esc50_before_rename.csv

# STEP 3: Augment — double ESC-50 clips 40 → 80 per subclass
# Input:  ESC-50-master/audio/ (400 WAV originals)
# Output: ESC-50-master/audio/ (720 WAV = 400 orig + 320 aug)
#         main_metadata.csv UPDATED (720 rows)
# Script: augment_esc50.py
#   python augment_esc50.py --base ./ESC-50-master \
#          --global-meta ./main_metadata.csv --dry-run
#   python augment_esc50.py --base ./ESC-50-master \
#          --global-meta ./main_metadata.csv
# Result: 320 augmented files created (crow_001_aug.wav etc.)
#         insects skipped — already 80 (crickets+insects merged)
#         main_metadata.csv: 720 rows
# Backups created: esc50_before_aug.csv
# Augmentation techniques:
#   crow          → pitch_shift  -2 semitones
#   frog          → add_noise    SNR=30dB
#   insects       → time_stretch rate=0.9
#   rain          → time_stretch rate=1.1
#   sea_waves     → pitch_shift  +2 semitones
#   wind          → add_noise    SNR=30dB
#   crackling_fire→ time_stretch rate=0.9
#   car_horn      → pitch_shift  +2 semitones
#   siren         → time_stretch rate=1.1


# ══════════════════════════════════════════════════════════════
# PHASE 2 — FSD50K
# ══════════════════════════════════════════════════════════════

# --- Claude Code Inspection ---
# Prompt: dev.csv structure, label match counts for all target labels,
#         sample rain file with labels, unique fire/insect label tokens
# Key findings:
#   - 40,966 WAV, 44100Hz mono, variable 0.3-30s
#   - Multi-label CSV (comma-separated AudioSet ontology)
#   - Car_horn label = Vehicle_horn_and_car_horn_and_honking (not Car_horn)
#   - Owl = 0 clips, Jackhammer = 0 clips — not in FSD50K
#   - crackling_fire: strict AND filter (Fire+Crackle) → only 26 clips
#   - engine_idling: strict AND filter (Engine+Idling) → 107 clips
#   - Rain: 500, Wind: 294, Frog: 72, Insect: 383, Crow: 75
# Additional check: sea_waves → Ocean(239) + Waves_and_surf(167) confirmed

# --- Scripts Run ---

# STEP 4: Filter — copy matching files to filtered_audio/
# Input:  fsd50k/FSD50K.dev_audio/ (40,966 WAV)
# Output: fsd50k/filtered_audio/ (1,874 variable-length WAV)
#         fsd50k/filtered_metadata.csv (1,874 rows)
#         main_metadata.csv UPDATED (2,594 rows)
#         filename_registry.csv UPDATED
# Script: filter_fsd50k.py
#   python filter_fsd50k.py --base ./fsd50k \
#          --global-meta ./main_metadata.csv \
#          --registry ./filename_registry.csv --dry-run
#   python filter_fsd50k.py --base ./fsd50k \
#          --global-meta ./main_metadata.csv \
#          --registry ./filename_registry.csv
# Result: 1,874 files staged across 10 subclasses
#         Registry auto-built from existing main_metadata.csv (first run)

# --- Second Claude Code Inspection (FSD50K filtered_audio) ---
# Prompt: duration stats on 1,874 files in filtered_audio/
# Key findings:
#   - 100% 44100Hz, 0 errors
#   - Duration: min=0.31s, max=30s, median=12.11s, mean=12.98s
#   - 81 files under 1s (discard), 353 files 1-5s (pad), 1074 files 10-30s (slice)

# STEP 5: Slice — variable length → 5s clips
# Input:  fsd50k/filtered_audio/ (1,874 WAV, variable length)
# Output: fsd50k/sliced_audio/ (~2,950 WAV, all 5s)
#         fsd50k/sliced_metadata.csv
#         main_metadata.csv UPDATED (replaces filtered entries with sliced)
#         filename_registry.csv UPDATED
# Script: slice_fsd50k.py
#   python slice_fsd50k.py --base ./fsd50k \
#          --global-meta ./main_metadata.csv \
#          --registry ./filename_registry.csv --dry-run
#   python slice_fsd50k.py --base ./fsd50k \
#          --global-meta ./main_metadata.csv \
#          --registry ./filename_registry.csv
# Result: ~2,950 sliced clips
#         81 files under 1s discarded
#         Slicing rule: non-overlapping 5s windows, pad last if >=1s


# ══════════════════════════════════════════════════════════════
# PHASE 3 — BIRDCLEF 2021
# ══════════════════════════════════════════════════════════════

# --- Claude Code Inspection ---
# Prompt: folder structure, train_extended.csv, ebird_code counts for
#         crow/raven/owl species, audio quality check on 6 files
# Key findings:
#   - 14,685 MP3, mixed 22050-48000Hz, stereo/mono mixed
#   - amecro (147) + fiscro (8) + comrav (889) = 1044 crow files
#   - grhowl (194) + brdowl (28) = 222 owl files
#   - sheowl = 0 files on disk (N-Z not downloaded)
#   - Mixed bitrates 71-320kbps, some corrupt headers (Xing warnings)
#   - Average duration: crow ~47s, owl ~65s → many 5s slices per file
#   - Rooster swapped to Owl (rooster only ~435 clips, owl has 3,846+)

# --- Scripts Run ---

# STEP 6: Filter + Convert + Slice (all in one pass)
# Input:  birdclef_2021/A-M/amecro/, fiscro/, comrav/, grhowl/, brdowl/
# Output: birdclef_2021/sliced_audio/ (16,242 WAV, all 5s, 22050Hz mono)
#         birdclef_2021/birdclef_metadata.csv (16,242 rows)
#         main_metadata.csv UPDATED (22,486 rows)
#         filename_registry.csv UPDATED
# Script: filter_birdclef.py
#   python filter_birdclef.py --base ./birdclef_2021 \
#          --global-meta ./main_metadata.csv \
#          --registry ./filename_registry.csv --dry-run
#   python filter_birdclef.py --base ./birdclef_2021 \
#          --global-meta ./main_metadata.csv \
#          --registry ./filename_registry.csv
# Result:
#   crow: 12,585 slices (1044 MP3 files)
#   owl:   3,657 slices (222 MP3 files)
#   1 file discarded (too short), 0 errors
#   Xing stream warnings and corrupt headers — librosa recovered automatically
# Conversion: librosa.load(sr=22050, mono=True) — handles all SR + formats


# ══════════════════════════════════════════════════════════════
# PHASE 4 — ANURASET
# ══════════════════════════════════════════════════════════════

# --- Claude Code Inspection ---
# Prompt: folder structure, weak_labels.csv, audio/ subfolder counts,
#         audio quality check on 5 files
# Key findings:
#   - 93,378 WAV, 22050Hz mono PCM_16, exactly 3.0s
#   - 4 monitoring sites: INCT4(24360), INCT17(20532), INCT41(21228), INCT20955(27258)
#   - Filename encodes site+date+time+start_sec+end_sec
#   - weak_labels.csv: recording-level labels (42 frog species)
#   - All clips = frog — no filtering needed
#   - 3s clips need 5s — zero-padding REJECTED (40% silence = spurious correlation)
#   - Solution: consecutive clip pairing A+B = 6s → trim to 5s

# --- Scripts Run ---

# STEP 7: Pair + Pad (consecutive clip concatenation)
# Input:  anuraset/anuraset/audio/ (93,378 WAV, 3s each)
# Output: anuraset/sliced_audio/ (700 WAV, 5s, consecutive pairs)
#         anuraset/anuraset_metadata.csv (700 rows)
#         main_metadata.csv UPDATED (23,186 rows)
#         filename_registry.csv UPDATED
# Script: filter_anuraset.py
#   python filter_anuraset.py --base ./anuraset \
#          --global-meta ./main_metadata.csv \
#          --registry ./filename_registry.csv --dry-run
#   python filter_anuraset.py --base ./anuraset \
#          --global-meta ./main_metadata.csv \
#          --registry ./filename_registry.csv
# Result:
#   88,432 valid consecutive pairs found across 4 sites
#   700 pairs sampled (stratified: INCT4:183, INCT17:154, INCT20955:204, INCT41:159)
#   700 clips written (5s = 3s audio + 2s from next clip, trimmed)
#   0 errors
# Pairing logic: INCT4_20191005_173000_0_3.wav + INCT4_20191005_173000_3_6.wav
#                same recording key, consecutive seconds → concat → trim to 5s


# ══════════════════════════════════════════════════════════════
# PHASE 5 — URBANSOUND8K
# ══════════════════════════════════════════════════════════════

# --- Claude Code Inspection ---
# Prompt: folder structure, UrbanSound8K.csv all columns,
#         class counts, audio quality check 3 files per class
# Key findings:
#   - 8,732 WAV across 10 folds, max 4s clips
#   - car_horn: 429, engine_idling: 1000, siren: 929, jackhammer: 1000
#   - Mixed SR: 11025-96000Hz, mixed bit depth PCM_16/PCM_24/FLOAT
#   - car_horn avg duration only 2.46s — many clips under 1s
#   - Zero-padding 1s REJECTED — same spurious correlation concern as AnuraSet
#   - Solution: bidirectional A+B and B+A pairing (extends AnuraSet approach)
#   - Combined duration must be >=5s, same class only

# --- Scripts Run ---

# STEP 8: Pair bidirectional (A+B and B+A)
# Input:  urbansound8k/fold1/ through fold10/ (8,732 WAV)
# Output: urbansound8k/sliced_audio/ (3,788 WAV, 5s, paired)
#         urbansound8k/urbansound_metadata.csv (3,788 rows)
#         main_metadata.csv UPDATED (26,974 rows)
#         filename_registry.csv UPDATED
# Script: filter_urbansound.py
#   python filter_urbansound.py --base ./urbansound8k \
#          --global-meta ./main_metadata.csv \
#          --registry ./filename_registry.csv --dry-run
#   python filter_urbansound.py --base ./urbansound8k \
#          --global-meta ./main_metadata.csv \
#          --registry ./filename_registry.csv
# Result:
#   car_horn:      94 pairs →  188 clips (120 pairs discarded, combined <5s)
#   engine_idling: 600 pairs → 1200 clips
#   siren:         600 pairs → 1200 clips
#   jackhammer:    600 pairs → 1200 clips
#   Total: 3,788 clips, 0 errors
# NOTE: car_horn only 481 total (ESC+FSD50K+Urban) — 19 short of 500
#   Fix option: three-way pairing A+B+C for short clips (deferred to post-EDA)
# Conversion: librosa.load(sr=22050, mono=True) normalises all SR variants


# ══════════════════════════════════════════════════════════════
# PHASE 6 — FREEFIELD1010
# ══════════════════════════════════════════════════════════════

# --- Claude Code Inspection ---
# Prompt: folder structure, JSON sidecar format, tag counts for
#         rain/wind/ocean/fire/bird tags, audio quality
# Key findings:
#   - 7,690 WAV, 44100Hz mono PCM_16, exactly 10.0s — cleanest dataset
#   - Metadata: per-file JSON sidecar with free-text tags array
#   - 7,240 unique tags — noisy, needs include/exclude filter
#   - rain: 426 files, wind: 397, sea(ocean+waves): 393, fire: 56 (sparse)
#   - Each 10s file → exactly 2 × 5s slices, no padding needed
#   - Exclusive assignment to prevent duplicates: rain > sea_waves > wind > fire
#   - Also has bonus: crow(42), owl(29), frog(92), insects(212) — not processed

# --- Scripts Run ---

# STEP 9: Filter by JSON tags + slice 10s → 2×5s
# Input:  freefield1010/01/ through 10/ (7,690 WAV + JSON sidecars)
# Output: freefield1010/sliced_audio/ (1,314 WAV, all 5s)
#         freefield1010/freefield_metadata.csv (1,314 rows)
#         main_metadata.csv UPDATED (28,288 rows)
#         filename_registry.csv UPDATED
# Script: filter_freefield.py
#   python filter_freefield.py --base ./freefield1010 \
#          --global-meta ./main_metadata.csv \
#          --registry ./filename_registry.csv --dry-run
#   python filter_freefield.py --base ./freefield1010 \
#          --global-meta ./main_metadata.csv \
#          --registry ./filename_registry.csv
# Result:
#   rain:           200 files → 400 clips
#   sea_waves:      200 files → 400 clips
#   wind:           200 files → 400 clips
#   crackling_fire:  57 files → 114 clips (all available — only 57 matched)
#   Total: 1,314 clips, 0 errors
# After this step:
#   rain(976), sea_waves(718), wind(771) → ALL ABOVE 500 ✓
#   crackling_fire(220) → still needs more


# ══════════════════════════════════════════════════════════════
# PHASE 7 — FOREST WILD FIRE SOUND DATASET
# ══════════════════════════════════════════════════════════════

# --- Quick Claude Code Check ---
# Command run:
#   from pathlib import Path; import soundfile as sf
#   wavs = list(Path("./forest_wild_fire_sound_dataset").rglob("*.wav"))
#   print(len(wavs))  # 289
#   info = sf.info(str(wavs[0]))  # 44100Hz, ~50s
# Key findings:
#   - 289 WAV files, 44100Hz, ~50s each
#   - Two subfolders with Google Drive artifact names
#   - No metadata — entire dataset is crackling fire
#   - Each 50s file → 10 × 5s slices = ~2,890 clips available
#   - Only need 280 more clips — process first 30 files only (early stop)

# --- Scripts Run ---

# STEP 10: Slice with early stopping
# Input:  forest_wild_fire_sound_dataset/ (289 WAV, ~50s each)
# Output: forest_wild_fire_sound_dataset/sliced_audio/ (300 WAV, 5s)
#         forest_wild_fire_sound_dataset/wildfire_metadata.csv (300 rows)
#         main_metadata.csv UPDATED (28,588 rows)
#         filename_registry.csv UPDATED
# Script: filter_wildfire.py
#   python filter_wildfire.py \
#          --base ./forest_wild_fire_sound_dataset \
#          --global-meta ./main_metadata.csv \
#          --registry ./filename_registry.csv --dry-run
#   python filter_wildfire.py \
#          --base ./forest_wild_fire_sound_dataset \
#          --global-meta ./main_metadata.csv \
#          --registry ./filename_registry.csv
# Result:
#   300 clips from 30 files (early stop — target reached)
#   259 files remain as untouched backup
#   crackling_fire total: 220 + 300 = 520 ✓
#   0 errors
# Efficient reading: sf.read(start=sample_start, frames=220500)
#   reads only the 5s window from 50s file — no full file load needed


# ══════════════════════════════════════════════════════════════
# DATASETS NOT PROCESSED (targets already met)
# ══════════════════════════════════════════════════════════════

# FSC22_forest
#   Would feed: rain, wind, crackling_fire
#   Reason skipped: crackling_fire hit 520 from forest_wildfire alone
#                   rain(2109) and wind(1138) already above 500
#   Status: downloaded, not processed

# 99Sounds Nature Sounds
#   Would feed: sea_waves, wind
#   Reason skipped: sea_waves(1273) and wind(1138) already above 500
#                   would need heavy downsampling 192kHz → 22050Hz
#   Status: downloaded, not processed

# InsectSet459
#   Would feed: insects
#   Reason skipped: insects(1242) from FSD50K alone — well above 500
#   Status: not downloaded (was planned, skipped early)

# voice_of_birds (archive)
#   Reason skipped: wrong species (Tinamou, Guan, Megapode) — no crow/owl
#   Status: downloaded, inspected, rejected

# toads_frogs_anuran / archive (1)
#   Reason skipped: 1,536 M4A files, no metadata CSV
#                   AnuraSet is far superior (93K pre-segmented WAV)
#   Status: downloaded, inspected, rejected

# rain_dataset
#   Reason skipped: weather statistics CSV only — zero audio files
#   Status: downloaded, inspected immediately rejected

# audio_noise_dataset
#   Reason skipped: 10 WebM browser files, no labels
#   Status: downloaded, inspected immediately rejected


# ══════════════════════════════════════════════════════════════
# FINAL STATE
# ══════════════════════════════════════════════════════════════

# Global files created/updated:
#   main_metadata.csv         28,588 rows
#   filename_registry.csv     12 rows (one per subclass)

# Final clip counts from main_metadata.csv:
#   crow           12,864  ✓ (trim to 500)
#   owl             3,657  ✓ (trim to 500)
#   frog            1,050  ✓ (trim to 500)
#   insects         1,242  ✓ (trim to 500)
#   rain            2,109  ✓ (trim to 500)
#   sea_waves       1,273  ✓ (trim to 500)
#   wind            1,138  ✓ (trim to 500)
#   crackling_fire    551  ✓ (trim to 500)
#   car_horn          481  ⚠ (19 short — trim all to 481 or fix post-EDA)
#   engine_idling   1,499  ✓ (trim to 500)
#   siren           1,524  ✓ (trim to 500)
#   jackhammer      1,200  ✓ (trim to 500)
#   TOTAL          28,588

# Scripts created (in order of execution):
#   filter_esc50.py           ESC-50 category filtering
#   rename_esc50.py           ESC-50 file renaming + main_metadata.csv creation
#   augment_esc50.py          ESC-50 augmentation 40→80 per subclass
#   filter_fsd50k.py          FSD50K multi-label filtering + registry
#   slice_fsd50k.py           FSD50K variable-length → 5s slicing
#   filter_birdclef.py        BirdCLEF MP3→WAV + slice (crow + owl)
#   filter_anuraset.py        AnuraSet consecutive pair concat (frog)
#   filter_urbansound.py      UrbanSound8K bidirectional A+B/B+A (urban)
#   filter_freefield.py       freefield1010 JSON tag filter + slice (nature)
#   filter_wildfire.py        forest wildfire early-stop slicing (fire)

# Inspection scripts (Claude Code prompts run before each dataset):
#   audit_audio_datasets.py   initial wildlife dataset audit (all 7 folders)
#   inspect_nature_urban.py   nature + urban dataset audit
#   audio_audit.py            FSD50K filtered_audio duration stats

# Next step: EDA
#   See: EDA checklist in document Section 3.8



# Example
#   Problem: frog clips sound wrong
#   → Check anuraset/anuraset_metadata.csv (source_clip_a, source_clip_b columns)
#   → Trace back to filter_anuraset.py STEP 7
#   → Check consecutive pairing logic in audit trail
#   → Verify 88,432 pairs found, 700 sampled stratified

#   Problem: car_horn model accuracy low
#   → Audit trail shows only 481 clips (19 short)
#   → car_horn clips avg 2.46s → many pairs discarded
#   → Fix: three-way pairing A+B+C for short clips (deferred to post-EDA)
