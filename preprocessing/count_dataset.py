import csv
from collections import defaultdict
from pathlib import Path

base = Path(".")
metadata_csv = base / "main_metadata.csv"

# Read metadata
all_rows = []
with open(metadata_csv, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    print(f"Columns: {fieldnames}\n")
    for row in reader:
        all_rows.append(row)

print(f"Total rows in main_metadata.csv: {len(all_rows)}\n")

# Count 1 — per subclass
sub_counts = defaultdict(int)
for r in all_rows:
    sub_counts[r["subclass"]] += 1

subclass_order = [
    "crow", "owl", "frog", "insects",
    "rain", "sea_waves", "wind", "crackling_fire",
    "car_horn", "engine_idling", "siren", "jackhammer"
]

print("=" * 65)
print(f"  {'Subclass':<20} {'Main Class':<12} {'Clips':>8}")
print("=" * 65)
main_class = {
    "crow":"wildlife","owl":"wildlife","frog":"wildlife","insects":"wildlife",
    "rain":"nature","sea_waves":"nature","wind":"nature","crackling_fire":"nature",
    "car_horn":"urban","engine_idling":"urban","siren":"urban","jackhammer":"urban"
}
for sub in subclass_order:
    print(f"  {sub:<20} {main_class[sub]:<12} {sub_counts[sub]:>8}")
print("=" * 65)
print(f"  {'TOTAL':<32} {sum(sub_counts.values()):>8}\n")

# Count 2 — per source within each subclass
print("=" * 65)
print("SOURCE BREAKDOWN PER SUBCLASS")
print("=" * 65)
sub_source = defaultdict(lambda: defaultdict(int))
for r in all_rows:
    sub_source[r["subclass"]][r["source"]] += 1

for sub in subclass_order:
    print(f"\n  {sub} ({sub_counts[sub]} total):")
    for src, cnt in sorted(sub_source[sub].items(),
                           key=lambda x: x[1], reverse=True):
        pct = cnt / sub_counts[sub] * 100
        bar = "█" * int(pct / 5)
        print(f"    {src:<25} {cnt:>5}  ({pct:5.1f}%)  {bar}")

# Count 3 — per main class
print("\n" + "=" * 65)
print("MAIN CLASS TOTALS")
print("=" * 65)
mc_counts = defaultdict(int)
for r in all_rows:
    mc_counts[r["main_class"]] += 1
for mc in ["wildlife", "nature", "urban"]:
    print(f"  {mc:<12} {mc_counts[mc]:>8}")
print(f"  {'TOTAL':<12} {sum(mc_counts.values()):>8}")

# Count 4 — actual WAV files on disk per folder
print("\n" + "=" * 65)
print("WAV FILES ON DISK PER FOLDER")
print("=" * 65)
audio_folders = [
    ("ESC-50-master/audio",                         "ESC-50"),
    ("fsd50k/sliced_audio",                          "FSD50K"),
    ("birdclef_2021/sliced_audio",                   "BirdCLEF"),
    ("anuraset/sliced_audio",                        "AnuraSet"),
    ("urbansound8k/sliced_audio",                    "UrbanSound8K"),
    ("freefield1010/sliced_audio",                   "freefield1010"),
    ("forest_wild_fire_sound_dataset/sliced_audio",  "forest_wildfire"),
]
total_disk = 0
for folder_rel, label in audio_folders:
    folder = base / folder_rel
    if folder.exists():
        count = len(list(folder.glob("*.wav")))
        total_disk += count
        print(f"  {label:<25} {count:>8} WAV files   ({folder_rel})")
    else:
        print(f"  {label:<25} {'NOT FOUND':>8}             ({folder_rel})")
print(f"  {'TOTAL ON DISK':<25} {total_disk:>8}")

# Count 5 — metadata vs disk mismatch
print("\n" + "=" * 65)
print("METADATA vs DISK CHECK")
print("=" * 65)
print(f"  Rows in main_metadata.csv:  {len(all_rows)}")
print(f"  WAV files on disk (total):  {total_disk}")
diff = len(all_rows) - total_disk
if diff == 0:
    print("  Match: PERFECT")
elif diff > 0:
    print(f"  Metadata has {diff} more rows than disk files — {diff} files missing from disk")
else:
    print(f"  Disk has {abs(diff)} more files than metadata rows")

# Count 6 — augmented vs original
print("\n" + "=" * 65)
print("AUGMENTED vs ORIGINAL")
print("=" * 65)
aug_count  = sum(1 for r in all_rows if r.get("augmented") == "yes")
orig_count = sum(1 for r in all_rows if r.get("augmented") == "no")
print(f"  Original clips:   {orig_count}")
print(f"  Augmented clips:  {aug_count}")
print(f"  Total:            {orig_count + aug_count}")
