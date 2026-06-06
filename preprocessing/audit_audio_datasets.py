from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import librosa
from scipy.io import wavfile


ROOT = Path.cwd()
DATASETS = [
    "rain_dataset",
    "audio_noise_dataset",
    "FSC22_forest",
    "forest_wild_fire_sound_dataset",
    "freefield1010",
    "urbansound8k",
    "99Sounds Nature Sounds",
    "4060432",
    "ESC-50-master",
]

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}
NONSTANDARD_AUDIO_EXTS = {".webm"}
META_EXTS = {".csv", ".json", ".txt", ".md", ".README"}
LABEL_COL_RE = re.compile(r"(label|category|class|tag|description|target|name|caption)", re.I)

NATURE_KEYS = {
    "rain": [r"\brain\b", r"\braining\b", r"\brainfall\b"],
    "sea_waves": [r"sea[_ -]?waves?", r"ocean", r"waves?", r"water"],
    "wind": [r"\bwind\b", r"windy"],
    "crackling_fire": [r"crackl", r"\bfire\b", r"wild[_ -]?fire", r"forest[_ -]?fire"],
}
URBAN_KEYS = {
    "car_horn": [r"car[_ -]?horn"],
    "engine_idling": [r"engine[_ -]?idling", r"\bengine\b"],
    "siren": [r"\bsiren\b"],
    "jackhammer": [r"\bjackhammer\b"],
}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_first_lines(path: Path, n: int = 10) -> list[str]:
    try:
        return path.read_text(errors="replace").splitlines()[:n]
    except Exception as exc:
        return [f"<could not read text: {exc}>"]


def tree_dirs(path: Path, max_depth: int = 3) -> list[str]:
    rows = []
    base_depth = len(path.parts)
    for p in sorted(path.rglob("*")):
        depth = len(p.parts) - base_depth
        if depth > max_depth:
            continue
        if p.is_dir():
            rows.append("  " * (depth - 1) + p.name + "/")
    return rows[:500]


def root_non_audio(path: Path) -> list[Path]:
    out = []
    for p in sorted(path.iterdir()):
        if p.is_file() and p.suffix.lower() not in AUDIO_EXTS:
            out.append(p)
    return out


def metadata_files(path: Path) -> list[Path]:
    if path.name == "freefield1010":
        sample_json = sorted(path.rglob("*.json"))[:3]
        return sorted([p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".csv"}]) + sample_json
    out = []
    for p in path.rglob("*"):
        if p.is_file() and (p.suffix.lower() in META_EXTS or p.name.lower().startswith("readme")):
            out.append(p)
    return sorted(out)


def audio_files(path: Path, include_nonstandard: bool = False) -> list[Path]:
    exts = AUDIO_EXTS | (NONSTANDARD_AUDIO_EXTS if include_nonstandard else set())
    return sorted([p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def wav_info(path: Path) -> dict:
    try:
        sr, data = wavfile.read(path, mmap=True)
        frames = data.shape[0]
        channels = 1 if data.ndim == 1 else data.shape[1]
        duration = frames / float(sr) if sr else 0
        clipped = False
        silent = False
        try:
            arr = data[: min(frames, sr * 10 if sr else frames)]
            if arr.size:
                silent = float(abs(arr).mean()) == 0.0
                if hasattr(arr, "dtype") and arr.dtype.kind in "iu":
                    info = None
                    if arr.dtype.kind == "i":
                        info = __import__("numpy").iinfo(arr.dtype)
                    elif arr.dtype.kind == "u":
                        info = __import__("numpy").iinfo(arr.dtype)
                    if info:
                        clipped = bool((arr <= info.min).any() or (arr >= info.max).any())
        except Exception:
            pass
        return {
            "ok": True,
            "sample_rate": int(sr),
            "channels": int(channels),
            "duration": duration,
            "format": path.suffix.lower().lstrip("."),
            "silent": silent,
            "clipped": clipped,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "format": path.suffix.lower().lstrip(".")}


def audio_info(path: Path) -> dict:
    if path.suffix.lower() == ".wav":
        return wav_info(path)
    return {"ok": False, "error": "non-WAV; no decoder available in current env", "format": path.suffix.lower().lstrip(".")}


def librosa_audio_info(path: Path) -> dict:
    try:
        y, sr = librosa.load(path, sr=None, mono=False)
        channels = 1 if y.ndim == 1 else int(y.shape[0])
        duration = float(librosa.get_duration(y=y, sr=sr))
        silent = bool(abs(y).mean() == 0) if y.size else True
        clipped = bool((abs(y) >= 0.999).any()) if y.size else False
        return {
            "ok": True,
            "sample_rate": int(sr),
            "channels": channels,
            "duration": duration,
            "format": path.suffix.lower().lstrip("."),
            "silent": silent,
            "clipped": clipped,
            "loader": "librosa.load(sr=None, mono=False)",
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "format": path.suffix.lower().lstrip("."),
            "loader": "librosa.load(sr=None, mono=False)",
        }


def metadata_summary(path: Path) -> dict:
    item = {"path": rel(path), "columns": [], "rows": None, "head": [], "label_values": {}}
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            item["columns"] = list(df.columns)
            item["rows"] = int(len(df))
            item["head"] = df.head(10).fillna("").astype(str).to_dict(orient="records")
            for col in df.columns:
                if LABEL_COL_RE.search(str(col)):
                    vals = sorted(map(str, df[col].dropna().unique()))
                    item["label_values"][str(col)] = vals[:300]
        elif path.suffix.lower() == ".json":
            with path.open(errors="replace") as f:
                data = json.load(f)
            if isinstance(data, dict):
                item["columns"] = list(data.keys())[:100]
                item["rows"] = len(data)
                sample = list(data.items())[:10]
                item["head"] = [{str(k): str(v)[:500]} for k, v in sample]
            elif isinstance(data, list):
                item["rows"] = len(data)
                item["head"] = data[:10]
                if data and isinstance(data[0], dict):
                    item["columns"] = list(data[0].keys())
        else:
            lines = read_first_lines(path, 10)
            item["head"] = lines
            item["rows"] = sum(1 for _ in path.open(errors="replace"))
    except Exception as exc:
        item["error"] = str(exc)
    return item


def regex_any(text: str, pats: list[str]) -> bool:
    return any(re.search(p, text, flags=re.I) for p in pats)


def duration_sum(files: list[Path]) -> tuple[int, float, int, Counter]:
    total_5s = 0
    total_dur = 0.0
    usable = 0
    srs = Counter()
    for p in files:
        info = audio_info(p)
        if info.get("ok"):
            usable += 1
            dur = float(info.get("duration") or 0)
            total_dur += dur
            total_5s += int(math.floor(dur / 5.0))
            srs[str(info.get("sample_rate"))] += 1
    return total_5s, total_dur, usable, srs


def match_by_path(files: list[Path], pats: list[str]) -> list[Path]:
    return [p for p in files if regex_any(rel(p), pats)]


def esc50_counts(ds_path: Path) -> dict:
    csv_path = ds_path / "meta" / "esc50.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    out = {}
    mapping = {
        "rain": ["rain"],
        "sea_waves": ["sea_waves"],
        "wind": ["wind"],
        "crackling_fire": ["crackling_fire"],
    }
    for sub, cats in mapping.items():
        m = df["category"].isin(cats)
        files = [ds_path / "audio" / f for f in df.loc[m, "filename"].astype(str)]
        five, dur, usable, srs = duration_sum(files)
        out[sub] = {"mechanism": f"meta/esc50.csv category in {cats}", "clips": int(m.sum()), "five_sec": five, "sample_rates": dict(srs)}
    return out


def urbansound_counts(ds_path: Path) -> dict:
    csv_path = ds_path / "UrbanSound8K.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    out = {}
    labels = {
        "car_horn": "car_horn",
        "engine_idling": "engine_idling",
        "siren": "siren",
        "jackhammer": "jackhammer",
    }
    for sub, label in labels.items():
        m = df["class"].astype(str).str.lower().eq(label)
        files = [ds_path / f"fold{fold}" / fn for fold, fn in zip(df.loc[m, "fold"], df.loc[m, "slice_file_name"])]
        five, dur, usable, srs = duration_sum(files)
        out[sub] = {"mechanism": f"UrbanSound8K.csv class == {label}", "clips": int(m.sum()), "five_sec": five, "sample_rates": dict(srs)}
    return out


def fsd_counts(ds_path: Path) -> dict:
    gt = ds_path / "FSD50K.ground_truth" / "dev.csv"
    if not gt.exists():
        return {}
    df = pd.read_csv(gt)
    label_col = "labels"
    out = {"nature_raw": {}, "urban_raw": {}, "mapped": {}}
    raw_nature = ["Rain", "Water", "Wind", "Fire", "Crackling"]
    raw_urban = ["Car_horn", "Engine", "Siren", "Jackhammer"]
    for key in raw_nature:
        out["nature_raw"][key] = int(df[label_col].astype(str).str.contains(key, case=False, regex=False).sum())
    for key in raw_urban:
        out["urban_raw"][key] = int(df[label_col].astype(str).str.contains(key, case=False, regex=False).sum())
    audio_dir = ds_path / "FSD50K.dev_audio"
    for sub, pats in (NATURE_KEYS | URBAN_KEYS).items():
        text_pat = "|".join(pats)
        m = df[label_col].astype(str).str.contains(text_pat, case=False, regex=True)
        files = [audio_dir / f"{fid}.wav" for fid in df.loc[m, "fname"]]
        five, dur, usable, srs = duration_sum(files)
        out["mapped"][sub] = {
            "mechanism": f"FSD50K.ground_truth/dev.csv labels regex {pats}",
            "clips": int(m.sum()),
            "five_sec": five,
            "audio_found": usable,
            "sample_rates": dict(srs),
        }
    return out


def fsc22_counts(ds_path: Path) -> dict:
    metas = list(ds_path.rglob("*.csv"))
    if not metas:
        return {}
    csv_path = metas[0]
    df = pd.read_csv(csv_path)
    label_cols = [c for c in df.columns if LABEL_COL_RE.search(str(c))]
    file_cols = [c for c in df.columns if re.search(r"(file|audio|name)", str(c), re.I)]
    out = {}
    for sub, pats in NATURE_KEYS.items():
        mask = pd.Series(False, index=df.index)
        for col in label_cols:
            mask |= df[col].astype(str).str.contains("|".join(pats), case=False, regex=True, na=False)
        files = []
        for _, row in df.loc[mask].iterrows():
            for fc in file_cols:
                val = str(row.get(fc, ""))
                if val and val.lower() != "nan":
                    cand = ds_path / val
                    if not cand.exists():
                        name = Path(val).name
                        matches = list(ds_path.rglob(name))
                        if matches:
                            cand = matches[0]
                    if cand.exists() and cand.suffix.lower() in AUDIO_EXTS:
                        files.append(cand)
                        break
        if not files:
            # FSC22 audio filenames start with numeric class ids; use metadata Class ID where present.
            id_cols = [c for c in df.columns if re.search(r"(class.*id|category.*id|label.*id|^id$)", str(c), re.I)]
            matched_ids = set()
            for col in id_cols:
                for val in df.loc[mask, col].dropna().astype(str):
                    matched_ids.add(val.split(".")[0])
            all_audio = audio_files(ds_path)
            files = [p for p in all_audio if p.name.split("_")[0] in matched_ids]
        five, dur, usable, srs = duration_sum(files)
        out[sub] = {
            "mechanism": f"{rel(csv_path)} label columns {label_cols}",
            "clips": int(mask.sum()) if len(files) == 0 else len(files),
            "five_sec": five,
            "sample_rates": dict(srs),
        }
    return out


def generic_nature_counts(ds_path: Path) -> dict:
    files = audio_files(ds_path)
    out = {}
    for sub, pats in NATURE_KEYS.items():
        matched = match_by_path(files, pats)
        five, dur, usable, srs = duration_sum(matched)
        out[sub] = {
            "mechanism": "folder/path regex " + str(pats) if matched else "NOT PRESENT",
            "clips": len(matched),
            "five_sec": five,
            "sample_rates": dict(srs),
        }
    return out


def freefield_metadata_index(ds_path: Path) -> tuple[dict[str, dict], Counter]:
    index = {}
    tags = Counter()
    for p in ds_path.rglob("*.json"):
        try:
            data = json.load(p.open(errors="replace"))
        except Exception:
            continue
        sid = str(data.get("id") or p.stem)
        index[sid] = data
        for tag in data.get("tags", []) or []:
            tags[str(tag).lower()] += 1
    return index, tags


def freefield_counts(ds_path: Path) -> dict:
    files = audio_files(ds_path)
    meta_index, _ = freefield_metadata_index(ds_path)
    tag_pats = {
        "rain": [r"\brain\b", r"\braining\b", r"\brainfall\b"],
        "sea_waves": [r"\bsea\b", r"\bocean\b", r"\bwave\b", r"\bwaves\b", r"\bbeach\b"],
        "wind": [r"\bwind\b", r"windy"],
        "crackling_fire": [r"crackl", r"\bfire\b", r"\bflame\b", r"\bburning\b"],
    }
    out = {}
    for sub, pats in tag_pats.items():
        matched = []
        for p in files:
            data = meta_index.get(p.stem, {})
            haystack = " ".join([
                str(data.get("original_filename", "")),
                " ".join(map(str, data.get("tags", []) or [])),
                rel(p),
            ])
            if regex_any(haystack, pats):
                matched.append(p)
        five, dur, usable, srs = duration_sum(matched)
        out[sub] = {
            "mechanism": f"Freefield1010 per-clip JSON tags/original_filename regex {pats}" if matched else "NOT PRESENT",
            "clips": len(matched),
            "five_sec": five,
            "sample_rates": dict(srs),
        }
    return out


def nature99_counts(ds_path: Path) -> dict:
    files = audio_files(ds_path)
    selectors = {
        "rain": [p for p in files if "99Sounds Nature Sounds/Rain/" in rel(p)],
        "wind": [p for p in files if "99Sounds Nature Sounds/Wind/" in rel(p)],
        "sea_waves": [p for p in files if regex_any(rel(p), [r"sea[_ -]?waves?", r"ocean", r"waves?.*sand beach", r"sand beach"])],
        "crackling_fire": [],
    }
    out = {}
    for sub, matched in selectors.items():
        five, dur, usable, srs = duration_sum(matched)
        out[sub] = {
            "mechanism": "folder/path strict selector" if matched else "NOT PRESENT",
            "clips": len(matched),
            "five_sec": five,
            "sample_rates": dict(srs),
        }
    return out


def fsc22_counts_strict(ds_path: Path) -> dict:
    csv_path = next(ds_path.rglob("*.csv"), None)
    if not csv_path:
        return {}
    df = pd.read_csv(csv_path)
    audio_root = next((p for p in ds_path.rglob("*") if p.is_dir() and p.name == "Audio Wise V1.0"), ds_path)
    class_map = {
        "rain": ["Rain"],
        "sea_waves": [],
        "wind": ["Wind"],
        "crackling_fire": ["Fire"],
    }
    out = {}
    for sub, classes in class_map.items():
        if not classes:
            out[sub] = {"mechanism": "NOT PRESENT", "clips": 0, "five_sec": 0, "sample_rates": {}}
            continue
        mask = df["Class Name"].isin(classes)
        files = [audio_root / fn for fn in df.loc[mask, "Dataset File Name"].astype(str)]
        five, dur, usable, srs = duration_sum(files)
        out[sub] = {
            "mechanism": f"{rel(csv_path)} Class Name in {classes}",
            "clips": len(files),
            "five_sec": five,
            "sample_rates": dict(srs),
        }
    return out


def fsd_counts_strict(ds_path: Path) -> dict:
    gt = ds_path / "FSD50K.ground_truth" / "dev.csv"
    if not gt.exists():
        return {}
    df = pd.read_csv(gt)
    audio_dir = ds_path / "FSD50K.dev_audio"

    def has_token(label: str) -> pd.Series:
        return df["labels"].astype(str).str.split(",").apply(lambda vals: label in vals)

    token_map = {
        "rain": ["Rain"],
        "sea_waves": [],
        "wind": ["Wind"],
        "crackling_fire": ["Fire"],
        "car_horn": ["Vehicle_horn_and_car_horn_and_honking"],
        "engine_idling": ["Engine"],
        "siren": ["Siren"],
        "jackhammer": [],
    }
    mapped = {}
    for sub, labels in token_map.items():
        if not labels:
            mapped[sub] = {"mechanism": "NOT PRESENT", "clips": 0, "five_sec": 0, "audio_found": 0, "sample_rates": {}}
            continue
        mask = pd.Series(False, index=df.index)
        for label in labels:
            mask |= has_token(label)
        files = [audio_dir / f"{fid}.wav" for fid in df.loc[mask, "fname"]]
        five, dur, usable, srs = duration_sum(files)
        mapped[sub] = {
            "mechanism": f"FSD50K.ground_truth/dev.csv labels token in {labels}",
            "clips": int(mask.sum()),
            "five_sec": five,
            "audio_found": usable,
            "sample_rates": dict(srs),
        }
    raw_nature = {}
    raw_urban = {}
    for key in ["Rain", "Water", "Wind", "Fire", "Crackling"]:
        raw_nature[key] = int(df["labels"].astype(str).str.contains(key, case=False, regex=False).sum())
    for key in ["Car_horn", "Engine", "Siren", "Jackhammer"]:
        raw_urban[key] = int(df["labels"].astype(str).str.contains(key, case=False, regex=False).sum())
    return {"mapped": mapped, "nature_raw": raw_nature, "urban_raw": raw_urban}


def report_dataset(name: str) -> dict:
    ds_path = ROOT / name
    all_audio = audio_files(ds_path)
    all_audio_with_nonstd = audio_files(ds_path, include_nonstandard=True)
    meta = metadata_files(ds_path)
    parent_names = sorted({p.parent.name for p in all_audio})
    samples = random.Random(12).sample(all_audio, min(5, len(all_audio)))
    sample_info_files = []
    used_parents = set()
    for p in all_audio:
        if p.parent not in used_parents:
            sample_info_files.append(p)
            used_parents.add(p.parent)
        if len(sample_info_files) == 3:
            break
    sample_infos = [{"path": rel(p), **librosa_audio_info(p)} for p in sample_info_files]
    all_infos = [audio_info(p) for p in random.Random(7).sample(all_audio, min(250, len(all_audio)))]
    sr_counts = Counter(str(i.get("sample_rate")) for i in all_infos if i.get("ok"))
    formats = Counter(p.suffix.lower().lstrip(".") for p in all_audio_with_nonstd)
    low_sr = any(i.get("ok") and int(i.get("sample_rate", 0)) < 16000 for i in all_infos)
    silent = any(i.get("silent") for i in all_infos if i.get("ok"))
    clipped = any(i.get("clipped") for i in all_infos if i.get("ok"))

    if name == "ESC-50-master":
        nature = esc50_counts(ds_path)
    elif name == "4060432":
        fsd = fsd_counts_strict(ds_path)
        nature = {k: v for k, v in fsd.get("mapped", {}).items() if k in NATURE_KEYS}
    elif name == "FSC22_forest":
        nature = fsc22_counts_strict(ds_path)
    elif name == "99Sounds Nature Sounds":
        nature = nature99_counts(ds_path)
    elif name == "freefield1010":
        nature = freefield_counts(ds_path)
    else:
        nature = generic_nature_counts(ds_path)

    urban = urbansound_counts(ds_path) if name == "urbansound8k" else {}
    if name == "4060432":
        fsd = fsd_counts_strict(ds_path)
        urban = {k: v for k, v in fsd.get("mapped", {}).items() if k in URBAN_KEYS}

    useful_5s = sum(v.get("five_sec", 0) for v in nature.values()) + sum(v.get("five_sec", 0) for v in urban.values())
    flags = []
    if useful_5s < 100:
        flags.append("total useful 5s clips after filtering below 100")
    if any(ext not in {"wav", "mp3"} for ext in formats):
        flags.append(f"non-standard or less common formats present: {dict(formats)}")
    if not meta and not any(parent_names):
        flags.append("no metadata or label mechanism found")
    elif not meta:
        flags.append("no metadata found; folder/path labels only")
    if low_sr:
        flags.append("sample rate below 16000 Hz observed")
    if silent:
        flags.append("silent sampled audio observed")
    if clipped:
        flags.append("clipping observed in sampled audio")
    if all_audio_with_nonstd and not all_audio:
        flags.append("only non-requested audio extension found")

    decision = "USE"
    if not all_audio or useful_5s < 100:
        decision = "SKIP"
    elif flags:
        decision = "USE WITH CAUTION"

    return {
        "name": name,
        "exists": ds_path.exists(),
        "tree": tree_dirs(ds_path) if ds_path.exists() else [],
        "root_non_audio": [rel(p) for p in root_non_audio(ds_path)] if ds_path.exists() else [],
        "root_previews": {rel(p): read_first_lines(p, 10) for p in root_non_audio(ds_path) if p.suffix.lower() in {".csv", ".txt", ".md"} or p.name.lower().startswith("readme")},
        "subfolders": sorted([p.name for p in ds_path.iterdir() if p.is_dir()]) if ds_path.exists() else [],
        "audio_count": len(all_audio),
        "audio_count_including_webm": len(all_audio_with_nonstd),
        "sample_files": [{"path": rel(p), "parent": rel(p.parent)} for p in samples],
        "audio_parent_names": parent_names,
        "metadata": [metadata_summary(p) for p in meta],
        "audio_quality_samples": sample_infos,
        "sample_rate_counts_sampled": dict(sr_counts),
        "formats": dict(formats),
        "nature": nature,
        "urban": urban,
        "fsd_extra": fsd_counts_strict(ds_path) if name == "4060432" else {},
        "freefield_tags": dict(freefield_metadata_index(ds_path)[1].most_common(80)) if name == "freefield1010" else {},
        "flags": flags,
        "decision": decision,
    }


def compact_list(values: list[str], limit: int = 80) -> str:
    if not values:
        return "None"
    shown = values[:limit]
    suffix = "" if len(values) <= limit else f" ... (+{len(values) - limit} more)"
    return ", ".join(shown) + suffix


def write_report(reports: list[dict], out: Path) -> None:
    lines = []
    lines.append("# Audio Dataset Audit\n")
    lines.append("Note: Step 4 sample checks use librosa.load(sr=None, mono=False). Bulk duration/slice estimates use WAV headers for speed.\n")
    for r in reports:
        lines.append(f"\n## {r['name']}\n")
        lines.append("### Step 1 - Identify Dataset\n")
        lines.append("Folder structure up to 3 levels deep:\n")
        lines.extend([f"- `{x}`" for x in r["tree"][:150]] or ["- None"])
        lines.append(f"\nRoot non-audio files: {compact_list(r['root_non_audio'], 40)}\n")
        for path, preview in r["root_previews"].items():
            lines.append(f"First 10 lines of `{path}`:")
            lines.append("```text")
            lines.extend(preview)
            lines.append("```")
        lines.append(f"Subfolder names: {compact_list(r['subfolders'], 100)}\n")

        lines.append("### Step 2 - Count Audio Files\n")
        lines.append(f"Total requested-extension audio files: {r['audio_count']} (`wav/mp3/flac/ogg/m4a/aiff`). Including WebM/nonstandard: {r['audio_count_including_webm']}.")
        lines.append("Sample filenames:")
        lines.extend([f"- `{s['path']}` parent `{s['parent']}`" for s in r["sample_files"]] or ["- None"])
        lines.append(f"Unique parent folder names containing requested audio: {compact_list(r['audio_parent_names'], 150)}\n")

        lines.append("### Step 3 - Inspect Metadata Files\n")
        if not r["metadata"]:
            lines.append("No CSV/JSON/txt/md metadata files found.\n")
        for m in r["metadata"]:
            lines.append(f"Metadata `{m['path']}`")
            lines.append(f"- Columns/keys: {m.get('columns')}")
            lines.append(f"- Total row/key count: {m.get('rows')}")
            if m.get("label_values"):
                for col, vals in m["label_values"].items():
                    lines.append(f"- Unique label-like values in `{col}`: {compact_list(vals, 120)}")
            lines.append("- First 10 rows/lines:")
            lines.append("```text")
            for row in m.get("head", []):
                lines.append(str(row))
            lines.append("```")
        if r["name"] == "freefield1010":
            lines.append(f"Freefield1010 aggregated top JSON tags across per-clip metadata: {r['freefield_tags']}\n")

        lines.append("### Step 4 - Audio Quality Check\n")
        if not r["audio_quality_samples"]:
            lines.append("No requested-extension audio files to sample.\n")
        for info in r["audio_quality_samples"]:
            lines.append(f"- `{info['path']}`: format={info.get('format')}, ok={info.get('ok')}, sample_rate={info.get('sample_rate')}, duration={info.get('duration')}, channels={info.get('channels')}, silent={info.get('silent')}, clipped={info.get('clipped')}, error={info.get('error')}")
        lines.append(f"Sample-rate counts from sampled files: {r['sample_rate_counts_sampled']}; formats: {r['formats']}\n")

        lines.append("### Step 5 - Nature Subclass Mapping\n")
        for sub in ["rain", "sea_waves", "wind", "crackling_fire"]:
            v = r["nature"].get(sub)
            if not v or v.get("clips", 0) == 0:
                lines.append(f"- {sub}: NOT PRESENT")
            else:
                lines.append(f"- {sub}: {v['mechanism']}; clips={v['clips']}; estimated 5s clips={v['five_sec']}; sample_rates={v.get('sample_rates')}")

        if r["name"] == "urbansound8k":
            lines.append("\n### Step 6 - Urban Subclass Mapping\n")
            for sub in ["car_horn", "engine_idling", "siren", "jackhammer"]:
                v = r["urban"].get(sub)
                if not v or v.get("clips", 0) == 0:
                    lines.append(f"- {sub}: NOT PRESENT")
                else:
                    lines.append(f"- {sub}: {v['mechanism']}; clips={v['clips']}; estimated 5s clips={v['five_sec']}; sample_rates={v.get('sample_rates')}")

        if r["name"] == "4060432":
            lines.append("\n### FSD50K Requested Label Checks\n")
            lines.append(f"- Nature raw label contains counts: {r['fsd_extra'].get('nature_raw')}")
            lines.append(f"- Urban raw label contains counts: {r['fsd_extra'].get('urban_raw')}")

        lines.append("\n### Step 7 - Problem Flags\n")
        lines.extend([f"- {f}" for f in r["flags"]] or ["- None"])
        lines.append(f"Decision: {r['decision']}\n")

    lines.append("\n## Step 8 - Summary Table\n")
    lines.append("| Folder | Dataset Name | Total Audio Files | Nature Subclasses Found | Urban Subclasses Found | Label Mechanism | Sample Rate | Decision: Use/Skip |")
    lines.append("|---|---:|---:|---|---|---|---|---|")
    for r in reports:
        nature_found = ", ".join([f"{k}:{v.get('clips', 0)}/{v.get('five_sec', 0)}x5s" for k, v in r["nature"].items() if v.get("clips", 0)]) or "None"
        urban_found = ", ".join([f"{k}:{v.get('clips', 0)}/{v.get('five_sec', 0)}x5s" for k, v in r["urban"].items() if v.get("clips", 0)]) or "None"
        label_mech = "metadata" if r["metadata"] else ("folder/path" if r["audio_parent_names"] else "none")
        sr = ", ".join([f"{k}:{v}" for k, v in r["sample_rate_counts_sampled"].items()]) or "unknown"
        lines.append(f"| `{r['name']}` | {r['name']} | {r['audio_count']} | {nature_found} | {urban_found} | {label_mech} | {sr} | {r['decision']} |")
    out.write_text("\n".join(lines))


def main() -> None:
    reports = [report_dataset(name) for name in DATASETS]
    out = ROOT / "audio_dataset_audit_report.md"
    write_report(reports, out)
    print(out)


if __name__ == "__main__":
    main()
