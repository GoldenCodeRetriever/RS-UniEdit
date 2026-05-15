#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build the RS-OmniEdit season editing subset from SeasoNet."""

import os
import re
import json
import shutil
import random
import asyncio
import hashlib
import traceback
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from openai import AsyncOpenAI

try:
    import tifffile
except Exception:
    tifffile = None

try:
    import rasterio as rio
except Exception:
    rio = None


# Paths and local model config

DATA_ROOT = os.environ.get("SEASONET_ROOT", "data/SeasoNet")
META_CSV = os.path.join(DATA_ROOT, "meta.csv")

# Your structure is expected to be:
#   SeasoNet/spring, SeasoNet/summer, SeasoNet/fall, SeasoNet/winter, SeasoNet/snow
EXTRACTED_ROOT = DATA_ROOT

OUTPUT_ROOT = "./RS-OmniEdit-Season"
SOURCE_DIR = os.path.join(OUTPUT_ROOT, "source")
TARGET_DIR = os.path.join(OUTPUT_ROOT, "target")
REFERENCE_DIR = os.path.join(OUTPUT_ROOT, "reference")

OUTPUT_TEXT_ONLY_JSON = os.path.join(OUTPUT_ROOT, "data_text_only.json")
OUTPUT_IMAGE_REFERENCED_JSON = os.path.join(OUTPUT_ROOT, "data_image_referenced.json")
INSTRUCTION_POOL_JSON = os.path.join(OUTPUT_ROOT, "instruction_pools.json")
SPLIT_LOCATIONS_JSON = os.path.join(OUTPUT_ROOT, "split_locations.json")
BUILD_SUMMARY_JSON = os.path.join(OUTPUT_ROOT, "build_summary.json")
RECORDS_WITH_META_JSON = os.path.join(OUTPUT_ROOT, "records_with_original_meta.json")

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")

client = AsyncOpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
)
MODEL_NAME = os.environ.get("RS_UNIEDIT_LLM_MODEL", "Qwen/Qwen3.6-35B-A3B")
CONCURRENCY_LIMIT = 32


# Generation config

SEED = 42
DRY_RUN = False

# If True, regenerate instruction pools. After the first successful run, you can set it False.
REGEN_POOL = True
INSTRUCTIONS_PER_REQUEST = 50
REQUESTS_PER_TASK = 3

TRACKS = ["text_only", "image-referenced"]
SEASONS = ["spring", "summer", "fall", "winter", "snow"]

# all = all directed transitions among five seasons, 20 subtasks.
DIRECTION_MODE = "all"  # choices: all, common, custom
CUSTOM_DIRECTIONS: List[Tuple[str, str]] = []
COMMON_DIRECTIONS = [
    ("spring", "summer"),
    ("summer", "fall"),
    ("fall", "winter"),
    ("winter", "spring"),
    ("winter", "snow"),
    ("snow", "winter"),
    ("summer", "snow"),
    ("snow", "spring"),
]

TARGET_COUNTS = {
    "train": {"text_only": 13500, "image-referenced": 13500},
    "val": {"text_only": 1200, "image-referenced": 1200},
    "test": {"text_only": 300, "image-referenced": 300},
}
SPLIT_RATIOS = {"train": 0.90, "val": 0.08, "test": 0.02}

MAX_CLOUD_PCT = 10.0
# To keep winter as ordinary/no-snow winter, non-snow seasons are filtered by snow coverage if available.
MAX_SNOW_PCT_FOR_NON_SNOW = 5.0

# Set > 0 if you want to filter pairs with weak visible season difference.
# This requires reading images and is slower. Default keeps it disabled.
MIN_MEAN_RGB_DELTA = 0.0

RESIZE_TO = 512
IMAGE_FORMAT = "png"  # png / jpg
RGB_TIF_SUFFIX = "_10m_RGB.tif"

COPY_IMAGES = True
USE_ABSOLUTE_OUTPUT_PATH = False
CLEAR_EXISTING_OUTPUT_IMAGES = True

# Manual column override when auto inference fails.
MANUAL_COLUMNS = {
    "season": None,
    "path": None,
    "grid": None,
    "x": None,
    "y": None,
    "location": None,
    "date": None,
    "cloud": None,
    "snow": None,
}
COORD_ROUND = 6


# Constants

TASK_NAME = "Season"
SOURCE_DATASET = "SeasoNet"
SPLITS = ["train", "val", "test"]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

SEASON_DISPLAY = {
    "spring": "spring",
    "summer": "summer",
    "fall": "autumn",
    "winter": "winter",
    "snow": "snowy",
}
SEASON_TERMS = {
    "spring": ["spring"],
    "summer": ["summer"],
    "fall": ["autumn", "fall"],
    "winter": ["winter"],
    "snow": ["snow", "snowy", "snow-covered", "snow covered"],
}

REQUIRED_RECORD_KEYS = [
    "pair_id",
    "source_dataset",
    "split",
    "task",
    "subtask",
    "track",
    "source_image",
    "target_image",
    "reference_image",
    "instruction",
]

RECORDS_WITH_META: List[dict] = []


# Utilities

def safe_name(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^\w\-\.]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def short_hash(text: str, n: int = 10) -> str:
    return hashlib.md5(str(text).encode("utf-8")).hexdigest()[:n]


def norm_col(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def json_safe(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        if np.isnan(x):
            return None
        return float(x)
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return x


def output_path_for_json(path: str) -> str:
    if USE_ABSOLUTE_OUTPUT_PATH:
        return os.path.abspath(path)
    return os.path.relpath(path, OUTPUT_ROOT).replace("\\", "/")


def ensure_dirs() -> None:
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    if CLEAR_EXISTING_OUTPUT_IMAGES:
        for d in [SOURCE_DIR, TARGET_DIR, REFERENCE_DIR]:
            if os.path.exists(d):
                shutil.rmtree(d)
        for p in [
            OUTPUT_TEXT_ONLY_JSON,
            OUTPUT_IMAGE_REFERENCED_JSON,
            SPLIT_LOCATIONS_JSON,
            BUILD_SUMMARY_JSON,
            RECORDS_WITH_META_JSON,
        ]:
            if os.path.exists(p):
                os.remove(p)
        if REGEN_POOL and os.path.exists(INSTRUCTION_POOL_JSON):
            os.remove(INSTRUCTION_POOL_JSON)

    os.makedirs(SOURCE_DIR, exist_ok=True)
    os.makedirs(TARGET_DIR, exist_ok=True)
    os.makedirs(REFERENCE_DIR, exist_ok=True)


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_instruction_for_dedup(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,!?:;\"'")


# Season and subtask helpers

def normalize_season(x: Any) -> Optional[str]:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    s = str(x).strip().lower()
    if s in {"spring", "spr", "sp"}:
        return "spring"
    if s in {"summer", "sum", "su"}:
        return "summer"
    if s in {"fall", "autumn", "aut", "fa"}:
        return "fall"
    if s in {"winter", "win", "wi"}:
        return "winter"
    if s in {"snow", "snowy", "snow_set", "snowy_winter", "winter_snow"}:
        return "snow"

    if "spring" in s:
        return "spring"
    if "summer" in s:
        return "summer"
    if "fall" in s or "autumn" in s:
        return "fall"
    # Put snow before winter so snowy_winter maps to snow when written in path/metadata.
    if "snow" in s:
        return "snow"
    if "winter" in s:
        return "winter"
    return None


def season_display(season: str) -> str:
    return SEASON_DISPLAY.get(season, season)


def target_phrase(season: str) -> str:
    if season == "snow":
        return "snowy conditions"
    return season_display(season)


def subtask_name(src: str, tgt: str) -> str:
    return f"{src}_to_{tgt}"


def parse_subtask(subtask: str) -> Tuple[str, str]:
    a, b = subtask.split("_to_", 1)
    return a, b


def get_directions() -> List[Tuple[str, str]]:
    if DIRECTION_MODE == "all":
        return [(a, b) for a in SEASONS for b in SEASONS if a != b]
    if DIRECTION_MODE == "common":
        return list(COMMON_DIRECTIONS)
    if DIRECTION_MODE == "custom":
        return list(CUSTOM_DIRECTIONS)
    raise ValueError(f"Unknown DIRECTION_MODE: {DIRECTION_MODE}")


def contains_target_season(text: str, target_season: str) -> bool:
    low = text.lower()
    return any(term in low for term in SEASON_TERMS[target_season])


# Metadata column inference

def find_col(columns: List[str], candidates: List[str]) -> Optional[str]:
    normalized = {norm_col(c): c for c in columns}
    for cand in candidates:
        key = norm_col(cand)
        if key in normalized:
            return normalized[key]
    for cand in candidates:
        key = norm_col(cand)
        for col in columns:
            if key in norm_col(col):
                return col
    return None


def infer_xy_cols(columns: List[str]) -> Tuple[Optional[str], Optional[str]]:
    pairs = [
        ("center_x", "center_y"),
        ("centre_x", "centre_y"),
        ("patch_center_x", "patch_center_y"),
        ("patch_centre_x", "patch_centre_y"),
        ("x_center", "y_center"),
        ("x_centre", "y_centre"),
        ("longitude", "latitude"),
        ("lon", "lat"),
        ("lng", "lat"),
        ("x", "y"),
    ]
    normalized = {norm_col(c): c for c in columns}
    for x, y in pairs:
        if norm_col(x) in normalized and norm_col(y) in normalized:
            return normalized[norm_col(x)], normalized[norm_col(y)]
    return None, None


def infer_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    columns = list(df.columns)
    x_col, y_col = infer_xy_cols(columns)

    cols = {
        "season": MANUAL_COLUMNS["season"] or find_col(columns, ["season", "season_name"]),
        "path": MANUAL_COLUMNS["path"] or find_col(columns, ["path", "sample_path", "filepath", "file_path", "filename", "relative_path"]),
        "grid": MANUAL_COLUMNS["grid"] or find_col(columns, ["grid", "grid_id", "which_grid", "grid_index"]),
        "x": MANUAL_COLUMNS["x"] or x_col,
        "y": MANUAL_COLUMNS["y"] or y_col,
        "location": MANUAL_COLUMNS["location"] or find_col(columns, ["location_id", "loc_id", "patch_id", "site_id"]),
        "date": MANUAL_COLUMNS["date"] or find_col(columns, ["date", "datetime", "acquisition_date", "acquisition_time", "timestamp"]),
        "cloud": MANUAL_COLUMNS["cloud"] or find_col(columns, ["cloud_coverage", "cloud_cover", "cloud_percentage", "cloud_percent", "cloud"]),
        "snow": MANUAL_COLUMNS["snow"] or find_col(columns, ["snow_coverage", "snow_cover", "snow_percentage", "snow_percent", "snow"]),
    }

    missing = []
    if cols["path"] is None:
        missing.append("path")
    if cols["season"] is None:
        # season can still be inferred from path, so this is not fatal.
        print("[Warning] No season column found. Will infer season from path.")
    if cols["location"] is None and (cols["x"] is None or cols["y"] is None):
        print("[Warning] No location or x/y columns found. Will infer location from sample folder name.")
    if missing:
        print("\n[ERROR] Failed to infer required columns:", missing)
        print("Available columns:")
        for i, c in enumerate(columns):
            print(f"  {i:03d}: {c}")
        raise RuntimeError("Please set MANUAL_COLUMNS at the top of the script.")

    return cols


def normalize_percent_series(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    valid = v.dropna()
    if len(valid) == 0:
        return v
    # Metadata may be in [0, 1] or [0, 100].
    if valid.quantile(0.95) <= 1.5:
        v = v * 100.0
    return v


def parse_location_from_path(raw_path: str, grid: str = "grid") -> str:
    """
    Example folder:
      31UGR_20180418T104021_50_001993_6_158971
    Use tile + coordinates and ignore date so different seasons of the same place match.
    """
    p = Path(str(raw_path))
    stem = p.stem
    if p.suffix.lower() in IMG_EXTS:
        stem = p.parent.name

    parts = stem.split("_")
    # tile, datetime, lat_int, lat_frac, lon_int, lon_frac
    if len(parts) >= 6 and re.match(r"\d{8}T\d{6}", parts[1]):
        tile = parts[0]
        lat = f"{parts[2]}.{parts[3]}"
        lon = f"{parts[4]}.{parts[5]}"
        return f"{grid}__{tile}__{lat}__{lon}"
    return f"{grid}__{stem}"


def build_location_key(row: pd.Series, cols: Dict[str, Optional[str]]) -> str:
    grid = str(row[cols["grid"]]) if cols.get("grid") else "grid"

    if cols.get("location"):
        return f"{grid}__{row[cols['location']]}"

    if cols.get("x") and cols.get("y"):
        try:
            x = f"{float(row[cols['x']]):.{COORD_ROUND}f}"
        except Exception:
            x = str(row[cols["x"]])
        try:
            y = f"{float(row[cols['y']]):.{COORD_ROUND}f}"
        except Exception:
            y = str(row[cols["y"]])
        return f"{grid}__{x}__{y}"

    return parse_location_from_path(str(row[cols["path"]]), grid=grid)


def prepare_dataframe(df_raw: pd.DataFrame, cols: Dict[str, Optional[str]]) -> pd.DataFrame:
    df = df_raw.copy()

    if cols.get("season"):
        df["_season_norm"] = df[cols["season"]].apply(normalize_season)
    else:
        df["_season_norm"] = df[cols["path"]].apply(normalize_season)

    # If path itself includes /spring/ etc., use it to correct missing season values.
    missing = df["_season_norm"].isna()
    if missing.any():
        df.loc[missing, "_season_norm"] = df.loc[missing, cols["path"]].apply(normalize_season)

    df = df[df["_season_norm"].isin(SEASONS)].copy()

    if cols.get("cloud"):
        df["_cloud_pct"] = normalize_percent_series(df[cols["cloud"]])
        df = df[(df["_cloud_pct"].isna()) | (df["_cloud_pct"] <= MAX_CLOUD_PCT)].copy()
    else:
        df["_cloud_pct"] = np.nan

    if cols.get("snow"):
        df["_snow_pct"] = normalize_percent_series(df[cols["snow"]])
        non_snow = df["_season_norm"] != "snow"
        df = df[(~non_snow) | (df["_snow_pct"].isna()) | (df["_snow_pct"] <= MAX_SNOW_PCT_FOR_NON_SNOW)].copy()
    else:
        df["_snow_pct"] = np.nan

    df["_location_key"] = df.apply(lambda r: build_location_key(r, cols), axis=1)
    df["_sample_identity"] = df.apply(lambda r: sample_identity(r, cols), axis=1)
    df = df.dropna(subset=["_season_norm", "_location_key"]).reset_index(drop=True)
    return df


# Path resolving and strict RGB reading

def sample_identity(row: pd.Series, cols: Dict[str, Optional[str]]) -> str:
    raw = str(row[cols["path"]])
    season = normalize_season(row.get("_season_norm", None)) or normalize_season(raw) or "unknown"
    grid = str(row[cols["grid"]]) if cols.get("grid") else "grid"
    return f"{season}__{grid}__{parse_location_from_path(raw, grid=grid)}__{short_hash(raw, 8)}"


def resolve_sample_dir_from_row(row_dict: Dict[str, Any], cols: Dict[str, Optional[str]]) -> Optional[Path]:
    raw = str(row_dict[cols["path"]])
    season = str(row_dict.get("_season_norm", normalize_season(raw) or ""))
    grid_val = str(row_dict[cols["grid"]]) if cols.get("grid") and cols["grid"] in row_dict else ""

    raw_path = Path(raw)
    candidates: List[Path] = []

    if raw_path.is_absolute():
        candidates.append(raw_path)

    candidates.extend([
        Path(EXTRACTED_ROOT) / raw,
        Path(DATA_ROOT) / raw,
        Path(EXTRACTED_ROOT) / season / raw,
        Path(DATA_ROOT) / season / raw,
    ])

    # If metadata path only stores sample folder name, try season/grid/sample.
    sample_name = raw_path.stem if raw_path.suffix.lower() in IMG_EXTS else raw_path.name
    if grid_val:
        candidates.extend([
            Path(EXTRACTED_ROOT) / season / grid_val / sample_name,
            Path(DATA_ROOT) / season / grid_val / sample_name,
        ])
    candidates.extend([
        Path(EXTRACTED_ROOT) / season / "grid1" / sample_name,
        Path(EXTRACTED_ROOT) / season / "grid2" / sample_name,
        Path(DATA_ROOT) / season / "grid1" / sample_name,
        Path(DATA_ROOT) / season / "grid2" / sample_name,
    ])

    for c in candidates:
        if c.exists():
            return c.parent if c.is_file() else c
    return None


def find_rgb_tif(sample_dir: Path) -> Optional[Path]:
    if sample_dir.is_file():
        if sample_dir.name.endswith(RGB_TIF_SUFFIX):
            return sample_dir
        return None

    direct = list(sample_dir.glob(f"*{RGB_TIF_SUFFIX}"))
    if direct:
        return sorted(direct)[0]

    recursive = list(sample_dir.rglob(f"*{RGB_TIF_SUFFIX}"))
    if recursive:
        return sorted(recursive)[0]
    return None


def read_tif(path: Path) -> np.ndarray:
    if tifffile is not None:
        return tifffile.imread(str(path))
    if rio is not None:
        with rio.open(path) as src:
            arr = src.read()
        return arr
    with Image.open(path) as img:
        return np.array(img)


def ensure_hwc(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    # Convert channel-first to channel-last.
    if arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
        arr = np.moveaxis(arr, 0, -1)
    return arr


def stretch_rgb_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = ensure_hwc(arr).astype(np.float32)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] >= 3:
        arr = arr[:, :, :3]
    else:
        raise ValueError(f"Unsupported channel count: {arr.shape}")

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    amin, amax = float(np.min(arr)), float(np.max(arr))

    if amax <= 1.5:
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if amin >= 0 and amax <= 255:
        return np.clip(arr, 0, 255).astype(np.uint8)

    # Joint stretch gives more stable natural-color visualization than independent per-channel stretch.
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(valid, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = (arr - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    # Mild gamma correction for display.
    out = np.power(out, 1 / 2.2)
    return (out * 255).astype(np.uint8)


def read_rgb_image_from_row(row_dict: Dict[str, Any], cols: Dict[str, Optional[str]]) -> Image.Image:
    sample_dir = resolve_sample_dir_from_row(row_dict, cols)
    if sample_dir is None:
        raise FileNotFoundError(f"Cannot resolve sample dir for path: {row_dict.get(cols['path'])}")

    rgb_path = find_rgb_tif(sample_dir)
    if rgb_path is None:
        raise FileNotFoundError(f"Cannot find *{RGB_TIF_SUFFIX} under: {sample_dir}")

    arr = read_tif(rgb_path)
    rgb = stretch_rgb_to_uint8(arr)
    img = Image.fromarray(rgb).convert("RGB")
    if RESIZE_TO and RESIZE_TO > 0:
        img = img.resize((RESIZE_TO, RESIZE_TO), Image.BICUBIC)
    return img


def mean_rgb_delta_from_rows(src_row: Dict[str, Any], tgt_row: Dict[str, Any], cols: Dict[str, Optional[str]]) -> float:
    src_img = np.array(read_rgb_image_from_row(src_row, cols)).astype(np.float32)
    tgt_img = np.array(read_rgb_image_from_row(tgt_row, cols)).astype(np.float32)
    src_mean = src_img.reshape(-1, 3).mean(axis=0)
    tgt_mean = tgt_img.reshape(-1, 3).mean(axis=0)
    return float(np.mean(np.abs(src_mean - tgt_mean)))


def save_image_from_row(
    row_dict: Dict[str, Any],
    dst_path: str,
    cols: Dict[str, Optional[str]],
) -> None:
    if os.path.exists(dst_path):
        return
    img = read_rgb_image_from_row(row_dict, cols)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if IMAGE_FORMAT.lower() in {"jpg", "jpeg"}:
        img.save(dst_path, quality=95)
    else:
        img.save(dst_path)


# Candidate construction and splitting

def choose_best_row(group: pd.DataFrame) -> pd.Series:
    sort_cols = []
    if "_cloud_pct" in group.columns:
        sort_cols.append("_cloud_pct")
    if "_snow_pct" in group.columns:
        sort_cols.append("_snow_pct")
    if sort_cols:
        return group.sort_values(sort_cols, ascending=True, na_position="last").iloc[0]
    return group.iloc[0]


def assign_splits(location_keys: List[str]) -> Dict[str, str]:
    keys = list(location_keys)
    rng = random.Random(SEED)
    rng.shuffle(keys)

    n = len(keys)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])

    split_map = {}
    for i, key in enumerate(keys):
        if i < n_train:
            split_map[key] = "train"
        elif i < n_train + n_val:
            split_map[key] = "val"
        else:
            split_map[key] = "test"
    return split_map


def build_candidates(df: pd.DataFrame, directions: List[Tuple[str, str]]) -> List[dict]:
    candidates = []
    for loc, group in tqdm(df.groupby("_location_key"), desc="Building season candidates"):
        split = group["_split"].iloc[0]
        season_to_row: Dict[str, pd.Series] = {}
        for season, sg in group.groupby("_season_norm"):
            season_to_row[season] = choose_best_row(sg)

        for src, tgt in directions:
            if src not in season_to_row or tgt not in season_to_row:
                continue
            src_row = season_to_row[src].to_dict()
            tgt_row = season_to_row[tgt].to_dict()

            if MIN_MEAN_RGB_DELTA > 0:
                try:
                    delta = mean_rgb_delta_from_rows(src_row, tgt_row, GLOBAL_COLS)
                    if delta < MIN_MEAN_RGB_DELTA:
                        continue
                except Exception:
                    continue

            candidates.append({
                "location_key": loc,
                "split": split,
                "source_season": src,
                "target_season": tgt,
                "subtask": subtask_name(src, tgt),
                "source_row": src_row,
                "target_row": tgt_row,
                "source_identity": src_row["_sample_identity"],
                "target_identity": tgt_row["_sample_identity"],
            })
    return candidates


def build_reference_index(df: pd.DataFrame) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    index: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for _, row in df.iterrows():
        key = (row["_split"], row["_season_norm"])
        index[key].append(row.to_dict())
    rng = random.Random(SEED)
    for key in index:
        rng.shuffle(index[key])
    return index


def choose_reference(
    ref_index: Dict[Tuple[str, str], List[Dict[str, Any]]],
    split: str,
    target_season: str,
    source_location_key: str,
    source_identity: str,
    target_identity: str,
    used_identities: set,
) -> Optional[Dict[str, Any]]:
    candidates = ref_index.get((split, target_season), [])
    if not candidates:
        return None

    # Random start avoids always consuming the first references.
    if len(candidates) == 1:
        order = [0]
    else:
        start = random.randrange(len(candidates))
        order = list(range(start, len(candidates))) + list(range(0, start))

    for idx in order:
        r = candidates[idx]
        ident = r["_sample_identity"]
        if r["_location_key"] == source_location_key:
            continue
        if ident in {source_identity, target_identity}:
            continue
        if ident in used_identities:
            continue
        return r
    return None


# Instruction pool generation

def get_scenario(source_season: str, target_season: str, track: str) -> str:
    src = season_display(source_season)
    tgt = target_phrase(target_season)
    base = (
        f"Change an image from {src} to {tgt}. "
        f"The requested edit is a season appearance change."
    )
    if track == "image-referenced":
        base += " Use a reference image as guidance for the target season appearance."
    return base


def get_track_requirement(track: str) -> str:
    if track == "text_only":
        return "Do not mention any reference image."
    if track == "image-referenced":
        return "Mention the reference image naturally as guidance for the target season appearance."
    raise ValueError(f"Unknown track: {track}")


def fallback_instructions(source_season: str, target_season: str, track: str) -> List[str]:
    src = season_display(source_season)
    tgt = target_phrase(target_season)
    target_short = season_display(target_season) if target_season != "snow" else "snowy"

    base = [
        f"Turn this image into {tgt}.",
        f"Make this scene look like {tgt}.",
        f"Change the season from {src} to {tgt}.",
        f"Convert this from {src} to {tgt}.",
        f"Give this image a {target_short} appearance.",
        f"Transform this scene into {tgt}.",
        f"Edit this image to look like {tgt}.",
        f"Make it look like {tgt}.",
    ]

    if track == "image-referenced":
        base = [
            f"Use the reference image to turn this into {tgt}.",
            f"Make this look like {tgt} using the reference image.",
            f"Change this from {src} to {tgt} following the reference image.",
            f"Use the reference image as a guide to make this look like {tgt}.",
            f"Convert this from {src} to {tgt} using the reference image.",
            f"Follow the reference image and make this scene look like {tgt}.",
        ]
    return base


def is_valid_instruction_for_season(
    instruction: str,
    source_season: str,
    target_season: str,
    track: str,
) -> bool:
    if not isinstance(instruction, str):
        return False
    ins = re.sub(r"\s+", " ", instruction.strip().strip('"').strip("'"))
    if len(ins) < 6 or len(ins) > 180:
        return False
    low = ins.lower()

    # Only filter obvious junk, not natural wording.
    junk = [
        "requirements", "requirement", "output format", "json", "here are", "example",
        "note:", "instruction:", "goal:", "as an ai", "i cannot", "self-evaluation",
        "checklist", "valid instruction", "bad example", "good example",
    ]
    if any(x in low for x in junk):
        return False

    if any(ch in ins for ch in ["{", "}", "[", "]", "`"]):
        return False

    if track == "text_only" and "reference" in low:
        return False
    if track == "image-referenced" and "reference" not in low:
        return False

    # Light target-direction check: target season should be clear.
    if not contains_target_season(low, target_season):
        return False

    return True


def deduplicate_and_filter_instructions(
    instructions: List[str],
    source_season: str,
    target_season: str,
    track: str,
) -> List[str]:
    cleaned = []
    seen = set()
    for x in instructions:
        ins = re.sub(r"\s+", " ", str(x).strip().strip('"').strip("'"))
        ins = ins.strip(" ,")
        if not is_valid_instruction_for_season(ins, source_season, target_season, track):
            continue
        key = normalize_instruction_for_dedup(ins)
        if key in seen:
            continue
        cleaned.append(ins)
        seen.add(key)
    return cleaned


async def build_instruction_pool_one_request(source_season: str, target_season: str, track: str) -> List[str]:
    src = season_display(source_season)
    tgt = target_phrase(target_season)
    scenario = get_scenario(source_season, target_season, track)
    track_requirement = get_track_requirement(track)

    prompt = f"""You are helping construct a remote sensing image editing dataset.

Task: Generate a list of {INSTRUCTIONS_PER_REQUEST} diverse English editing instructions.

Scenario: {scenario}

Editing goal: change an image from {src} to {tgt}.

Requirements:
1. The instructions must sound like natural human editing requests.
2. Each instruction must be a direct command, not an explanation.
3. Keep the wording concise and natural.
4. The instruction should clearly express a season edit toward {tgt}, or a transition from {src} to {tgt}.
5. {track_requirement}
6. Do not output labels, headings, notes, explanations, or self-evaluation.
7. Output strictly in JSON format with a single field named "instructions".

Output format:
{{
  "instructions": [
    "Turn this image into {tgt}.",
    "Make this scene look like {tgt}.",
    "Change the season from {src} to {tgt}."
  ]
}}
"""

    try:
        resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content.strip()
        result = json.loads(text)
        values = result.get("instructions", [])
        if not isinstance(values, list):
            return []
        return [str(v).strip() for v in values if str(v).strip()]
    except Exception:
        return []


async def generate_one_instruction_pool(
    semaphore: asyncio.Semaphore,
    source_season: str,
    target_season: str,
    track: str,
) -> Tuple[str, List[str], dict]:
    subtask = subtask_name(source_season, target_season)
    key = f"{track}|{subtask}"
    print(f"[Info] Generating instruction pool: {key}")

    async def one():
        async with semaphore:
            return await build_instruction_pool_one_request(source_season, target_season, track)

    results = await asyncio.gather(*[one() for _ in range(REQUESTS_PER_TASK)])
    raw = []
    for r in results:
        raw.extend(r)

    cleaned = deduplicate_and_filter_instructions(raw, source_season, target_season, track)
    seen = set(normalize_instruction_for_dedup(x) for x in cleaned)
    fallback_used = 0
    for ins in fallback_instructions(source_season, target_season, track):
        if not is_valid_instruction_for_season(ins, source_season, target_season, track):
            continue
        k = normalize_instruction_for_dedup(ins)
        if k not in seen:
            cleaned.append(ins)
            seen.add(k)
            fallback_used += 1

    stats = {
        "key": key,
        "requested": INSTRUCTIONS_PER_REQUEST * REQUESTS_PER_TASK,
        "raw_returned": len(raw),
        "valid_unique": len(cleaned),
        "fallback_used": fallback_used,
    }
    return key, cleaned, stats


async def generate_instruction_pools(directions: List[Tuple[str, str]]) -> Tuple[Dict[str, List[str]], List[dict]]:
    if os.path.exists(INSTRUCTION_POOL_JSON) and not REGEN_POOL:
        print(f"[Info] Loading cached instruction pools: {INSTRUCTION_POOL_JSON}")
        with open(INSTRUCTION_POOL_JSON, "r", encoding="utf-8") as f:
            pools = json.load(f)
        stats = []
        for key, values in list(pools.items()):
            try:
                track, subtask = key.split("|", 1)
                src, tgt = parse_subtask(subtask)
            except Exception:
                continue
            cleaned = deduplicate_and_filter_instructions(values, src, tgt, track)
            for ins in fallback_instructions(src, tgt, track):
                if normalize_instruction_for_dedup(ins) not in {normalize_instruction_for_dedup(x) for x in cleaned}:
                    cleaned.append(ins)
            pools[key] = cleaned
            stats.append({"key": key, "requested": "loaded", "raw_returned": len(values), "valid_unique": len(cleaned), "fallback_used": 0})
        return pools, stats

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = []
    for src, tgt in directions:
        for track in TRACKS:
            tasks.append(generate_one_instruction_pool(semaphore, src, tgt, track))

    results = await asyncio.gather(*tasks)
    pools = {}
    stats = []
    for key, values, s in results:
        if not values:
            raise RuntimeError(f"Instruction pool is empty: {key}")
        pools[key] = values
        stats.append(s)

    save_json(pools, INSTRUCTION_POOL_JSON)
    return pools, stats


class InstructionSampler:
    def __init__(self, pools: Dict[str, List[str]], seed: int = 42):
        self.rng = random.Random(seed)
        self.pools = {}
        self.indices = {}
        for k, values in pools.items():
            values = list(values)
            if not values:
                raise RuntimeError(f"Empty instruction pool: {k}")
            self.rng.shuffle(values)
            self.pools[k] = values
            self.indices[k] = 0

    def sample(self, key: str) -> str:
        if key not in self.pools:
            raise KeyError(f"Missing instruction pool: {key}")
        idx = self.indices[key]
        if idx >= len(self.pools[key]):
            self.rng.shuffle(self.pools[key])
            idx = 0
        value = self.pools[key][idx]
        self.indices[key] = idx + 1
        return value


# Record construction

def make_output_filename(pair_id: str, role: str) -> str:
    ext = "jpg" if IMAGE_FORMAT.lower() in {"jpg", "jpeg"} else "png"
    return f"{safe_name(pair_id)}_{role}.{ext}"


def make_record(
    cand: dict,
    track: str,
    sampler: InstructionSampler,
    ref_index: Dict[Tuple[str, str], List[Dict[str, Any]]],
    used_identities: set,
    sample_idx: int,
    cols: Dict[str, Optional[str]],
) -> Optional[dict]:
    split = cand["split"]
    src_id = cand["source_identity"]
    tgt_id = cand["target_identity"]

    if src_id in used_identities or tgt_id in used_identities:
        return None

    ref_row = None
    ref_id = ""
    if track == "image-referenced":
        ref_row = choose_reference(
            ref_index=ref_index,
            split=split,
            target_season=cand["target_season"],
            source_location_key=cand["location_key"],
            source_identity=src_id,
            target_identity=tgt_id,
            used_identities=used_identities,
        )
        if ref_row is None:
            return None
        ref_id = ref_row["_sample_identity"]

    pair_id = f"seasonet_{split}_{sample_idx:06d}_{cand['subtask']}_{track}"
    pair_id = safe_name(pair_id)

    src_rel = f"source/{make_output_filename(pair_id, 'src')}"
    tgt_rel = f"target/{make_output_filename(pair_id, 'tgt')}"
    ref_rel = ""
    if track == "image-referenced":
        ref_rel = f"reference/{make_output_filename(pair_id, 'ref')}"

    if COPY_IMAGES:
        try:
            save_image_from_row(cand["source_row"], os.path.join(OUTPUT_ROOT, src_rel), cols)
            save_image_from_row(cand["target_row"], os.path.join(OUTPUT_ROOT, tgt_rel), cols)
            if ref_row is not None:
                save_image_from_row(ref_row, os.path.join(OUTPUT_ROOT, ref_rel), cols)
        except Exception as e:
            print(f"[Warning] Failed to save images for {pair_id}: {repr(e)}")
            return None

    pool_key = f"{track}|{cand['subtask']}"
    instruction = sampler.sample(pool_key)
    if not is_valid_instruction_for_season(instruction, cand["source_season"], cand["target_season"], track):
        return None

    used_identities.add(src_id)
    used_identities.add(tgt_id)
    if ref_id:
        used_identities.add(ref_id)

    record = {
        "pair_id": pair_id,
        "source_dataset": SOURCE_DATASET,
        "split": split,
        "task": TASK_NAME,
        "subtask": cand["subtask"],
        "track": track,
        "source_image": output_path_for_json(os.path.join(OUTPUT_ROOT, src_rel)),
        "target_image": output_path_for_json(os.path.join(OUTPUT_ROOT, tgt_rel)),
        "reference_image": output_path_for_json(os.path.join(OUTPUT_ROOT, ref_rel)) if ref_rel else "",
        "instruction": instruction,
    }

    RECORDS_WITH_META.append({
        "pair_id": pair_id,
        "split": split,
        "track": track,
        "subtask": cand["subtask"],
        "location_key": cand["location_key"],
        "source_identity": src_id,
        "target_identity": tgt_id,
        "reference_identity": ref_id,
        "source_meta": compact_meta(cand["source_row"], cols),
        "target_meta": compact_meta(cand["target_row"], cols),
        "reference_meta": compact_meta(ref_row, cols) if ref_row is not None else None,
    })
    return record


def compact_meta(row: Optional[Dict[str, Any]], cols: Dict[str, Optional[str]]) -> Optional[dict]:
    if row is None:
        return None
    def get(name: str):
        col = cols.get(name)
        if col and col in row:
            return json_safe(row.get(col))
        return None
    return {
        "season": json_safe(row.get("_season_norm")),
        "path": get("path"),
        "grid": get("grid"),
        "date": get("date"),
        "cloud_pct": json_safe(row.get("_cloud_pct")),
        "snow_pct": json_safe(row.get("_snow_pct")),
        "location_key": json_safe(row.get("_location_key")),
    }


def build_records(
    candidates: List[dict],
    pools: Dict[str, List[str]],
    ref_index: Dict[Tuple[str, str], List[Dict[str, Any]]],
    cols: Dict[str, Optional[str]],
) -> Tuple[List[dict], List[dict], Dict[str, set], List[dict]]:
    sampler = InstructionSampler(pools, seed=SEED)

    candidates_by_split_subtask: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for c in candidates:
        candidates_by_split_subtask[(c["split"], c["subtask"])].append(c)

    rng = random.Random(SEED)
    for key in candidates_by_split_subtask:
        rng.shuffle(candidates_by_split_subtask[key])

    directions = get_directions()
    subtask_order = [subtask_name(a, b) for a, b in directions]
    text_records: List[dict] = []
    image_records: List[dict] = []
    used_by_split: Dict[str, set] = {s: set() for s in SPLITS}
    build_stats = []
    global_sample_idx = 1

    for split in SPLITS:
        used = used_by_split[split]
        split_stats = {
            "split": split,
            "strict_unique": True,
            "target": TARGET_COUNTS[split],
            "generated": {"text_only": 0, "image-referenced": 0},
            "skipped_no_candidate": 0,
            "skipped_save_or_validation": 0,
        }

        for track in TRACKS:
            target_n = TARGET_COUNTS[split][track]
            pbar = tqdm(total=target_n, desc=f"Building {split}/{track}")
            cursor = 0
            stagnant_rounds = 0
            max_stagnant_rounds = max(1000, len(subtask_order) * 200)

            while split_stats["generated"][track] < target_n:
                subtask = subtask_order[cursor % len(subtask_order)]
                cursor += 1
                bucket = candidates_by_split_subtask.get((split, subtask), [])

                cand = None
                # Pop until a candidate is not already used.
                while bucket:
                    x = bucket.pop()
                    if x["source_identity"] in used or x["target_identity"] in used:
                        continue
                    cand = x
                    break

                if cand is None:
                    split_stats["skipped_no_candidate"] += 1
                    stagnant_rounds += 1
                    if stagnant_rounds >= max_stagnant_rounds:
                        print(f"[Warning] Stop early for {split}/{track}; no more valid candidates.")
                        break
                    continue

                record = make_record(
                    cand=cand,
                    track=track,
                    sampler=sampler,
                    ref_index=ref_index,
                    used_identities=used,
                    sample_idx=global_sample_idx,
                    cols=cols,
                )
                if record is None:
                    split_stats["skipped_save_or_validation"] += 1
                    stagnant_rounds += 1
                    if stagnant_rounds >= max_stagnant_rounds:
                        print(f"[Warning] Stop early for {split}/{track}; too many failed attempts.")
                        break
                    continue

                if track == "text_only":
                    text_records.append(record)
                else:
                    image_records.append(record)
                split_stats["generated"][track] += 1
                global_sample_idx += 1
                stagnant_rounds = 0
                pbar.update(1)

            pbar.close()

        split_stats["unique_original_images_used"] = len(used)
        build_stats.append(split_stats)

    return text_records, image_records, used_by_split, build_stats


# Validation and statistics

def validate_one_record(record: dict) -> None:
    if list(record.keys()) != REQUIRED_RECORD_KEYS:
        raise RuntimeError(f"Invalid JSON keys in {record.get('pair_id')}: {list(record.keys())}")
    if record["source_dataset"] != SOURCE_DATASET:
        raise RuntimeError(f"Invalid source_dataset: {record['source_dataset']}")
    if record["task"] != TASK_NAME:
        raise RuntimeError(f"Invalid task: {record['task']}")
    if record["split"] not in SPLITS:
        raise RuntimeError(f"Invalid split: {record['split']}")
    if record["track"] not in TRACKS:
        raise RuntimeError(f"Invalid track: {record['track']}")

    src, tgt = parse_subtask(record["subtask"])
    if not is_valid_instruction_for_season(record["instruction"], src, tgt, record["track"]):
        raise RuntimeError(f"Invalid instruction in {record['pair_id']}: {record['instruction']}")

    if record["track"] == "text_only" and record["reference_image"] != "":
        raise RuntimeError(f"text_only has reference_image: {record['pair_id']}")
    if record["track"] == "image-referenced" and record["reference_image"] == "":
        raise RuntimeError(f"image-referenced missing reference_image: {record['pair_id']}")

    if COPY_IMAGES:
        for key in ["source_image", "target_image", "reference_image"]:
            if record[key]:
                path = os.path.join(OUTPUT_ROOT, record[key])
                if not os.path.exists(path):
                    raise RuntimeError(f"Missing output image file: {path}")


def validate_records(records: List[dict], expected_track: str) -> None:
    ids = [r["pair_id"] for r in records]
    duplicates = [k for k, v in Counter(ids).items() if v > 1]
    if duplicates:
        raise RuntimeError(f"Duplicate pair_id in {expected_track}: {duplicates[:10]}")
    for r in records:
        if r["track"] != expected_track:
            raise RuntimeError(f"Track mismatch: expected {expected_track}, got {r['track']}")
        validate_one_record(r)


def validate_strict_unique_meta() -> None:
    for split in SPLITS:
        ids = []
        for item in RECORDS_WITH_META:
            if item["split"] != split:
                continue
            ids.append(item.get("source_identity"))
            ids.append(item.get("target_identity"))
            if item.get("reference_identity"):
                ids.append(item.get("reference_identity"))
        duplicates = [k for k, v in Counter(ids).items() if k and v > 1]
        if duplicates:
            raise RuntimeError(f"Strict unique violation in {split}: {duplicates[:10]}")


def print_stats(text_records: List[dict], image_records: List[dict], pool_stats: List[dict], build_stats: List[dict]) -> None:
    print("\n========== Global Statistics ==========")
    print(f"Text-only records: {len(text_records)}")
    print(f"Image-referenced records: {len(image_records)}")
    print(f"Total records: {len(text_records) + len(image_records)}")

    print("\n========== Build Stats ==========")
    for s in build_stats:
        print(json.dumps(s, ensure_ascii=False, indent=2))

    print("\n========== Instruction Pool Stats ==========")
    for s in sorted(pool_stats, key=lambda x: x["key"]):
        print(f"{s['key']}: raw={s['raw_returned']}, valid={s['valid_unique']}, fallback={s['fallback_used']}")

    for title, records in [("Text-only", text_records), ("Image-referenced", image_records)]:
        print(f"\n========== {title} Counts ==========")
        print(f"Total: {len(records)}")
        for field in ["split", "subtask"]:
            print(f"\n[{field}]")
            for k, v in sorted(Counter(r[field] for r in records).items()):
                print(f"{k}: {v}")


# Main

GLOBAL_COLS: Dict[str, Optional[str]] = {}


async def main() -> None:
    global GLOBAL_COLS
    random.seed(SEED)
    np.random.seed(SEED)

    print(f"[Info] DATA_ROOT: {DATA_ROOT}")
    print(f"[Info] META_CSV: {META_CSV}")
    print(f"[Info] EXTRACTED_ROOT: {EXTRACTED_ROOT}")
    print(f"[Info] OUTPUT_ROOT: {OUTPUT_ROOT}")
    print(f"[Info] MODEL_NAME: {MODEL_NAME}")
    print(f"[Info] SEASONS: {SEASONS}")
    print(f"[Info] TARGET_COUNTS: {TARGET_COUNTS}")

    if not os.path.exists(META_CSV):
        raise FileNotFoundError(f"meta.csv not found: {META_CSV}")

    ensure_dirs()

    print("\n[Info] Reading meta.csv...")
    df_raw = pd.read_csv(META_CSV, low_memory=False)
    print(f"[Info] Raw rows: {len(df_raw):,}")
    print(f"[Info] Raw columns: {len(df_raw.columns)}")

    cols = infer_columns(df_raw)
    GLOBAL_COLS = cols
    print("\n[Info] Detected columns:")
    for k, v in cols.items():
        print(f"  {k}: {v}")

    df = prepare_dataframe(df_raw, cols)
    print(f"\n[Info] Rows after cloud/snow filtering: {len(df):,}")
    print("[Info] Season distribution:")
    print(df["_season_norm"].value_counts(dropna=False).to_string())

    location_keys = sorted(df["_location_key"].unique().tolist())
    print(f"[Info] Unique locations: {len(location_keys):,}")

    split_map = assign_splits(location_keys)
    df["_split"] = df["_location_key"].map(split_map)
    split_locations = defaultdict(list)
    for loc, sp in split_map.items():
        split_locations[sp].append(loc)
    save_json({k: sorted(v) for k, v in split_locations.items()}, SPLIT_LOCATIONS_JSON)

    print("\n[Info] Location split distribution:")
    print(df.groupby("_split")["_location_key"].nunique().to_string())

    directions = get_directions()
    available_seasons = set(df["_season_norm"].unique())
    directions = [(a, b) for a, b in directions if a in available_seasons and b in available_seasons]
    print("\n[Info] Directions:")
    for a, b in directions:
        print(f"  {a} -> {b}")
    if not directions:
        raise RuntimeError("No valid directions available after filtering.")

    candidates = build_candidates(df, directions)
    print("\n[Info] Candidate counts:")
    print(f"  total: {len(candidates):,}")
    for split in SPLITS:
        print(f"  {split}: {sum(1 for c in candidates if c['split'] == split):,}")
    print("\n[Info] Candidate counts by subtask:")
    for k, v in sorted(Counter(c["subtask"] for c in candidates).items()):
        print(f"  {k}: {v:,}")

    if DRY_RUN:
        print("\n[Info] DRY_RUN=True. Stop before LLM calls and image export.")
        return

    pools, pool_stats = await generate_instruction_pools(directions)
    ref_index = build_reference_index(df)

    text_records, image_records, used_by_split, build_stats = build_records(
        candidates=candidates,
        pools=pools,
        ref_index=ref_index,
        cols=cols,
    )

    validate_records(text_records, "text_only")
    validate_records(image_records, "image-referenced")
    validate_strict_unique_meta()

    save_json(text_records, OUTPUT_TEXT_ONLY_JSON)
    save_json(image_records, OUTPUT_IMAGE_REFERENCED_JSON)
    save_json(RECORDS_WITH_META, RECORDS_WITH_META_JSON)

    summary = {
        "source_dataset": SOURCE_DATASET,
        "task": TASK_NAME,
        "seed": SEED,
        "data_root": DATA_ROOT,
        "meta_csv": META_CSV,
        "seasons": SEASONS,
        "direction_mode": DIRECTION_MODE,
        "directions": [subtask_name(a, b) for a, b in directions],
        "snow_policy": "snow is an independent fifth season; winter and snow are not merged",
        "detected_columns": cols,
        "filters": {
            "max_cloud_pct": MAX_CLOUD_PCT,
            "max_snow_pct_for_non_snow": MAX_SNOW_PCT_FOR_NON_SNOW,
            "min_mean_rgb_delta": MIN_MEAN_RGB_DELTA,
        },
        "target_counts": TARGET_COUNTS,
        "actual_counts": {
            "text_only": len(text_records),
            "image_referenced": len(image_records),
            "total": len(text_records) + len(image_records),
        },
        "strict_unique_originals_by_split": {k: len(v) for k, v in used_by_split.items()},
        "build_stats": build_stats,
        "pool_stats": pool_stats,
    }
    save_json(summary, BUILD_SUMMARY_JSON)

    print_stats(text_records, image_records, pool_stats, build_stats)

    print("\n[Done] Saved files:")
    print(f"  {OUTPUT_TEXT_ONLY_JSON}")
    print(f"  {OUTPUT_IMAGE_REFERENCED_JSON}")
    print(f"  {INSTRUCTION_POOL_JSON}")
    print(f"  {SPLIT_LOCATIONS_JSON}")
    print(f"  {BUILD_SUMMARY_JSON}")
    print(f"  {RECORDS_WITH_META_JSON}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Interrupted]")
    except Exception as e:
        print(f"\n[ERROR] {repr(e)}")
        traceback.print_exc()
        raise
