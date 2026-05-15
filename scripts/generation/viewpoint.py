#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build the RS-OmniEdit viewpoint editing subset from SUES-200."""

import os
import re
import json
import shutil
import asyncio
import random
import traceback
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from tqdm import tqdm
from openai import AsyncOpenAI


# Paths and local model config

DATA_ROOT = os.environ.get("SUES200_ROOT", "data/SUES-200-512x512")
DRONE_ROOT = os.path.join(DATA_ROOT, "drone_view_512")

OUTPUT_ROOT = "./RS-OmniEdit-Viewpoint"
SOURCE_DIR = os.path.join(OUTPUT_ROOT, "source")
TARGET_DIR = os.path.join(OUTPUT_ROOT, "target")

OUTPUT_JSON = os.path.join(OUTPUT_ROOT, "data.json")
INSTRUCTION_POOL_JSON = os.path.join(OUTPUT_ROOT, "instruction_pools.json")
BUILD_SUMMARY_JSON = os.path.join(OUTPUT_ROOT, "build_summary.json")
RECORDS_WITH_META_JSON = os.path.join(OUTPUT_ROOT, "records_with_meta.json")

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")

client = AsyncOpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
    timeout=300.0,
)

MODEL_NAME = os.environ.get("RS_UNIEDIT_LLM_MODEL", "Qwen/Qwen3.6-35B-A3B")

# This is intentionally lower than 32 to avoid many long generation requests
# occupying the server at the same time.
CONCURRENCY_LIMIT = 8


# Generation config

SEED = 42
DRY_RUN = False

# If True, regenerate instruction pools.
# After the first successful run, you can set it False.
REGEN_POOL = True
INSTRUCTIONS_PER_REQUEST = 50
REQUESTS_PER_TASK = 4

TEST_SIZE = 200

SOURCE_DATASET_NAME = "SUES-200"
TASK_NAME = "viewpoint"
TRACK_NAME = "text_only"

HEIGHTS = [150, 200, 250, 300]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

COPY_IMAGES = True
CLEAR_EXISTING_OUTPUT_IMAGES = True

# If your vLLM/Qwen endpoint supports disabling thinking, keep True.
# If requests fail because the server does not accept this option,
# the code automatically retries without this extra body.
DISABLE_THINKING = True


# Required JSON key order

REQUIRED_RECORD_KEYS = [
    "pair_id",
    "source_dataset",
    "task",
    "subtask",
    "track",
    "split",
    "source_image",
    "target_image",
    "instruction",
]


# Utility

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def clean_instruction(text: str) -> str:
    if text is None:
        return ""

    text = str(text).strip()
    text = text.replace("\n", " ").strip()
    text = re.sub(r"^\s*[-*•\d\.\)\(]+\s*", "", text)
    text = text.strip(" '\"\t\r\n")
    text = re.sub(r"\s+", " ", text)

    # Keep only the first sentence if the model rambles.
    parts = re.split(r"(?<=[.!?])\s+", text)
    if parts:
        text = parts[0].strip()

    if len(text) > 0 and text[-1] not in ".!?":
        text += "."

    return text


def normalize_instruction_for_dedup(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,!?:;\"'")


def deduplicate_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        k = normalize_instruction_for_dedup(x)
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def ensure_output_dirs() -> None:
    ensure_dir(OUTPUT_ROOT)

    if CLEAR_EXISTING_OUTPUT_IMAGES:
        for d in [SOURCE_DIR, TARGET_DIR]:
            if os.path.exists(d):
                shutil.rmtree(d)

        for p in [
            OUTPUT_JSON,
            BUILD_SUMMARY_JSON,
            RECORDS_WITH_META_JSON,
        ]:
            if os.path.exists(p):
                os.remove(p)

        if REGEN_POOL and os.path.exists(INSTRUCTION_POOL_JSON):
            os.remove(INSTRUCTION_POOL_JSON)

    ensure_dir(SOURCE_DIR)
    ensure_dir(TARGET_DIR)


# Scan SUES-200

def collect_height_groups(drone_root: str) -> Dict[Tuple[str, str], Dict[int, str]]:
    """
    Collect:
        groups[(location_id, view_id)][height] = relative_path_from_DATA_ROOT

    Example:
        groups[("0005", "3")][150] = "drone_view_512/0005/150/3.jpg"
        groups[("0005", "3")][200] = "drone_view_512/0005/200/3.jpg"
    """
    groups: Dict[Tuple[str, str], Dict[int, str]] = defaultdict(dict)

    root = Path(drone_root)
    if not root.exists():
        raise FileNotFoundError(f"Drone root not found: {drone_root}")

    location_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    for loc_dir in tqdm(location_dirs, desc="Scanning SUES-200"):
        location_id = loc_dir.name

        for h in HEIGHTS:
            h_dir = loc_dir / str(h)
            if not h_dir.exists():
                continue

            for img_path in sorted(h_dir.iterdir()):
                if not is_image_file(img_path):
                    continue

                view_id = img_path.stem
                rel_path = img_path.relative_to(DATA_ROOT).as_posix()
                groups[(location_id, view_id)][h] = rel_path

    return groups


# Pair construction

def choose_nonrepeating_pairs_for_group(
    height_to_path: Dict[int, str],
    rng: random.Random,
) -> List[Tuple[int, int]]:
    """
    For a group like:
        (location_id, view_id) -> {150: ..., 200: ..., 250: ..., 300: ...}

    Requirements:
    - Each image is used at most once.
    - If all 4 heights exist, randomly choose one of the 3 perfect matchings:
        (150, 200) + (250, 300)
        (150, 250) + (200, 300)
        (150, 300) + (200, 250)

    For 3 heights:
        choose one random pair and leave one image unused.
    For 2 heights:
        choose that pair.
    """
    hs = sorted(height_to_path.keys())
    n = len(hs)

    if n < 2:
        return []

    if set(hs) == set(HEIGHTS):
        matchings = [
            [(150, 200), (250, 300)],
            [(150, 250), (200, 300)],
            [(150, 300), (200, 250)],
        ]
        return rng.choice(matchings)

    if n == 2:
        return [(hs[0], hs[1])]

    if n == 3:
        candidates = []
        for i in range(len(hs)):
            for j in range(i + 1, len(hs)):
                candidates.append((hs[i], hs[j]))
        return [rng.choice(candidates)]

    rng.shuffle(hs)
    pairs = []
    while len(hs) >= 2:
        a = hs.pop()
        b = hs.pop()
        pairs.append(tuple(sorted((a, b))))
    return pairs


def build_unoriented_records(
    groups: Dict[Tuple[str, str], Dict[int, str]],
    seed: int,
) -> List[dict]:
    """
    First build undirected non-repeating pairs.
    Each original image globally appears at most once.
    """
    rng = random.Random(seed)
    group_keys = list(groups.keys())
    rng.shuffle(group_keys)

    used_images = set()
    records = []

    for (location_id, view_id) in group_keys:
        height_to_path = groups[(location_id, view_id)]
        chosen_pairs = choose_nonrepeating_pairs_for_group(height_to_path, rng)

        for h1, h2 in chosen_pairs:
            p1 = height_to_path[h1]
            p2 = height_to_path[h2]

            if p1 in used_images or p2 in used_images:
                continue

            used_images.add(p1)
            used_images.add(p2)

            records.append({
                "location_id": location_id,
                "view_id": view_id,
                "height_a": int(h1),
                "image_a": p1,
                "height_b": int(h2),
                "image_b": p2,
            })

    return records


def orient_pairs_balanced(records: List[dict], seed: int) -> List[dict]:
    """
    Orient approximately half as raise and half as lower.
    """
    rng = random.Random(seed)
    records = records[:]
    rng.shuffle(records)

    n = len(records)
    n_raise = n // 2

    if n % 2 == 1 and rng.random() < 0.5:
        n_raise += 1

    final = []

    for idx, r in enumerate(records):
        h1, h2 = r["height_a"], r["height_b"]
        p1, p2 = r["image_a"], r["image_b"]

        low_h, high_h = sorted([h1, h2])
        low_p = p1 if h1 == low_h else p2
        high_p = p1 if h1 == high_h else p2

        if idx < n_raise:
            src_h, tgt_h = low_h, high_h
            src_p, tgt_p = low_p, high_p
            direction = "raise"
            subtask = "uav_height_switching_raise"
        else:
            src_h, tgt_h = high_h, low_h
            src_p, tgt_p = high_p, low_p
            direction = "lower"
            subtask = "uav_height_switching_lower"

        final.append({
            "location_id": r["location_id"],
            "view_id": r["view_id"],
            "source_height": int(src_h),
            "target_height": int(tgt_h),
            "height_delta": int(abs(tgt_h - src_h)),
            "direction": direction,
            "subtask": subtask,
            "source_image": src_p,
            "target_image": tgt_p,
        })

    return final


# Instruction generation

def transition_key(src_h: int, tgt_h: int) -> str:
    return f"{src_h}_to_{tgt_h}"


def build_pool_prompt(src_h: int, tgt_h: int, n: int) -> str:
    delta = abs(tgt_h - src_h)
    direction_word = "raise" if tgt_h > src_h else "lower"
    direction_phrase = "increase the UAV altitude" if tgt_h > src_h else "decrease the UAV altitude"

    prompt = f"""You are helping construct a remote sensing image editing dataset.

Task: Generate {n} diverse English image-editing instructions.

Scenario:
The source image is a UAV remote sensing image captured at about {src_h} meters altitude.
The target image should look like it was captured at about {tgt_h} meters altitude.
The edit should only {direction_phrase}.

Requirements:
1. Each instruction must sound like a natural human editing request.
2. Each instruction must be a direct command, not an explanation.
3. Only describe changing UAV capture height / altitude.
4. Do not mention weather, season, color, style, objects, buildings, roads, or scene preservation.
5. Some instructions can mention both {src_h} meters and {tgt_h} meters.
6. Some instructions can mention changing by about {delta} meters.
7. Some instructions can simply say to {direction_word} the altitude without exact numbers.
8. Keep each instruction concise and natural.
9. Output strictly in JSON format with a single field named "instructions".
10. Output no headings, no notes, no markdown, and no extra fields.

Output format:
{{
  "instructions": [
    "{'Raise' if tgt_h > src_h else 'Lower'} this {src_h}-meter UAV image to about {tgt_h} meters.",
    "{'Move this drone shot a bit higher.' if tgt_h > src_h else 'Move this drone shot a bit lower.'}",
    "{'Increase' if tgt_h > src_h else 'Decrease'} the UAV altitude by about {delta} meters."
  ]
}}
"""
    return prompt.strip()


def parse_instruction_lines_from_text(text: str) -> List[str]:
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        s = re.sub(r"^\s*[-*•\d\.\)\(]+\s*", "", s)
        s = clean_instruction(s)
        if len(s) >= 4:
            lines.append(s)
    return deduplicate_keep_order(lines)


def parse_instruction_json(text: str) -> List[str]:
    """
    Parse JSON response:
        {"instructions": ["...", "..."]}
    If JSON parsing fails, fall back to line parsing.
    """
    try:
        result = json.loads(text)
        values = result.get("instructions", [])
        if not isinstance(values, list):
            return []
        lines = []
        for v in values:
            s = clean_instruction(str(v))
            if len(s) >= 4:
                lines.append(s)
        return deduplicate_keep_order(lines)
    except Exception:
        return parse_instruction_lines_from_text(text)


def fallback_templates_for_transition(src_h: int, tgt_h: int, num_needed: int) -> List[str]:
    delta = abs(tgt_h - src_h)
    direction = "raise" if tgt_h > src_h else "lower"

    if direction == "raise":
        templates = [
            f"Raise this {src_h}-meter UAV image to about {tgt_h} meters.",
            f"Change this {src_h} m drone view to roughly {tgt_h} m.",
            f"Move this UAV image up to around {tgt_h} meters.",
            f"Increase the UAV altitude by about {delta} meters.",
            f"Raise the drone viewpoint by roughly {delta} meters.",
            "Shift this UAV image to a higher altitude.",
            "Move this drone shot a bit higher.",
            f"Bring this UAV view from about {src_h} meters to around {tgt_h} meters.",
            f"Push this aerial view higher by roughly {delta} meters.",
            f"Adjust this UAV image to roughly {tgt_h} meters altitude.",
            f"Lift this drone image by around {delta} meters.",
            "Set this UAV shot to a higher altitude.",
            f"Make this {src_h} m UAV view look like it was taken near {tgt_h} m.",
            "Raise the camera height for this drone image.",
            f"Take this UAV scene up by about {delta} meters.",
            f"Move the drone higher, from around {src_h} m toward {tgt_h} m.",
            f"Give this UAV shot the look of a higher {tgt_h} m capture.",
            "Edit this image as if the drone flew higher.",
            "Make this aerial view feel more elevated.",
            f"Raise the capture height by roughly {delta} meters.",
        ]
    else:
        templates = [
            f"Lower this {src_h}-meter UAV image to about {tgt_h} meters.",
            f"Change this {src_h} m drone view to roughly {tgt_h} m.",
            f"Move this UAV image down to around {tgt_h} meters.",
            f"Decrease the UAV altitude by about {delta} meters.",
            f"Lower the drone viewpoint by roughly {delta} meters.",
            "Shift this UAV image to a lower altitude.",
            "Move this drone shot a bit lower.",
            f"Bring this UAV view from about {src_h} meters down to around {tgt_h} meters.",
            f"Pull this aerial view lower by roughly {delta} meters.",
            f"Adjust this UAV image to roughly {tgt_h} meters altitude.",
            f"Drop this drone image by around {delta} meters.",
            "Set this UAV shot to a lower altitude.",
            f"Make this {src_h} m UAV view look like it was taken near {tgt_h} m.",
            "Lower the camera height for this drone image.",
            f"Take this UAV scene down by about {delta} meters.",
            f"Move the drone lower, from around {src_h} m toward {tgt_h} m.",
            f"Give this UAV shot the look of a lower {tgt_h} m capture.",
            "Edit this image as if the drone flew lower.",
            "Make this aerial view feel closer to the ground.",
            f"Lower the capture height by roughly {delta} meters.",
        ]

    out = []
    while len(out) < num_needed:
        out.extend(templates)
    return out[:num_needed]


def is_valid_instruction(ins: str, src_h: int, tgt_h: int) -> bool:
    if not isinstance(ins, str):
        return False

    text = clean_instruction(ins)
    if len(text) < 5 or len(text) > 180:
        return False

    low = text.lower()

    junk = [
        "requirements",
        "output format",
        "json",
        "here are",
        "instruction:",
        "note:",
        "as an ai",
        "i cannot",
        "explanation",
        "markdown",
    ]
    if any(x in low for x in junk):
        return False

    banned_edit_terms = [
        "season",
        "weather",
        "cloud",
        "fog",
        "rain",
        "snow",
        "color",
        "style",
        "building",
        "road",
        "vegetation",
        "object",
        "remove",
        "add",
        "replace",
    ]
    if any(x in low for x in banned_edit_terms):
        return False

    # Direction should be plausible.
    if tgt_h > src_h:
        allowed = ["raise", "higher", "increase", "up", "elevated", "lift", "altitude", "height"]
    else:
        allowed = ["lower", "decrease", "down", "drop", "closer", "altitude", "height"]

    if not any(x in low for x in allowed):
        return False

    return True


def deduplicate_and_filter_instructions(
    instructions: List[str],
    src_h: int,
    tgt_h: int,
) -> List[str]:
    cleaned = []
    seen = set()

    for x in instructions:
        ins = clean_instruction(x)
        if not is_valid_instruction(ins, src_h, tgt_h):
            continue

        k = normalize_instruction_for_dedup(ins)
        if k in seen:
            continue

        cleaned.append(ins)
        seen.add(k)

    return cleaned


async def request_instruction_batch_once(
    src_h: int,
    tgt_h: int,
    n: int,
    use_extra_body: bool,
) -> List[str]:
    prompt = build_pool_prompt(src_h, tgt_h, n)

    kwargs = dict(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.8,
        top_p=0.95,
        max_tokens=1600,
        response_format={"type": "json_object"},
    )

    if use_extra_body and DISABLE_THINKING:
        kwargs["extra_body"] = {
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }

    resp = await client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content.strip()
    return parse_instruction_json(text)


async def request_instruction_batch(
    src_h: int,
    tgt_h: int,
    n: int,
    semaphore: asyncio.Semaphore,
) -> List[str]:
    async with semaphore:
        try:
            return await request_instruction_batch_once(src_h, tgt_h, n, use_extra_body=True)
        except Exception as e1:
            # Retry once without extra_body, because some vLLM builds may not accept chat_template_kwargs.
            try:
                return await request_instruction_batch_once(src_h, tgt_h, n, use_extra_body=False)
            except Exception as e2:
                print(f"[WARN] LLM request failed for {src_h}_to_{tgt_h}: {repr(e2)}")
                return []


async def build_instruction_pool_for_one_transition(
    src_h: int,
    tgt_h: int,
    semaphore: asyncio.Semaphore,
) -> Tuple[str, List[str], dict]:
    key = transition_key(src_h, tgt_h)
    print(f"[Info] Generating instruction pool: {key}")

    async def one_request():
        return await request_instruction_batch(
            src_h=src_h,
            tgt_h=tgt_h,
            n=INSTRUCTIONS_PER_REQUEST,
            semaphore=semaphore,
        )

    results = await asyncio.gather(*[one_request() for _ in range(REQUESTS_PER_TASK)])

    raw = []
    for lines in results:
        raw.extend(lines)

    cleaned = deduplicate_and_filter_instructions(raw, src_h, tgt_h)

    target_min = INSTRUCTIONS_PER_REQUEST * REQUESTS_PER_TASK
    fallback_used = 0

    if len(cleaned) < target_min:
        needed = target_min - len(cleaned)
        fallbacks = fallback_templates_for_transition(src_h, tgt_h, needed * 2)
        fallbacks = deduplicate_and_filter_instructions(fallbacks, src_h, tgt_h)

        existing = set(normalize_instruction_for_dedup(x) for x in cleaned)
        for ins in fallbacks:
            k = normalize_instruction_for_dedup(ins)
            if k not in existing:
                cleaned.append(ins)
                existing.add(k)
                fallback_used += 1
            if len(cleaned) >= target_min:
                break

    if len(cleaned) == 0:
        cleaned = fallback_templates_for_transition(src_h, tgt_h, 20)
        fallback_used = len(cleaned)

    stats = {
        "key": key,
        "requested": target_min,
        "raw_returned": len(raw),
        "valid_unique": len(cleaned),
        "fallback_used": fallback_used,
    }

    return key, cleaned, stats


async def build_or_load_instruction_pools(records: List[dict]) -> Tuple[Dict[str, List[str]], List[dict]]:
    ensure_dir(OUTPUT_ROOT)

    if (not REGEN_POOL) and os.path.exists(INSTRUCTION_POOL_JSON):
        print(f"[Info] Loading cached instruction pools: {INSTRUCTION_POOL_JSON}")
        with open(INSTRUCTION_POOL_JSON, "r", encoding="utf-8") as f:
            pools = json.load(f)

        stats = []
        for key, values in pools.items():
            try:
                src_h, tgt_h = map(int, key.split("_to_"))
            except Exception:
                continue
            cleaned = deduplicate_and_filter_instructions(values, src_h, tgt_h)
            if len(cleaned) == 0:
                cleaned = fallback_templates_for_transition(src_h, tgt_h, 20)
            pools[key] = cleaned
            stats.append({
                "key": key,
                "requested": "loaded",
                "raw_returned": len(values),
                "valid_unique": len(cleaned),
                "fallback_used": 0,
            })

        return pools, stats

    ordered_tasks = sorted({
        (r["source_height"], r["target_height"])
        for r in records
    })

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    jobs = [
        build_instruction_pool_for_one_transition(src_h, tgt_h, semaphore)
        for (src_h, tgt_h) in ordered_tasks
    ]

    pools = {}
    stats = []

    for coro in tqdm(asyncio.as_completed(jobs), total=len(jobs), desc="Building instruction pools"):
        key, pool, s = await coro
        pools[key] = pool
        stats.append(s)

    save_json(pools, INSTRUCTION_POOL_JSON)
    print(f"[Info] Saved instruction pools to {INSTRUCTION_POOL_JSON}")

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


def assign_instructions(records: List[dict], instruction_pools: Dict[str, List[str]], seed: int) -> List[dict]:
    sampler = InstructionSampler(instruction_pools, seed=seed)

    final = []

    for r in records:
        key = transition_key(r["source_height"], r["target_height"])
        ins = sampler.sample(key)

        item = {
            "pair_id": "",
            "source_dataset": SOURCE_DATASET_NAME,
            "task": TASK_NAME,
            "subtask": r["subtask"],
            "track": TRACK_NAME,
            "split": "",
            "source_image": r["source_image"],
            "target_image": r["target_image"],
            "instruction": ins,
        }

        final.append(item)

    return final


# Train/Test split

def split_train_test(records: List[dict], test_size: int, seed: int):
    """
    Split into:
        test: exactly TEST_SIZE samples
        train: all remaining samples

    Test set is balanced between raise/lower as much as possible.
    """
    if len(records) < test_size:
        raise ValueError(f"Total records ({len(records)}) < TEST_SIZE ({test_size})")

    rng = random.Random(seed)

    raise_records = [r for r in records if r["subtask"] == "uav_height_switching_raise"]
    lower_records = [r for r in records if r["subtask"] == "uav_height_switching_lower"]

    rng.shuffle(raise_records)
    rng.shuffle(lower_records)

    test_raise = min(len(raise_records), test_size // 2)
    test_lower = min(len(lower_records), test_size - test_raise)

    if test_raise + test_lower < test_size:
        remaining = test_size - (test_raise + test_lower)

        leftover_raise = len(raise_records) - test_raise
        take_more_raise = min(leftover_raise, remaining)
        test_raise += take_more_raise
        remaining -= take_more_raise

        leftover_lower = len(lower_records) - test_lower
        take_more_lower = min(leftover_lower, remaining)
        test_lower += take_more_lower
        remaining -= take_more_lower

    if test_raise + test_lower != test_size:
        raise ValueError("Could not allocate exactly TEST_SIZE samples.")

    test_records = raise_records[:test_raise] + lower_records[:test_lower]
    train_records = raise_records[test_raise:] + lower_records[test_lower:]

    rng.shuffle(test_records)
    rng.shuffle(train_records)

    for i, r in enumerate(train_records, start=1):
        r["split"] = "train"
        r["pair_id"] = f"sues_height_train_{i:07d}"

    for i, r in enumerate(test_records, start=1):
        r["split"] = "test"
        r["pair_id"] = f"sues_height_test_{i:07d}"

    all_records = train_records + test_records
    return all_records, train_records, test_records


# Image copy / materialization

def materialize_images_copy(records: List[dict]) -> List[dict]:
    """
    Copy selected source/target images into:
        OUTPUT_ROOT/source/{pair_id}.jpg
        OUTPUT_ROOT/target/{pair_id}.jpg

    Then update JSON paths to:
        source/{pair_id}.jpg
        target/{pair_id}.jpg
    """
    ensure_dir(SOURCE_DIR)
    ensure_dir(TARGET_DIR)

    for r in tqdm(records, desc="Copying source/target images"):
        original_src_rel = r["source_image"]
        original_tgt_rel = r["target_image"]

        src_abs = os.path.join(DATA_ROOT, original_src_rel)
        tgt_abs = os.path.join(DATA_ROOT, original_tgt_rel)

        if not os.path.exists(src_abs):
            raise FileNotFoundError(f"Source image not found: {src_abs}")
        if not os.path.exists(tgt_abs):
            raise FileNotFoundError(f"Target image not found: {tgt_abs}")

        src_ext = Path(src_abs).suffix.lower()
        tgt_ext = Path(tgt_abs).suffix.lower()

        src_out_rel = f"source/{r['pair_id']}{src_ext}"
        tgt_out_rel = f"target/{r['pair_id']}{tgt_ext}"

        src_out_abs = os.path.join(OUTPUT_ROOT, src_out_rel)
        tgt_out_abs = os.path.join(OUTPUT_ROOT, tgt_out_rel)

        shutil.copy2(src_abs, src_out_abs)
        shutil.copy2(tgt_abs, tgt_out_abs)

        r["source_image"] = src_out_rel
        r["target_image"] = tgt_out_rel

    return records


# Validation, checks, and summary

def parse_height_from_original_path(path: str) -> int:
    # Expected:
    # drone_view_512/0005/150/3.jpg
    return int(path.split("/")[-2])


def summarize_records_from_original_paths(records: List[dict]) -> dict:
    subtask_counts = defaultdict(int)
    split_counts = defaultdict(int)
    transition_counts = defaultdict(int)

    for r in records:
        subtask_counts[r["subtask"]] += 1
        split_counts[r["split"]] += 1

        src_h = parse_height_from_original_path(r["source_image"])
        tgt_h = parse_height_from_original_path(r["target_image"])
        transition_counts[f"{src_h}_to_{tgt_h}"] += 1

    return {
        "num_records": len(records),
        "split_counts": dict(sorted(split_counts.items())),
        "subtask_counts": dict(sorted(subtask_counts.items())),
        "transition_counts": dict(sorted(transition_counts.items())),
    }


def check_no_original_image_reuse(records: List[dict]) -> List[dict]:
    used = {}
    duplicates = []

    for r in records:
        for role in ["source_image", "target_image"]:
            img = r[role]
            if img in used:
                duplicates.append({
                    "image": img,
                    "first_pair_id": used[img],
                    "second_pair_id": r["pair_id"],
                })
            else:
                used[img] = r["pair_id"]

    return duplicates


def check_output_images_exist(records: List[dict]) -> List[str]:
    missing = []

    for r in records:
        for key in ["source_image", "target_image"]:
            p = os.path.join(OUTPUT_ROOT, r[key])
            if not os.path.exists(p):
                missing.append(p)

    return missing


def validate_final_records(records: List[dict]) -> None:
    ids = [r["pair_id"] for r in records]
    duplicate_ids = [k for k, v in Counter(ids).items() if v > 1]
    if duplicate_ids:
        raise RuntimeError(f"Duplicate pair_id found: {duplicate_ids[:10]}")

    for r in records:
        if list(r.keys()) != REQUIRED_RECORD_KEYS:
            raise RuntimeError(f"Invalid JSON keys in {r.get('pair_id')}: {list(r.keys())}")

        if r["source_dataset"] != SOURCE_DATASET_NAME:
            raise RuntimeError(f"Invalid source_dataset: {r['source_dataset']}")

        if r["task"] != TASK_NAME:
            raise RuntimeError(f"Invalid task: {r['task']}")

        if r["track"] != TRACK_NAME:
            raise RuntimeError(f"Invalid track: {r['track']}")

        if r["split"] not in {"train", "test"}:
            raise RuntimeError(f"Invalid split: {r['split']}")

        if r["subtask"] not in {"uav_height_switching_raise", "uav_height_switching_lower"}:
            raise RuntimeError(f"Invalid subtask: {r['subtask']}")

        if not r["source_image"].startswith("source/"):
            raise RuntimeError(f"source_image should be under source/: {r['source_image']}")

        if not r["target_image"].startswith("target/"):
            raise RuntimeError(f"target_image should be under target/: {r['target_image']}")

        if not isinstance(r["instruction"], str) or len(r["instruction"].strip()) < 4:
            raise RuntimeError(f"Invalid instruction in {r['pair_id']}")


def make_meta_records(oriented_records: List[dict], final_records_before_copy: List[dict]) -> List[dict]:
    """
    Build debugging metadata before final records are copied and paths are rewritten.
    """
    final_by_key = {
        (r["source_image"], r["target_image"]): r
        for r in final_records_before_copy
    }

    meta_records = []

    for base in oriented_records:
        key = (base["source_image"], base["target_image"])
        final = final_by_key.get(key)
        if final is None:
            continue

        meta_records.append({
            "pair_id": final["pair_id"],
            "source_dataset": final["source_dataset"],
            "task": final["task"],
            "subtask": final["subtask"],
            "track": final["track"],
            "split": final["split"],
            "original_source_image": final["source_image"],
            "original_target_image": final["target_image"],
            "source_height": base["source_height"],
            "target_height": base["target_height"],
            "height_delta": base["height_delta"],
            "direction": base["direction"],
            "location_id": base["location_id"],
            "view_id": base["view_id"],
            "instruction": final["instruction"],
        })

    return meta_records


def print_pool_stats(pool_stats: List[dict]) -> None:
    print("\n========== Instruction Pool Stats ==========")
    for s in sorted(pool_stats, key=lambda x: x["key"]):
        print(
            f"{s['key']}: "
            f"requested={s['requested']}, "
            f"raw={s['raw_returned']}, "
            f"valid={s['valid_unique']}, "
            f"fallback={s['fallback_used']}"
        )


# Main

async def main() -> None:
    set_seed(SEED)
    ensure_output_dirs()

    print(f"[Info] DATA_ROOT: {DATA_ROOT}")
    print(f"[Info] DRONE_ROOT: {DRONE_ROOT}")
    print(f"[Info] OUTPUT_ROOT: {OUTPUT_ROOT}")
    print(f"[Info] MODEL_NAME: {MODEL_NAME}")
    print(f"[Info] CONCURRENCY_LIMIT: {CONCURRENCY_LIMIT}")
    print(f"[Info] REQUESTS_PER_TASK: {REQUESTS_PER_TASK}")
    print(f"[Info] INSTRUCTIONS_PER_REQUEST: {INSTRUCTIONS_PER_REQUEST}")

    print("\n========== Step 1: Scan SUES-200 ==========")
    groups = collect_height_groups(DRONE_ROOT)
    print(f"[INFO] Number of (location, view_id) groups found: {len(groups)}")

    print("\n========== Step 2: Build non-repeating pairs ==========")
    unoriented_records = build_unoriented_records(groups, seed=SEED)
    print(f"[INFO] Undirected pairs: {len(unoriented_records)}")

    print("\n========== Step 3: Orient pairs (half raise / half lower) ==========")
    oriented_records = orient_pairs_balanced(unoriented_records, seed=SEED + 999)
    print(f"[INFO] Directed pairs: {len(oriented_records)}")

    if DRY_RUN:
        oriented_records = oriented_records[:min(100, len(oriented_records))]
        print(f"[INFO] DRY_RUN=True, only keep {len(oriented_records)} pairs")

    print("\n========== Step 4: Build / load instruction pools ==========")
    instruction_pools, pool_stats = await build_or_load_instruction_pools(oriented_records)
    print_pool_stats(pool_stats)

    print("\n========== Step 5: Assign instructions ==========")
    final_records = assign_instructions(
        records=oriented_records,
        instruction_pools=instruction_pools,
        seed=SEED + 2026,
    )

    print("\n========== Step 6: Split train / test ==========")
    all_records, train_records, test_records = split_train_test(
        records=final_records,
        test_size=TEST_SIZE,
        seed=SEED + 2027,
    )

    print("\n========== Step 7: Check original image reuse ==========")
    duplicates = check_no_original_image_reuse(all_records)
    if len(duplicates) > 0:
        raise RuntimeError(f"Found duplicated original image usage: {duplicates[:5]}")
    print("[INFO] No original image reuse found.")

    print("\n========== Step 8: Build metadata and summary before copying ==========")
    meta_records = make_meta_records(oriented_records, all_records)
    all_summary = summarize_records_from_original_paths(all_records)
    train_summary = summarize_records_from_original_paths(train_records)
    test_summary = summarize_records_from_original_paths(test_records)

    print("\n========== Step 9: Copy source/target images ==========")
    if COPY_IMAGES:
        all_records = materialize_images_copy(all_records)

    print("\n========== Step 10: Check copied output images ==========")
    missing = check_output_images_exist(all_records)
    if len(missing) > 0:
        raise RuntimeError(f"Missing copied images: {missing[:5]}")
    print("[INFO] All copied source/target images exist.")

    print("\n========== Step 11: Validate final records ==========")
    validate_final_records(all_records)
    print("[INFO] Final records are valid.")

    print("\n========== Step 12: Save ==========")
    save_json(all_records, OUTPUT_JSON)
    save_json(meta_records, RECORDS_WITH_META_JSON)

    summary = {
        "data_root": DATA_ROOT,
        "drone_root": DRONE_ROOT,
        "output_root": OUTPUT_ROOT,
        "output_json": OUTPUT_JSON,
        "source_dir": SOURCE_DIR,
        "target_dir": TARGET_DIR,
        "copy_images": COPY_IMAGES,
        "clear_existing_output_images": CLEAR_EXISTING_OUTPUT_IMAGES,
        "seed": SEED,
        "test_size": TEST_SIZE,
        "regen_pool": REGEN_POOL,
        "instructions_per_request": INSTRUCTIONS_PER_REQUEST,
        "requests_per_task": REQUESTS_PER_TASK,
        "concurrency_limit": CONCURRENCY_LIMIT,
        "source_dataset_name": SOURCE_DATASET_NAME,
        "task_name": TASK_NAME,
        "track_name": TRACK_NAME,
        "heights": HEIGHTS,
        "num_groups": len(groups),
        "num_unoriented_pairs": len(unoriented_records),
        "num_oriented_pairs": len(oriented_records),
        "num_train": len(train_records),
        "num_test": len(test_records),
        "num_all": len(all_records),
        "duplicate_original_image_count": len(duplicates),
        "missing_output_image_count": len(missing),
        "all_summary": all_summary,
        "train_summary": train_summary,
        "test_summary": test_summary,
        "pool_stats": pool_stats,
    }

    save_json(summary, BUILD_SUMMARY_JSON)

    print("\n========== Done ==========")
    print(f"Data JSON : {OUTPUT_JSON}")
    print(f"Source dir: {SOURCE_DIR}")
    print(f"Target dir: {TARGET_DIR}")
    print(f"Pools JSON: {INSTRUCTION_POOL_JSON}")
    print(f"Summary   : {BUILD_SUMMARY_JSON}")
    print(f"Meta JSON : {RECORDS_WITH_META_JSON}")

    print("\nAll summary:")
    print(json.dumps(summary["all_summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Interrupted]")
    except Exception as e:
        print(f"\n[ERROR] {repr(e)}")
        traceback.print_exc()
        raise
