#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import hashlib
import json
import os
import random
import re
import shutil
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm


# Paths

LEVIR_ROOT = os.environ.get("LEVIR_ROOT", "data/LEVIR-MCI-dataset")
LEVIR_CAPTION_JSON = os.path.join(LEVIR_ROOT, "LevirCCcaptions.json")

SECOND_ROOT = os.environ.get("SECOND_ROOT", "data/SECOND-CC-AUG")
SECOND_CAPTION_JSON = os.path.join(SECOND_ROOT, "SECOND-CC-AUG.json")

OUTPUT_ROOT = "./RS-OmniEdit-Content"
SOURCE_DIR = os.path.join(OUTPUT_ROOT, "source")
TARGET_DIR = os.path.join(OUTPUT_ROOT, "target")
OUTPUT_JSON = os.path.join(OUTPUT_ROOT, "data.json")
SUMMARY_JSON = os.path.join(OUTPUT_ROOT, "build_summary.json")
CACHE_JSON = os.path.join(OUTPUT_ROOT, "instruction_cache_llm.json")


# Model config

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")

client = AsyncOpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
)
MODEL_NAME = os.environ.get("RS_UNIEDIT_LLM_MODEL", "Qwen/Qwen3.6-35B-A3B")


# Build config

SEED = 42
CONCURRENCY_LIMIT = 32
REQUEST_TIMEOUT_SECONDS = 90
SERVICE_CHECK_TIMEOUT_SECONDS = 3
MAX_RETRIES = 2
TEST_SIZE = 200
CACHE_SAVE_EVERY = 50
CREATE_BACKUP = True
USE_SECOND_CC = True
USE_SECOND_CC_AUGMENT = False


# Heuristics

NO_CHANGE_PATTERNS = [
    r"\bno change\b",
    r"\bno changes\b",
    r"\bno difference\b",
    r"\bno differences\b",
    r"\bnothing has changed\b",
    r"\bnothing changed\b",
    r"\bunchanged\b",
    r"\bidentical\b",
    r"\bthe scene is the same\b",
    r"\bseem identical\b",
    r"\balmost nothing has changed\b",
    r"\bwithout change\b",
]

OBJECT_TERMS = [
    "building",
    "buildings",
    "house",
    "houses",
    "road",
    "roads",
    "lane",
    "lanes",
    "bridge",
    "bridges",
    "parking lot",
    "parking lots",
    "parking",
    "villa",
    "villas",
    "airport",
    "ship",
    "ships",
    "container",
    "containers",
    "stadium",
    "dam",
    "harbor",
    "port",
    "factory",
    "warehouse",
    "warehouses",
]

CHANGE_VERBS = [
    "appear",
    "appears",
    "appeared",
    "vanish",
    "vanishes",
    "vanished",
    "add",
    "added",
    "build",
    "built",
    "construct",
    "constructed",
    "remove",
    "removed",
    "demolish",
    "demolished",
    "disappear",
    "disappears",
    "replace",
    "replaced",
    "replacing",
    "expand",
    "expanded",
    "grow",
    "grew",
]

APPEARANCE_ONLY_TERMS = [
    "brown",
    "dark",
    "greened",
    "greener",
    "turned green",
    "turned brown",
    "became dark",
    "got dark",
    "color",
    "colour",
    "snow",
    "fog",
    "cloud",
]

COMMAND_HEADS = ("Add", "Build", "Construct", "Remove", "Replace", "Expand")


# Prompts

PROMPT = """You are constructing a remote sensing image editing dataset for object-level content changes.

Input:
You will receive five change captions for the same image pair.

Task:
1. Decide whether the captions describe an editable object/layout change.
2. If the pair has no meaningful change, or only vague appearance/color/season/atmosphere changes, output "NONE".
3. Otherwise write one concise English image editing instruction that covers the main object-level changes in the target image.

Requirements:
1. Output English only.
2. Use a direct imperative command.
3. Keep it short and natural.
4. Mention the main object/layout edits only.
5. If multiple major changes are present, cover them in one command.
6. Do not mention captions, descriptions, before/after images, or explanations.
7. If the change is only about color, vegetation tone, seasonal appearance, snow, fog, cloud, or other non-object appearance changes, output "NONE".
8. Output strictly as JSON with one field named "instruction".

Captions:
1. {caption_1}
2. {caption_2}
3. {caption_3}
4. {caption_4}
5. {caption_5}

Output format:
{{
  "instruction": "..."
}}
"""




# Utilities

def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_instruction(text: str) -> str:
    text = normalize_text(text)
    return text.strip(" .,!?:;\"'")


def contains_pattern(text: str, patterns: List[str]) -> bool:
    text = normalize_text(text)
    return any(re.search(p, text) for p in patterns)


def contains_term(text: str, terms: List[str]) -> bool:
    text = normalize_text(text)
    return any(t in text for t in terms)


def fast_copy(src: str, dst: str) -> None:
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def load_cache(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def save_cache(cache: Dict[str, str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def backup_file(path: str) -> None:
    if not CREATE_BACKUP or not os.path.exists(path):
        return
    if os.path.isdir(path):
        return
    tag = "bak_content_rebuild"
    backup_path = f"{path}.{tag}"
    shutil.copy2(path, backup_path)
    print(f"[Info] Backup: {backup_path}")


def ensure_dirs() -> None:
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(SOURCE_DIR, exist_ok=True)
    os.makedirs(TARGET_DIR, exist_ok=True)


def clear_output_images() -> None:
    for root in [SOURCE_DIR, TARGET_DIR]:
        for name in os.listdir(root):
            if name.endswith(".png"):
                os.remove(os.path.join(root, name))


def is_no_change_text(text: str) -> bool:
    return contains_pattern(text, NO_CHANGE_PATTERNS)


def score_caption_for_object_change(text: str) -> int:
    low = normalize_text(text)
    score = 0

    if contains_term(low, OBJECT_TERMS):
        score += 2
    if contains_term(low, CHANGE_VERBS):
        score += 2
    if " and " in low:
        score += 1
    if contains_term(low, APPEARANCE_ONLY_TERMS):
        score -= 1
    if is_no_change_text(low):
        score -= 4

    return score


def classify_levir_item(item: dict) -> Tuple[bool, int]:
    captions = [s["raw"] for s in item["sentences"]]
    if all(is_no_change_text(c) for c in captions):
        return False, 0

    best_score = max(score_caption_for_object_change(c) for c in captions)
    keep = best_score >= 2
    return keep, best_score


def classify_second_item(item: dict) -> Tuple[bool, int]:
    if item.get("changeflag") != 1:
        return False, 0

    filename = item["filename"]
    if not USE_SECOND_CC_AUGMENT and "_random_augment" in filename:
        return False, 0

    captions = [s["raw"] for s in item["sentences"]]
    joined = " ".join(captions)

    if not contains_term(joined, OBJECT_TERMS):
        return False, 0

    if not contains_term(joined, CHANGE_VERBS):
        return False, 0

    obj_hits = sum(1 for c in captions if contains_term(c, OBJECT_TERMS))
    change_hits = sum(1 for c in captions if contains_term(c, CHANGE_VERBS))
    appearance_hits = sum(1 for c in captions if contains_term(c, APPEARANCE_ONLY_TERMS))

    score = obj_hits * 2 + change_hits - appearance_hits
    keep = score >= 4
    return keep, score


def is_valid_instruction(text: str) -> bool:
    if not isinstance(text, str):
        return False

    ins = re.sub(r"\s+", " ", text.strip().strip('"').strip("'"))
    if not ins:
        return False

    if ins.upper() == "NONE":
        return False

    if len(ins) < 8 or len(ins) > 180:
        return False

    low = ins.lower()
    banned_patterns = [
        r"\bcaption(s)?\b",
        r"\bdescription(s)?\b",
        r"\bbefore image\b",
        r"\bafter image\b",
        r"\bsame image\b",
        r"\bunchanged\b",
        r"\bno change\b",
        r"\bno differences?\b",
        r"\bseason\b",
        r"\bfog\b",
        r"\bcloud\b",
        r"\bsnow\b",
        r"\bjson\b",
        r"\binstruction:\b",
        r"^\s*here is\b",
    ]
    if any(re.search(p, low) for p in banned_patterns):
        return False

    if not contains_term(low, CHANGE_VERBS):
        return False

    if not contains_term(low, OBJECT_TERMS):
        return False

    if any(ch in ins for ch in "{}[]`"):
        return False

    return True


def get_levir_image_paths(item: dict) -> Tuple[str, str]:
    split = item["filepath"]
    filename = item["filename"]
    src = os.path.join(LEVIR_ROOT, "images", split, "A", filename)
    tgt = os.path.join(LEVIR_ROOT, "images", split, "B", filename)
    return src, tgt


def get_second_image_paths(item: dict) -> Tuple[str, str]:
    split = item["filepath"]
    filename = item["filename"]
    src = os.path.join(SECOND_ROOT, split, "rgb", "A", filename)
    tgt = os.path.join(SECOND_ROOT, split, "rgb", "B", filename)
    return src, tgt


def build_prompt(captions: List[str]) -> str:
    padded = list(captions[:5])
    while len(padded) < 5:
        padded.append("")
    return PROMPT.format(
        caption_1=padded[0].strip(),
        caption_2=padded[1].strip(),
        caption_3=padded[2].strip(),
        caption_4=padded[3].strip(),
        caption_5=padded[4].strip(),
    )


def check_local_model_service() -> bool:
    try:
        request = urllib.request.Request(
            url=f"{OPENAI_BASE_URL.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=SERVICE_CHECK_TIMEOUT_SECONDS) as resp:
            return resp.status == 200
    except Exception:
        return False


def make_cache_key(captions: List[str]) -> str:
    joined = " || ".join(normalize_text(x) for x in captions)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


async def generate_instruction_from_captions(
    captions: List[str],
    sem: asyncio.Semaphore,
    cache: Dict[str, str],
    use_llm: bool,
) -> str:
    cache_key = " || ".join(normalize_text(x) for x in captions)
    if cache_key in cache:
        return cache[cache_key]

    if not use_llm:
        cache[cache_key] = ""
        return ""

    prompt = build_prompt(captions)

    async with sem:
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        response_format={"type": "json_object"},
                    ),
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                text = resp.choices[0].message.content.strip()
                result = json.loads(text)
                instruction = str(result.get("instruction", "")).strip()
                cache[cache_key] = instruction
                return instruction
            except Exception as e:
                if attempt == MAX_RETRIES:
                    print(f"[Warning] Failed to generate instruction after retries: {e}")
                    break
                await asyncio.sleep(1.0 * (attempt + 1))

    cache[cache_key] = ""
    return ""


def fallback_instruction(captions: List[str]) -> str:
    candidates = []
    seen = set()

    scored = sorted(captions, key=lambda x: score_caption_for_object_change(x), reverse=True)
    for text in scored:
        low = normalize_text(text)
        if is_no_change_text(low):
            continue
        if not contains_term(low, CHANGE_VERBS):
            continue
        if not contains_term(low, OBJECT_TERMS):
            continue

        base = text.strip().rstrip(".")
        if not base:
            continue

        normalized = normalize_instruction(base)
        if normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(base)
        if len(candidates) == 2:
            break

    if not candidates:
        return ""

    merged = " and ".join(x[0].lower() + x[1:] if x else x for x in candidates)
    merged = re.sub(r"\s+", " ", merged).strip(" ,.")

    if merged.lower().startswith(("a ", "an ", "the ", "some ", "many ", "several ", "more ")):
        merged = "Add " + merged
    elif not re.match(rf"^({'|'.join(COMMAND_HEADS)})\b", merged, flags=re.IGNORECASE):
        merged = "Make these object-level edits: " + merged

    merged = merged[0].upper() + merged[1:]
    if not merged.endswith("."):
        merged += "."

    if merged.startswith("Make these object-level edits:"):
        merged = merged.replace("Make these object-level edits:", "Add", 1)
        merged = re.sub(r"\s+", " ", merged)

    return merged


def process_candidate_prefilter_only(candidate: dict) -> Optional[dict]:
    src_path = candidate["src_path"]
    tgt_path = candidate["tgt_path"]

    if not os.path.exists(src_path) or not os.path.exists(tgt_path):
        return None

    return dict(candidate)


async def process_candidate(
    candidate: dict,
    sem: asyncio.Semaphore,
    cache: Dict[str, str],
    use_llm: bool,
) -> Optional[dict]:
    src_path = candidate["src_path"]
    tgt_path = candidate["tgt_path"]
    captions = candidate["captions"]

    if not os.path.exists(src_path) or not os.path.exists(tgt_path):
        return None

    instruction = await generate_instruction_from_captions(captions, sem, cache, use_llm)
    if not is_valid_instruction(instruction):
        return None

    candidate = dict(candidate)
    candidate["instruction"] = re.sub(r"\s+", " ", instruction.strip())
    return candidate


def process_candidate_rule_only(candidate: dict) -> Optional[dict]:
    src_path = candidate["src_path"]
    tgt_path = candidate["tgt_path"]
    captions = candidate["captions"]

    if not os.path.exists(src_path) or not os.path.exists(tgt_path):
        return None

    instruction = fallback_instruction(captions)
    if not is_valid_instruction(instruction):
        return None

    candidate = dict(candidate)
    candidate["instruction"] = re.sub(r"\s+", " ", instruction.strip())
    return candidate


def choose_test_set(candidates: List[dict], n: int, rng: random.Random) -> Tuple[List[dict], List[dict]]:
    by_source = defaultdict(list)
    for item in candidates:
        by_source[item["source_dataset"]].append(item)

    sources = sorted(by_source)
    if not sources:
        return [], candidates

    base_quota = n // len(sources)
    remainder = n % len(sources)
    test = []
    used_pairs = set()

    for source_idx, source in enumerate(sources):
        quota = base_quota + (1 if source_idx < remainder else 0)
        ranked = sorted(
            by_source[source],
            key=lambda x: (
                -x["quality_score"],
                x["filename"],
            ),
        )

        for item in ranked:
            if sum(1 for x in test if x["source_dataset"] == source) >= quota:
                break
            key = (item["source_dataset"], item["filename"])
            if key in used_pairs:
                continue
            test.append(item)
            used_pairs.add(key)

    if len(test) < n:
        ranked = sorted(
            candidates,
            key=lambda x: (
                -x["quality_score"],
                x["source_dataset"],
                x["filename"],
            ),
        )
        for item in ranked:
            if len(test) >= n:
                break
            key = (item["source_dataset"], item["filename"])
            if key in used_pairs:
                continue
            test.append(item)
            used_pairs.add(key)

    train = [x for x in candidates if (x["source_dataset"], x["filename"]) not in used_pairs]
    rng.shuffle(train)
    rng.shuffle(test)
    return test, train


def choose_test_pool(candidates: List[dict], n: int) -> Tuple[List[dict], List[dict]]:
    ranked = sorted(
        candidates,
        key=lambda x: (
            -x["quality_score"],
            x["source_dataset"] != "LEVIR-MCI",
            x["filename"],
        ),
    )

    pool = []
    used_pairs = set()

    for item in ranked:
        if len(pool) >= n:
            break
        key = (item["source_dataset"], item["filename"])
        if key in used_pairs:
            continue
        used_pairs.add(key)
        pool.append(item)

    remaining = [x for x in candidates if (x["source_dataset"], x["filename"]) not in used_pairs]
    return pool, remaining


def make_record(pair_id: int, split: str, item: dict, image_name: str) -> dict:
    prefix = "levir" if item["source_dataset"] == "LEVIR-MCI" else "second"
    return {
        "pair_id": f"{prefix}_{pair_id:06d}",
        "source_dataset": item["source_dataset"],
        "split": split,
        "task": "Content",
        "subtask": "object_change",
        "track": "text_only",
        "source_image": f"source/{image_name}",
        "target_image": f"target/{image_name}",
        "reference_image": "",
        "instruction": item["instruction"],
    }


def validate_records(records: List[dict]) -> None:
    required_keys = [
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

    pair_ids = [r["pair_id"] for r in records]
    dup = [k for k, v in Counter(pair_ids).items() if v > 1]
    if dup:
        raise RuntimeError(f"Duplicate pair_id found: {dup[:10]}")

    split_counter = Counter(r["split"] for r in records)
    if split_counter["test"] != TEST_SIZE:
        raise RuntimeError(f"Expected {TEST_SIZE} test records, got {split_counter['test']}")

    for r in records:
        if list(r.keys()) != required_keys:
            raise RuntimeError(f"Invalid record keys: {r.keys()}")
        if r["task"] != "Content":
            raise RuntimeError(f"Invalid task field: {r['task']}")
        if r["track"] != "text_only":
            raise RuntimeError(f"Invalid track field: {r['track']}")
        if r["subtask"] != "object_change":
            raise RuntimeError(f"Invalid subtask field: {r['subtask']}")
        if r["reference_image"] != "":
            raise RuntimeError(f"Content text_only record should not have reference image: {r['pair_id']}")
        if r["split"] not in {"train", "test"}:
            raise RuntimeError(f"Invalid split: {r['split']}")
        if not is_valid_instruction(r["instruction"]):
            raise RuntimeError(f"Invalid instruction: {r['pair_id']} -> {r['instruction']}")


def load_levir_candidates() -> Tuple[List[dict], dict]:
    with open(LEVIR_CAPTION_JSON, "r", encoding="utf-8") as f:
        items = json.load(f)["images"]

    kept = []
    stats = Counter()

    for item in items:
        keep, score = classify_levir_item(item)
        stats["total"] += 1
        if not keep:
            stats["filtered_out"] += 1
            continue

        src_path, tgt_path = get_levir_image_paths(item)
        kept.append({
            "source_dataset": "LEVIR-MCI",
            "filename": item["filename"],
            "sample_id": f"LEVIR::{item['filename']}",
            "original_split": item["filepath"],
            "src_path": src_path,
            "tgt_path": tgt_path,
            "captions": [s["raw"] for s in item["sentences"]],
            "quality_score": score,
        })
        stats["kept"] += 1

    return kept, dict(stats)


def load_second_candidates() -> Tuple[List[dict], dict]:
    with open(SECOND_CAPTION_JSON, "r", encoding="utf-8") as f:
        items = json.load(f)["images"]

    kept = []
    stats = Counter()

    for item in items:
        stats["total"] += 1
        keep, score = classify_second_item(item)
        if not keep:
            stats["filtered_out"] += 1
            continue

        src_path, tgt_path = get_second_image_paths(item)
        kept.append({
            "source_dataset": "SECOND-CC",
            "filename": item["filename"],
            "sample_id": f"SECOND::{item['filename']}",
            "original_split": item["filepath"],
            "src_path": src_path,
            "tgt_path": tgt_path,
            "captions": [s["raw"] for s in item["sentences"]],
            "quality_score": score,
        })
        stats["kept"] += 1

    return kept, dict(stats)


async def build_dataset() -> None:
    rng = random.Random(SEED)
    ensure_dirs()

    print(f"[Info] MODEL_NAME: {MODEL_NAME}")
    print(f"[Info] OUTPUT_ROOT: {OUTPUT_ROOT}")
    use_llm = check_local_model_service()
    print(f"[Info] Local model service available: {use_llm}")

    levir_candidates, levir_stats = load_levir_candidates()
    second_candidates, second_stats = load_second_candidates() if USE_SECOND_CC else ([], {})

    print(f"[Info] LEVIR prefilter kept: {levir_stats.get('kept', 0)} / {levir_stats.get('total', 0)}")
    if USE_SECOND_CC:
        print(f"[Info] SECOND prefilter kept: {second_stats.get('kept', 0)} / {second_stats.get('total', 0)}")

    all_candidates = levir_candidates + second_candidates
    file_ready_candidates = [x for x in (process_candidate_prefilter_only(item) for item in all_candidates) if x is not None]
    rule_valid_candidates = [x for x in (process_candidate_rule_only(item) for item in all_candidates) if x is not None]
    print(f"[Info] Candidates with usable image pairs after prefilter: {len(file_ready_candidates)}")
    print(f"[Info] Valid content candidates after rule screening: {len(rule_valid_candidates)}")

    valid_candidates = file_ready_candidates if use_llm else rule_valid_candidates

    if len(valid_candidates) <= TEST_SIZE:
        raise RuntimeError(
            f"Not enough valid samples to build dataset. Got {len(valid_candidates)}, need more than {TEST_SIZE}."
        )

    if use_llm:
        sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
        cache = load_cache(CACHE_JSON)
        pending = []

        for item in valid_candidates:
            key = make_cache_key(item["captions"])
            cached_instruction = cache.get(key, "").strip()
            if is_valid_instruction(cached_instruction):
                item["instruction"] = re.sub(r"\s+", " ", cached_instruction)
            else:
                pending.append(item)

        print(f"[Info] Cached LLM instructions reused: {len(valid_candidates) - len(pending)}")
        print(f"[Info] Remaining samples for LLM generation: {len(pending)}")
        coros = [process_candidate(item, sem, cache, True) for item in pending]
        processed_count = 0
        for fut in tqdm(asyncio.as_completed(coros), total=len(coros), desc="LLM Generation"):
            refined = await fut
            if refined is not None:
                key = make_cache_key(refined["captions"])
                cache[key] = refined["instruction"]
            processed_count += 1
            if processed_count % CACHE_SAVE_EVERY == 0:
                save_cache(cache, CACHE_JSON)

        save_cache(cache, CACHE_JSON)

        llm_valid_candidates = []
        unresolved_after_llm = []
        for item in valid_candidates:
            key = make_cache_key(item["captions"])
            instruction = cache.get(key, "").strip()
            if is_valid_instruction(instruction):
                item["instruction"] = re.sub(r"\s+", " ", instruction)
                llm_valid_candidates.append(item)
            else:
                unresolved_after_llm.append(item)

        print(f"[Info] Dropped after LLM validation: {len(unresolved_after_llm)}")
        valid_candidates = llm_valid_candidates

        if unresolved_after_llm:
            print(f"[Warning] Some samples still missing valid LLM instructions: {len(unresolved_after_llm)}")

        if len(valid_candidates) <= TEST_SIZE:
            raise RuntimeError(
                f"Not enough LLM-valid samples to build dataset. Got {len(valid_candidates)}, need more than {TEST_SIZE}."
            )

        test_items, train_items = choose_test_set(valid_candidates, TEST_SIZE, rng)
    else:
        test_items, train_items = choose_test_set(valid_candidates, TEST_SIZE, rng)

    print(f"[Info] Selected test items: {len(test_items)}")
    print(f"[Info] Selected train items: {len(train_items)}")

    backup_file(OUTPUT_JSON)
    clear_output_images()

    final_records = []
    ordered_items = [("test", x) for x in test_items] + [("train", x) for x in train_items]

    for idx, (split, item) in enumerate(ordered_items):
        image_name = f"{idx:06d}.png"
        new_src = os.path.join(SOURCE_DIR, image_name)
        new_tgt = os.path.join(TARGET_DIR, image_name)
        fast_copy(item["src_path"], new_src)
        fast_copy(item["tgt_path"], new_tgt)
        final_records.append(make_record(idx, split, item, image_name))

    validate_records(final_records)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_records, f, ensure_ascii=False, indent=2)

    summary = {
        "seed": SEED,
        "test_size": TEST_SIZE,
        "total_records": len(final_records),
        "train_records": sum(1 for r in final_records if r["split"] == "train"),
        "test_records": sum(1 for r in final_records if r["split"] == "test"),
        "source_dataset_counts": dict(Counter(r["source_dataset"] for r in final_records)),
        "split_source_dataset_counts": {
            split: dict(Counter(r["source_dataset"] for r in final_records if r["split"] == split))
            for split in ["train", "test"]
        },
        "levir_prefilter": levir_stats,
        "second_prefilter": second_stats,
        "valid_candidates_after_generation": len(valid_candidates),
        "used_llm": use_llm,
        "concurrency_limit": CONCURRENCY_LIMIT if use_llm else 0,
        "cache_json": CACHE_JSON if use_llm else "",
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[Done] Rebuilt RS-OmniEdit-Content")
    print(f"[Done] Saved dataset: {OUTPUT_JSON}")
    print(f"[Done] Saved summary: {SUMMARY_JSON}")


def main() -> None:
    asyncio.run(build_dataset())


if __name__ == "__main__":
    main()
