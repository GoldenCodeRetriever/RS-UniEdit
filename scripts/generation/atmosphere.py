#!/usr/bin/env python3
"""Build the RS-OmniEdit atmosphere task.

The script exposes both atmosphere data builders:
- fog: RRSHID clear/hazy editing records
- cloud: SEN12MS-CR clear/cloudy editing records
- all: run fog first, then append cloud records
"""

from __future__ import annotations

import argparse


def run_fog() -> None:
    """Build RRSHID fog and haze records."""
    __name__ = "rs_uniedit.atmosphere_fog"
    import os
    import re
    import json
    import shutil
    import random
    import asyncio
    from pathlib import Path
    from collections import Counter, defaultdict
    from typing import Dict, List, Tuple, Optional

    from tqdm import tqdm
    from openai import AsyncOpenAI


    # Fixed paths

    DATA_ROOT = os.environ.get("RRSHID_ROOT", "data/RRSHID")

    OUTPUT_ROOT = "./RS-OmniEdit-Atmosphere"
    SOURCE_DIR = os.path.join(OUTPUT_ROOT, "source")
    TARGET_DIR = os.path.join(OUTPUT_ROOT, "target")
    REFERENCE_DIR = os.path.join(OUTPUT_ROOT, "reference")

    OUTPUT_TEXT_ONLY_JSON = os.path.join(OUTPUT_ROOT, "data_text_only.json")
    OUTPUT_IMAGE_REFERENCED_JSON = os.path.join(OUTPUT_ROOT, "data_image_referenced.json")
    INSTRUCTION_POOL_JSON = os.path.join(OUTPUT_ROOT, "instruction_pools.json")

    CONCURRENCY_LIMIT = 32

    OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")

    client = AsyncOpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
    )

    MODEL_NAME = os.environ.get("RS_UNIEDIT_LLM_MODEL", "Qwen/Qwen3.6-35B-A3B")


    # Generation config

    SEED = 42

    # 第一次建议 True。生成好 instruction_pools.json 后，可以改成 False 复用。
    REGEN_POOL = True

    # 每个任务组合请求 3 次，每次 50 条，最后去重和过滤。
    INSTRUCTIONS_PER_REQUEST = 50
    REQUESTS_PER_TASK = 3

    TRACKS = ["text_only", "image-referenced"]

    # add_fog: clear -> hazy
    # remove_fog: hazy -> clear
    DIRECTIONS = ["add_fog", "remove_fog"]

    # auto: 优先文件名匹配；如果匹配不到但 clear/hazy 数量相等，则按排序配对。
    # name: 只允许文件名匹配。
    # sorted: 只按排序配对。
    PAIRING_MODE = "auto"

    # None 表示全部使用。
    MAX_PAIRS_PER_INTENSITY_SPLIT = None

    COPY_IMAGES = True
    USE_ABSOLUTE_OUTPUT_PATH = False
    CLEAR_EXISTING_OUTPUT_IMAGES = True

    # train 放宽复用，val/test 严格 unique。
    TRAIN_ALLOW_REUSE = True
    STRICT_UNIQUE_SPLITS = {"val", "test"}


    # Constants

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

    TASK_NAME = "Atmosphere"
    SOURCE_DATASET = "RRSHID"

    CONDITION_META = {
        "thin_fog": {
            "intensity": "thin",
            "atmosphere_name": "thin haze",
            "add_subtask": "clear_to_thin",
            "remove_subtask": "thin_to_clear",
        },
        "moderate_fog": {
            "intensity": "moderate",
            "atmosphere_name": "moderate fog",
            "add_subtask": "clear_to_moderate",
            "remove_subtask": "moderate_to_clear",
        },
        "thick_fog": {
            "intensity": "thick",
            "atmosphere_name": "thick fog",
            "add_subtask": "clear_to_thick",
            "remove_subtask": "thick_to_clear",
        },
    }

    SPLITS = ["train", "val", "test"]


    # Runtime registries

    # 记录“输出文件路径 -> 原始源文件绝对路径”
    # 这样同一张原图在 train 中多次使用时，可以复用同一个短文件名。
    OUTPUT_FILE_REGISTRY: Dict[str, str] = {}


    # Basic utilities

    def ensure_dirs():
        os.makedirs(OUTPUT_ROOT, exist_ok=True)

        if CLEAR_EXISTING_OUTPUT_IMAGES:
            for d in [SOURCE_DIR, TARGET_DIR, REFERENCE_DIR]:
                if os.path.exists(d):
                    shutil.rmtree(d)

        os.makedirs(SOURCE_DIR, exist_ok=True)
        os.makedirs(TARGET_DIR, exist_ok=True)
        os.makedirs(REFERENCE_DIR, exist_ok=True)


    def list_images(folder: Path) -> List[Path]:
        if not folder.exists():
            return []

        return sorted([
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ])


    def normalize_pair_key(path: Path) -> str:
        stem = path.stem.lower()

        suffix_patterns = [
            r"[_\- ]?clear$",
            r"[_\- ]?hazy$",
            r"[_\- ]?foggy$",
            r"[_\- ]?fog$",
            r"[_\- ]?gt$",
            r"[_\- ]?label$",
            r"[_\- ]?target$",
            r"[_\- ]?input$",
        ]

        for pat in suffix_patterns:
            stem = re.sub(pat, "", stem)

        stem = re.sub(r"\s+", "_", stem)
        return stem


    def safe_name(text: str) -> str:
        text = str(text)
        text = re.sub(r"[^\w\-\.]+", "_", text)
        text = re.sub(r"_+", "_", text)
        return text.strip("_")


    def build_image_map(paths: List[Path]) -> Dict[str, Path]:
        result = {}

        for p in paths:
            key = normalize_pair_key(p)
            if key not in result:
                result[key] = p

        return result


    def output_path_for_json(path: str) -> str:
        if USE_ABSOLUTE_OUTPUT_PATH:
            return os.path.abspath(path)
        return os.path.relpath(path, OUTPUT_ROOT).replace("\\", "/")


    def normalize_instruction_for_dedup(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        text = text.strip(" .,!?:;\"'")
        return text


    def infer_clear_hazy_from_path(src_path: str) -> str:
        parts = [p.lower() for p in Path(src_path).parts]
        if "clear" in parts:
            return "clear"
        if "hazy" in parts:
            return "hazy"

        # fallback
        low = src_path.lower()
        if "/clear/" in low or "\\clear\\" in low:
            return "clear"
        if "/hazy/" in low or "\\hazy\\" in low:
            return "hazy"

        return "image"


    def extract_image_id(src_path: str) -> str:
        # 直接用原始文件 stem 作为“图像编号”
        return safe_name(Path(src_path).stem)


    def build_short_output_stem(src_path: str) -> str:
        domain = infer_clear_hazy_from_path(src_path)
        image_id = extract_image_id(src_path)
        return safe_name(f"{SOURCE_DATASET}_{domain}_{image_id}")


    def copy_image(src_path: str, dst_dir: str) -> str:
        """
        使用短文件名保存：
          RRSHID_clear_图像编号.ext
          RRSHID_hazy_图像编号.ext

        如果：
        1) 同一个原始文件多次被复制到同一个目标目录，则复用同一个输出文件；
        2) 不同原始文件意外映射到了同一个短文件名，则自动加 _2, _3 ... 兜底。
        """
        src = Path(src_path).resolve()
        ext = src.suffix.lower() if src.suffix else ".png"

        stem = build_short_output_stem(str(src))
        candidate = Path(dst_dir) / f"{stem}{ext}"

        src_abs = str(src)
        dst_abs = str(candidate.resolve())

        if dst_abs in OUTPUT_FILE_REGISTRY:
            if OUTPUT_FILE_REGISTRY[dst_abs] == src_abs:
                if COPY_IMAGES and not candidate.exists():
                    shutil.copy2(src, candidate)
                return output_path_for_json(str(candidate))
            else:
                # 极少数短名冲突时，自动加后缀
                idx = 2
                while True:
                    candidate2 = Path(dst_dir) / f"{stem}_{idx}{ext}"
                    dst_abs2 = str(candidate2.resolve())

                    if dst_abs2 not in OUTPUT_FILE_REGISTRY:
                        OUTPUT_FILE_REGISTRY[dst_abs2] = src_abs
                        if COPY_IMAGES:
                            shutil.copy2(src, candidate2)
                        return output_path_for_json(str(candidate2))

                    if OUTPUT_FILE_REGISTRY[dst_abs2] == src_abs:
                        if COPY_IMAGES and not candidate2.exists():
                            shutil.copy2(src, candidate2)
                        return output_path_for_json(str(candidate2))

                    idx += 1
        else:
            OUTPUT_FILE_REGISTRY[dst_abs] = src_abs
            if COPY_IMAGES:
                shutil.copy2(src, candidate)
            return output_path_for_json(str(candidate))


    # Instruction validation

    def contains_any(text: str, keywords: List[str]) -> bool:
        return any(k in text for k in keywords)


    def is_valid_instruction_for_atmosphere(
        instruction: str,
        direction: str,
        track: str,
    ) -> bool:
        if not isinstance(instruction, str):
            return False

        ins = instruction.strip()
        if not ins:
            return False

        low = ins.lower()

        if len(ins) < 25 or len(ins) > 240:
            return False

        banned_phrases = [
            "core action",
            "checked",
            "avoid vague",
            "requirement",
            "requirements",
            "note:",
            "instruction:",
            "goal:",
            "output",
            "json",
            "markdown",
            "self-evaluation",
            "self evaluation",
            "as an ai",
            "i cannot",
            "here are",
            "valid instruction",
            "bad example",
            "good example",
        ]

        if any(x in low for x in banned_phrases):
            return False

        banned_chars = [":", "*", "/", "`", "{", "}", "[", "]"]
        if any(ch in ins for ch in banned_chars):
            return False

        if "?" in ins:
            return False

        banned_topics = [
            "cloud",
            "clouds",
            "cloudy",
            "season",
            "spring",
            "summer",
            "autumn",
            "fall",
            "winter",
            "resolution",
            "super-resolution",
            "super resolution",
            "upscale",
            "downsample",
            "viewpoint",
            "angle",
            "sensor",
            "sentinel",
            "landsat",
            "modis",
            "classification",
            "segmentation",
            "mask",
            "label map",
        ]

        if any(x in low for x in banned_topics):
            return False

        atmosphere_terms = [
            "haze",
            "hazy",
            "fog",
            "foggy",
            "dehaze",
            "mist",
            "misty",
        ]

        if not contains_any(low, atmosphere_terms):
            return False

        if direction == "add_fog":
            add_verbs = [
                "add",
                "introduce",
                "apply",
                "render",
                "simulate",
                "make",
                "turn",
                "create",
                "give",
            ]
            if not contains_any(low, add_verbs):
                return False

        elif direction == "remove_fog":
            remove_verbs = [
                "remove",
                "clear",
                "dehaze",
                "restore",
                "reduce",
                "recover",
                "clean",
            ]
            if not contains_any(low, remove_verbs):
                return False

        else:
            return False

        rs_terms = [
            "satellite",
            "aerial",
            "remote sensing",
            "overhead",
            "land-cover",
            "land cover",
            "roads",
            "buildings",
            "vegetation",
            "terrain",
            "water bodies",
            "geographic",
            "scene layout",
            "object positions",
        ]

        if not contains_any(low, rs_terms):
            return False

        preserve_terms = [
            "preserve",
            "keep",
            "maintain",
            "retain",
            "unchanged",
            "without changing",
            "without altering",
            "same",
        ]

        if not contains_any(low, preserve_terms):
            return False

        if track == "text_only":
            if "reference" in low:
                return False

        elif track == "image-referenced":
            if "reference" not in low:
                return False

        else:
            return False

        return True


    def deduplicate_and_filter_instructions(
        instructions: List[str],
        direction: str,
        track: str,
    ) -> List[str]:
        cleaned = []
        seen = set()

        for ins in instructions:
            ins = str(ins).strip()
            ins = re.sub(r"\s+", " ", ins)
            ins = ins.strip().strip('"').strip("'").strip(",")

            if not is_valid_instruction_for_atmosphere(ins, direction=direction, track=track):
                continue

            key = normalize_instruction_for_dedup(ins)

            if key in seen:
                continue

            cleaned.append(ins)
            seen.add(key)

        return cleaned


    # Read RRSHID pairs

    def discover_pairs() -> List[dict]:
        data_root = Path(DATA_ROOT).expanduser().resolve()
        all_pairs = []

        for condition_name in CONDITION_META.keys():
            condition_root = data_root / condition_name

            if not condition_root.exists():
                print(f"[Warning] Missing condition folder: {condition_root}")
                continue

            for split in SPLITS:
                clear_dir = condition_root / split / "clear"
                hazy_dir = condition_root / split / "hazy"

                clear_images = list_images(clear_dir)
                hazy_images = list_images(hazy_dir)

                if len(clear_images) == 0 or len(hazy_images) == 0:
                    print(
                        f"[Warning] Empty folder for {condition_name}/{split}: "
                        f"clear={len(clear_images)}, hazy={len(hazy_images)}"
                    )
                    continue

                if PAIRING_MODE == "sorted":
                    if len(clear_images) != len(hazy_images):
                        raise RuntimeError(
                            f"Sorted pairing requires equal number of images, but got "
                            f"{condition_name}/{split}: clear={len(clear_images)}, hazy={len(hazy_images)}"
                        )

                    selected = list(zip(sorted(clear_images), sorted(hazy_images)))

                    if MAX_PAIRS_PER_INTENSITY_SPLIT is not None:
                        random.shuffle(selected)
                        selected = selected[:MAX_PAIRS_PER_INTENSITY_SPLIT]

                    for idx, (clear_path, hazy_path) in enumerate(selected):
                        all_pairs.append({
                            "condition_name": condition_name,
                            "split": split,
                            "pair_key": f"sorted_{idx:06d}",
                            "clear_path": str(clear_path),
                            "hazy_path": str(hazy_path),
                        })

                    continue

                clear_map = build_image_map(clear_images)
                hazy_map = build_image_map(hazy_images)
                common_keys = sorted(set(clear_map.keys()) & set(hazy_map.keys()))

                if len(common_keys) == 0:
                    if PAIRING_MODE == "name":
                        raise RuntimeError(
                            f"No filename-matched pairs found in {condition_name}/{split}, "
                            f"and PAIRING_MODE is name."
                        )

                    if PAIRING_MODE == "auto" and len(clear_images) == len(hazy_images):
                        print(
                            f"[Warning] No filename match in {condition_name}/{split}. "
                            f"Falling back to sorted-order pairing."
                        )

                        selected = list(zip(sorted(clear_images), sorted(hazy_images)))

                        if MAX_PAIRS_PER_INTENSITY_SPLIT is not None:
                            random.shuffle(selected)
                            selected = selected[:MAX_PAIRS_PER_INTENSITY_SPLIT]

                        for idx, (clear_path, hazy_path) in enumerate(selected):
                            all_pairs.append({
                                "condition_name": condition_name,
                                "split": split,
                                "pair_key": f"sorted_{idx:06d}",
                                "clear_path": str(clear_path),
                                "hazy_path": str(hazy_path),
                            })

                        continue

                    raise RuntimeError(
                        f"No valid pairs found in {condition_name}/{split}. "
                        f"clear={len(clear_images)}, hazy={len(hazy_images)}"
                    )

                if MAX_PAIRS_PER_INTENSITY_SPLIT is not None:
                    random.shuffle(common_keys)
                    common_keys = common_keys[:MAX_PAIRS_PER_INTENSITY_SPLIT]
                    common_keys = sorted(common_keys)

                for key in common_keys:
                    all_pairs.append({
                        "condition_name": condition_name,
                        "split": split,
                        "pair_key": key,
                        "clear_path": str(clear_map[key]),
                        "hazy_path": str(hazy_map[key]),
                    })

        return all_pairs


    # Instruction pool generation

    def get_scenario(condition_name: str, direction: str, track: str) -> str:
        meta = CONDITION_META[condition_name]
        atmosphere_name = meta["atmosphere_name"]

        if direction == "add_fog":
            base = (
                f"Add {atmosphere_name} to a clear remote sensing image while keeping "
                f"the same land-cover layout, roads, buildings, vegetation, terrain, "
                f"water bodies, and object positions unchanged."
            )
        else:
            base = (
                f"Remove {atmosphere_name} from a hazy or foggy remote sensing image "
                f"and recover a clear view while keeping the same land-cover layout, "
                f"roads, buildings, vegetation, terrain, water bodies, and object positions unchanged."
            )

        if track == "image-referenced":
            base += (
                " The instruction should naturally mention using a reference image only as guidance "
                "for the target atmospheric appearance."
            )
        else:
            base += " The instruction should not mention any reference image."

        return base


    def get_track_requirement(track: str) -> str:
        if track == "text_only":
            return "Do not mention any reference image."
        if track == "image-referenced":
            return (
                "Mention the reference image naturally as guidance for the target fog or haze appearance, "
                "but make clear that the source scene layout should stay unchanged."
            )
        raise ValueError(f"Unknown track: {track}")


    def get_direction_requirement(direction: str) -> str:
        if direction == "add_fog":
            return (
                "Every instruction must ask to add, introduce, apply, render, simulate, make, "
                "or create haze or fog."
            )
        if direction == "remove_fog":
            return (
                "Every instruction must ask to remove, clear, dehaze, reduce, restore, "
                "or recover from haze or fog."
            )
        raise ValueError(f"Unknown direction: {direction}")


    def fallback_instructions(condition_name: str, direction: str, track: str) -> List[str]:
        meta = CONDITION_META[condition_name]
        atmosphere = meta["atmosphere_name"]

        if direction == "add_fog":
            base = [
                f"Add {atmosphere} to this satellite image while keeping the land-cover layout unchanged.",
                f"Introduce {atmosphere} into this aerial image without altering roads, buildings, or vegetation.",
                f"Render this overhead scene with {atmosphere} while preserving the terrain and object positions.",
                f"Apply {atmosphere} to this remote sensing image while maintaining the same geographic structure.",
                f"Make this satellite scene appear covered by {atmosphere} while keeping the ground layout the same.",
            ]
        else:
            base = [
                f"Remove the {atmosphere} from this satellite image while keeping the land-cover layout unchanged.",
                f"Clear the {atmosphere} from this aerial image without altering roads, buildings, or vegetation.",
                f"Dehaze this overhead scene while preserving the terrain and object positions.",
                f"Restore a clear view of this remote sensing image while maintaining the same geographic structure.",
                f"Reduce the {atmosphere} in this satellite scene while keeping the ground layout the same.",
            ]

        if track == "image-referenced":
            base = [
                "Use the reference image as guidance for the target atmospheric appearance, and " + x[0].lower() + x[1:]
                for x in base
            ]

        return base


    async def build_instruction_pool_one_request(
        condition_name: str,
        direction: str,
        track: str,
    ) -> List[str]:
        scenario = get_scenario(condition_name, direction, track)
        track_requirement = get_track_requirement(track)
        direction_requirement = get_direction_requirement(direction)

        prompt = f"""You are a remote sensing image editing data construction assistant.

    Task: Generate a list of {INSTRUCTIONS_PER_REQUEST} diverse, natural, and imperative image editing instructions in English.

    Scenario: {scenario}

    Requirements:
    1. Use varied imperative expressions that sound like real user editing requests.
    2. The instructions must be direct editing commands.
    3. Each instruction must be one natural sentence.
    4. {direction_requirement}
    5. {track_requirement}
    6. Mention remote sensing concepts naturally, such as satellite image, aerial image, overhead view, land-cover layout, roads, buildings, vegetation, water bodies, terrain, geographic structure, or object positions.
    7. The instruction must say that the underlying scene layout or object positions should be preserved.
    8. Do not use labels, headings, checklist language, explanations, or self-evaluation.
    9. Do not use phrases such as Core Action, Checked, Requirement, Note, Goal, or Instruction.
    10. Do not use slash alternatives such as haze or fog written with a slash. Choose one natural wording.
    11. Do not mention clouds, seasons, resolution changes, viewpoint changes, sensors, masks, segmentation maps, or classification maps.
    12. Output strictly in JSON format containing a list named instructions.

    Output format:
    {{
      "instructions": [
        "Remove the haze from this satellite image while keeping the land-cover layout unchanged.",
        "Add light fog to this aerial image without moving roads, buildings, or vegetation."
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
            instructions = result.get("instructions", [])

            if not isinstance(instructions, list):
                return []

            return [
                x.strip()
                for x in instructions
                if isinstance(x, str) and x.strip()
            ]

        except Exception:
            return []


    async def generate_one_instruction_pool(
        semaphore: asyncio.Semaphore,
        condition_name: str,
        direction: str,
        track: str,
    ) -> Tuple[str, List[str], dict]:
        key = f"{track}|{condition_name}|{direction}"
        print(f"[Info] Generating instruction pool: {key}")

        async def single_request():
            async with semaphore:
                return await build_instruction_pool_one_request(
                    condition_name=condition_name,
                    direction=direction,
                    track=track,
                )

        results = await asyncio.gather(
            *[single_request() for _ in range(REQUESTS_PER_TASK)]
        )

        merged = []
        for res in results:
            merged.extend(res)

        cleaned = deduplicate_and_filter_instructions(
            merged,
            direction=direction,
            track=track,
        )

        seen = set(normalize_instruction_for_dedup(x) for x in cleaned)
        fallback_used = 0

        for ins in fallback_instructions(condition_name, direction, track):
            if not is_valid_instruction_for_atmosphere(ins, direction=direction, track=track):
                continue

            key_norm = normalize_instruction_for_dedup(ins)

            if key_norm not in seen:
                cleaned.append(ins)
                seen.add(key_norm)
                fallback_used += 1

        stats = {
            "key": key,
            "requested": INSTRUCTIONS_PER_REQUEST * REQUESTS_PER_TASK,
            "raw_returned": len(merged),
            "valid_unique": len(cleaned),
            "fallback_used": fallback_used,
        }

        return key, cleaned, stats


    async def generate_instruction_pools() -> Tuple[Dict[str, List[str]], List[dict]]:
        if os.path.exists(INSTRUCTION_POOL_JSON) and not REGEN_POOL:
            print(f"[Info] Loading existing instruction pool: {INSTRUCTION_POOL_JSON}")

            with open(INSTRUCTION_POOL_JSON, "r", encoding="utf-8") as f:
                pools = json.load(f)

            pool_stats = []

            for key, values in sorted(pools.items()):
                try:
                    track, condition_name, direction = key.split("|")
                except ValueError:
                    continue

                cleaned = deduplicate_and_filter_instructions(
                    values,
                    direction=direction,
                    track=track,
                )

                pools[key] = cleaned

                pool_stats.append({
                    "key": key,
                    "requested": "loaded",
                    "raw_returned": len(values),
                    "valid_unique": len(cleaned),
                    "fallback_used": 0,
                })

            return pools, pool_stats

        semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

        tasks = []

        for condition_name in CONDITION_META.keys():
            for direction in DIRECTIONS:
                for track in TRACKS:
                    tasks.append(
                        generate_one_instruction_pool(
                            semaphore=semaphore,
                            condition_name=condition_name,
                            direction=direction,
                            track=track,
                        )
                    )

        results = await asyncio.gather(*tasks)

        pools = {}
        pool_stats = []

        for key, instructions, stats in results:
            if len(instructions) == 0:
                raise RuntimeError(f"Instruction pool is empty after filtering: {key}")

            pools[key] = instructions
            pool_stats.append(stats)

        with open(INSTRUCTION_POOL_JSON, "w", encoding="utf-8") as f:
            json.dump(pools, f, ensure_ascii=False, indent=2)

        print(f"[Info] Saved instruction pool to: {INSTRUCTION_POOL_JSON}")

        return pools, pool_stats


    class InstructionSampler:
        def __init__(self, pools: Dict[str, List[str]], seed: int = 42):
            self.pools = {}
            self.indices = {}
            self.rng = random.Random(seed)

            for key, values in pools.items():
                values = list(values)

                if len(values) == 0:
                    raise RuntimeError(f"Empty instruction pool: {key}")

                self.rng.shuffle(values)
                self.pools[key] = values
                self.indices[key] = 0

        def sample(self, key: str) -> str:
            if key not in self.pools:
                raise KeyError(f"Missing instruction pool: {key}")

            idx = self.indices[key]

            if idx >= len(self.pools[key]):
                self.rng.shuffle(self.pools[key])
                idx = 0

            instruction = self.pools[key][idx]
            self.indices[key] = idx + 1
            return instruction


    # Reference selection

    def build_reference_index(pairs: List[dict]) -> Dict[Tuple[str, str, str], List[str]]:
        index = defaultdict(list)

        for p in pairs:
            condition_name = p["condition_name"]
            split = p["split"]

            index[(condition_name, split, "clear")].append(p["clear_path"])
            index[(condition_name, split, "hazy")].append(p["hazy_path"])

        return index


    def choose_reference(
        ref_index: Dict[Tuple[str, str, str], List[str]],
        condition_name: str,
        split: str,
        target_domain: str,
        target_path: str,
        source_path: str,
        used_images: Optional[set] = None,
    ) -> str:
        candidates = ref_index.get((condition_name, split, target_domain), [])

        candidates = [
            x for x in candidates
            if x != target_path and x != source_path
        ]

        if used_images is not None:
            candidates = [x for x in candidates if x not in used_images]

        if not candidates:
            return ""

        return random.choice(candidates)


    def get_direction_fields(p: dict, direction: str) -> Tuple[str, str, str, str]:
        condition_name = p["condition_name"]
        meta = CONDITION_META[condition_name]

        if direction == "add_fog":
            source_path = p["clear_path"]
            target_path = p["hazy_path"]
            subtask = meta["add_subtask"]
            target_domain = "hazy"

        elif direction == "remove_fog":
            source_path = p["hazy_path"]
            target_path = p["clear_path"]
            subtask = meta["remove_subtask"]
            target_domain = "clear"

        else:
            raise ValueError(f"Unknown direction: {direction}")

        return source_path, target_path, subtask, target_domain


    def make_record(
        p: dict,
        direction: str,
        track: str,
        ref_index: Dict[Tuple[str, str, str], List[str]],
        sampler: InstructionSampler,
        used_images: Optional[set] = None,
    ) -> Optional[dict]:
        condition_name = p["condition_name"]
        split = p["split"]
        pair_key = safe_name(p["pair_key"])

        source_path, target_path, subtask, target_domain = get_direction_fields(p, direction)

        if used_images is not None:
            if source_path in used_images or target_path in used_images:
                return None

        reference_path = ""

        if track == "image-referenced":
            reference_path = choose_reference(
                ref_index=ref_index,
                condition_name=condition_name,
                split=split,
                target_domain=target_domain,
                target_path=target_path,
                source_path=source_path,
                used_images=used_images,
            )

            if reference_path == "":
                return None

        pair_id = f"RRSHID_{condition_name}_{split}_{pair_key}_{subtask}_{track}"
        pair_id = safe_name(pair_id)

        source_image = copy_image(
            src_path=source_path,
            dst_dir=SOURCE_DIR,
        )

        target_image = copy_image(
            src_path=target_path,
            dst_dir=TARGET_DIR,
        )

        if track == "image-referenced":
            reference_image = copy_image(
                src_path=reference_path,
                dst_dir=REFERENCE_DIR,
            )
        else:
            reference_image = ""

        pool_key = f"{track}|{condition_name}|{direction}"
        instruction = sampler.sample(pool_key)

        if not is_valid_instruction_for_atmosphere(
            instruction,
            direction=direction,
            track=track,
        ):
            raise RuntimeError(f"Invalid sampled instruction: {instruction}")

        if used_images is not None:
            used_images.add(source_path)
            used_images.add(target_path)

            if reference_path:
                used_images.add(reference_path)

        return {
            "pair_id": pair_id,
            "source_dataset": SOURCE_DATASET,
            "split": split,
            "task": TASK_NAME,
            "subtask": subtask,
            "track": track,
            "source_image": source_image,
            "target_image": target_image,
            "reference_image": reference_image,
            "instruction": instruction,
        }


    # Build JSON

    def split_pairs_by_split(pairs: List[dict]) -> Dict[str, List[dict]]:
        split_to_pairs = defaultdict(list)
        for p in pairs:
            split_to_pairs[p["split"]].append(p)
        return split_to_pairs


    def append_record_to_track_lists(record: dict, text_only_records: List[dict], image_referenced_records: List[dict]):
        if record["track"] == "text_only":
            text_only_records.append(record)
        elif record["track"] == "image-referenced":
            image_referenced_records.append(record)
        else:
            raise RuntimeError(f"Unknown track: {record['track']}")


    def build_records_train_relaxed(
        train_pairs: List[dict],
        ref_index: Dict[Tuple[str, str, str], List[str]],
        sampler: InstructionSampler,
    ) -> Tuple[List[dict], List[dict], dict]:
        """
        Train split:
        - allow original images to repeat across directions and tracks
        - allow a reference image to be reused later as source/target
        - still prevent reference == current source/target inside the same sample
        """
        text_only_records = []
        image_referenced_records = []

        pairs = list(train_pairs)
        random.shuffle(pairs)

        skipped_image_referenced = 0

        for p in tqdm(pairs, desc="Building train JSON with relaxed reuse"):
            for direction in DIRECTIONS:
                for track in TRACKS:
                    record = make_record(
                        p=p,
                        direction=direction,
                        track=track,
                        ref_index=ref_index,
                        sampler=sampler,
                        used_images=None,
                    )

                    if record is None:
                        if track == "image-referenced":
                            skipped_image_referenced += 1
                        continue

                    append_record_to_track_lists(record, text_only_records, image_referenced_records)

        stats = {
            "split": "train",
            "mode": "relaxed_reuse",
            "input_pairs": len(train_pairs),
            "text_only_records": len(text_only_records),
            "image_referenced_records": len(image_referenced_records),
            "skipped_image_referenced": skipped_image_referenced,
        }

        return text_only_records, image_referenced_records, stats


    def build_records_strict_unique_for_split(
        split_pairs: List[dict],
        split_name: str,
        ref_index: Dict[Tuple[str, str, str], List[str]],
        sampler: InstructionSampler,
    ) -> Tuple[List[dict], List[dict], set, dict]:
        """
        Val/Test split:
        - strict image-level unique inside this split
        - source/target/reference cannot repeat inside this split
        - each input pair generates at most one record
        """
        text_only_records = []
        image_referenced_records = []
        used_images = set()

        combos = [(direction, track) for direction in DIRECTIONS for track in TRACKS]

        pairs = list(split_pairs)
        random.shuffle(pairs)

        skipped_no_valid_combo = 0

        for i, p in enumerate(tqdm(pairs, desc=f"Building {split_name} JSON with strict unique")):
            rotated = combos[i % len(combos):] + combos[:i % len(combos)]

            added = False

            for direction, track in rotated:
                record = make_record(
                    p=p,
                    direction=direction,
                    track=track,
                    ref_index=ref_index,
                    sampler=sampler,
                    used_images=used_images,
                )

                if record is not None:
                    append_record_to_track_lists(record, text_only_records, image_referenced_records)
                    added = True
                    break

            if not added:
                skipped_no_valid_combo += 1

        stats = {
            "split": split_name,
            "mode": "strict_unique",
            "input_pairs": len(split_pairs),
            "text_only_records": len(text_only_records),
            "image_referenced_records": len(image_referenced_records),
            "unique_original_images_used": len(used_images),
            "skipped_no_valid_combo": skipped_no_valid_combo,
        }

        return text_only_records, image_referenced_records, used_images, stats


    def build_records_mixed_policy(
        pairs: List[dict],
        pools: Dict[str, List[str]],
    ) -> Tuple[List[dict], List[dict], Dict[str, set], List[dict]]:
        """
        Mixed policy:
        - train: relaxed reuse
        - val/test: strict unique
        """
        ref_index = build_reference_index(pairs)
        sampler = InstructionSampler(pools, seed=SEED)

        split_to_pairs = split_pairs_by_split(pairs)

        all_text_only_records = []
        all_image_referenced_records = []
        strict_used_images_by_split = {}
        build_stats = []

        train_pairs = split_to_pairs.get("train", [])
        train_text, train_img, train_stats = build_records_train_relaxed(
            train_pairs=train_pairs,
            ref_index=ref_index,
            sampler=sampler,
        )
        all_text_only_records.extend(train_text)
        all_image_referenced_records.extend(train_img)
        build_stats.append(train_stats)

        for split_name in ["val", "test"]:
            split_pairs = split_to_pairs.get(split_name, [])

            text_records, img_records, used_images, stats = build_records_strict_unique_for_split(
                split_pairs=split_pairs,
                split_name=split_name,
                ref_index=ref_index,
                sampler=sampler,
            )

            all_text_only_records.extend(text_records)
            all_image_referenced_records.extend(img_records)
            strict_used_images_by_split[split_name] = used_images
            build_stats.append(stats)

        return all_text_only_records, all_image_referenced_records, strict_used_images_by_split, build_stats


    # Validation and statistics

    def validate_one_record(record: dict):
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

        if list(record.keys()) != required_keys:
            raise RuntimeError(f"Invalid JSON keys in record: {record.get('pair_id', 'UNKNOWN')}")

        if record["task"] != TASK_NAME:
            raise RuntimeError(f"Invalid task name: {record['task']}")


    def validate_track_records(records: List[dict], expected_track: str):
        pair_ids = [r["pair_id"] for r in records]
        duplicate_pair_ids = [k for k, v in Counter(pair_ids).items() if v > 1]

        if duplicate_pair_ids:
            raise RuntimeError(f"Duplicate pair_id found in {expected_track}: {duplicate_pair_ids[:10]}")

        for r in records:
            validate_one_record(r)

            if r["track"] != expected_track:
                raise RuntimeError(f"Track mismatch: expected {expected_track}, got {r['track']}")

            if expected_track == "text_only":
                if r["reference_image"] != "":
                    raise RuntimeError(f"text_only record has non-empty reference_image: {r['pair_id']}")

            elif expected_track == "image-referenced":
                if r["reference_image"] == "":
                    raise RuntimeError(f"image-referenced record has empty reference_image: {r['pair_id']}")

            else:
                raise RuntimeError(f"Unknown expected track: {expected_track}")

            direction = None
            if r["subtask"].startswith("clear_to"):
                direction = "add_fog"
            elif r["subtask"].endswith("_to_clear"):
                direction = "remove_fog"

            if direction is not None:
                if not is_valid_instruction_for_atmosphere(
                    r["instruction"],
                    direction=direction,
                    track=expected_track,
                ):
                    raise RuntimeError(f"Invalid instruction in record {r['pair_id']}: {r['instruction']}")

            # 检查输出文件存在
            for key in ["source_image", "target_image", "reference_image"]:
                if r[key]:
                    abs_path = os.path.join(OUTPUT_ROOT, r[key])
                    if not os.path.exists(abs_path):
                        raise RuntimeError(f"Missing output image file: {abs_path}")


    def validate_eval_strict_unique_from_records(records: List[dict]):
        """
        Validate output paths are not duplicated within val/test records.
        Since val/test are strict unique, duplicated output paths would indicate a bug.
        """
        eval_records = [r for r in records if r["split"] in STRICT_UNIQUE_SPLITS]
        paths = []

        for r in eval_records:
            paths.append(r["source_image"])
            paths.append(r["target_image"])
            if r["reference_image"]:
                paths.append(r["reference_image"])

        duplicates = [k for k, v in Counter(paths).items() if v > 1]
        if duplicates:
            raise RuntimeError(f"Duplicate val/test output image paths found: {duplicates[:10]}")


    def print_instruction_pool_stats(pool_stats: List[dict]):
        print("\n========== Instruction Pool Statistics ==========")
        print(
            f"Each pool requests {REQUESTS_PER_TASK} × {INSTRUCTIONS_PER_REQUEST} "
            f"= {REQUESTS_PER_TASK * INSTRUCTIONS_PER_REQUEST} raw instructions."
        )

        for s in sorted(pool_stats, key=lambda x: x["key"]):
            print(
                f"{s['key']}: "
                f"requested={s['requested']}, "
                f"raw_returned={s['raw_returned']}, "
                f"valid_unique={s['valid_unique']}, "
                f"fallback_used={s['fallback_used']}"
            )


    def print_track_stats(records: List[dict], title: str):
        print(f"\n========== {title} ==========")
        print(f"Total records: {len(records)}")

        for field in ["split", "task", "subtask", "track"]:
            print(f"\n[{field}]")
            counter = Counter(r[field] for r in records)

            for k, v in sorted(counter.items()):
                print(f"{k}: {v}")


    def print_build_policy_stats(build_stats: List[dict], strict_used_images_by_split: Dict[str, set]):
        print("\n========== Build Policy Statistics ==========")
        print(f"TRAIN_ALLOW_REUSE: {TRAIN_ALLOW_REUSE}")
        print(f"STRICT_UNIQUE_SPLITS: {sorted(list(STRICT_UNIQUE_SPLITS))}")

        for s in build_stats:
            print(
                f"{s['split']}: mode={s['mode']}, "
                f"input_pairs={s['input_pairs']}, "
                f"text_only_records={s['text_only_records']}, "
                f"image_referenced_records={s['image_referenced_records']}"
                + (
                    f", unique_original_images_used={s.get('unique_original_images_used')}, "
                    f"skipped_no_valid_combo={s.get('skipped_no_valid_combo')}"
                    if s["mode"] == "strict_unique"
                    else f", skipped_image_referenced={s.get('skipped_image_referenced')}"
                )
            )

        total_strict_used = sum(len(v) for v in strict_used_images_by_split.values())
        print(f"val/test unique original images used across source/target/reference: {total_strict_used}")


    def print_global_stats(
        text_only_records: List[dict],
        image_referenced_records: List[dict],
        strict_used_images_by_split: Dict[str, set],
        build_stats: List[dict],
        pool_stats: List[dict],
    ):
        print("\n========== Global Statistics ==========")
        print(f"Text-only records: {len(text_only_records)}")
        print(f"Image-referenced records: {len(image_referenced_records)}")
        print(f"Total records: {len(text_only_records) + len(image_referenced_records)}")

        print_build_policy_stats(build_stats, strict_used_images_by_split)
        print_instruction_pool_stats(pool_stats)
        print_track_stats(text_only_records, "Text-Only JSON Statistics")
        print_track_stats(image_referenced_records, "Image-Referenced JSON Statistics")


    # Main

    async def main():
        random.seed(SEED)
        ensure_dirs()

        print(f"[Info] DATA_ROOT: {DATA_ROOT}")
        print(f"[Info] OUTPUT_ROOT: {OUTPUT_ROOT}")
        print(f"[Info] SOURCE_DIR: {SOURCE_DIR}")
        print(f"[Info] TARGET_DIR: {TARGET_DIR}")
        print(f"[Info] REFERENCE_DIR: {REFERENCE_DIR}")
        print(f"[Info] OUTPUT_TEXT_ONLY_JSON: {OUTPUT_TEXT_ONLY_JSON}")
        print(f"[Info] OUTPUT_IMAGE_REFERENCED_JSON: {OUTPUT_IMAGE_REFERENCED_JSON}")
        print(f"[Info] INSTRUCTION_POOL_JSON: {INSTRUCTION_POOL_JSON}")
        print(f"[Info] MODEL_NAME: {MODEL_NAME}")
        print(f"[Info] CONCURRENCY_LIMIT: {CONCURRENCY_LIMIT}")
        print(f"[Info] TASK_NAME: {TASK_NAME}")
        print(f"[Info] TRACKS: {TRACKS}")
        print(f"[Info] DIRECTIONS: {DIRECTIONS}")
        print(f"[Info] TRAIN_ALLOW_REUSE: {TRAIN_ALLOW_REUSE}")
        print(f"[Info] STRICT_UNIQUE_SPLITS: {sorted(list(STRICT_UNIQUE_SPLITS))}")

        pairs = discover_pairs()

        if not pairs:
            raise RuntimeError("No valid clear/hazy pairs found. Please check DATA_ROOT and RRSHID structure.")

        print(f"[Info] Found {len(pairs)} clear/hazy pairs.")

        pools, pool_stats = await generate_instruction_pools()

        text_only_records, image_referenced_records, strict_used_images_by_split, build_stats = build_records_mixed_policy(
            pairs=pairs,
            pools=pools,
        )

        validate_track_records(text_only_records, "text_only")
        validate_track_records(image_referenced_records, "image-referenced")
        validate_eval_strict_unique_from_records(text_only_records + image_referenced_records)

        with open(OUTPUT_TEXT_ONLY_JSON, "w", encoding="utf-8") as f:
            json.dump(text_only_records, f, ensure_ascii=False, indent=2)

        with open(OUTPUT_IMAGE_REFERENCED_JSON, "w", encoding="utf-8") as f:
            json.dump(image_referenced_records, f, ensure_ascii=False, indent=2)

        print_global_stats(
            text_only_records=text_only_records,
            image_referenced_records=image_referenced_records,
            strict_used_images_by_split=strict_used_images_by_split,
            build_stats=build_stats,
            pool_stats=pool_stats,
        )

        print(f"\n[Done] Saved text-only JSON to: {OUTPUT_TEXT_ONLY_JSON}")
        print(f"[Done] Saved image-referenced JSON to: {OUTPUT_IMAGE_REFERENCED_JSON}")

    asyncio.run(main())


def run_cloud() -> None:
    """Build SEN12MS-CR cloud records and append them to atmosphere JSON files."""
    __name__ = "rs_uniedit.atmosphere_cloud"
    import os
    import re
    import json
    import random
    import asyncio
    from pathlib import Path
    from collections import Counter, defaultdict
    from typing import Dict, List, Tuple, Optional

    import numpy as np
    from PIL import Image
    from tqdm import tqdm
    from openai import AsyncOpenAI

    try:
        import rasterio as rio
    except Exception as e:
        raise ImportError(
            "This script requires rasterio to correctly read SEN12MS-CR multi-band .tif files. "
            "Please install it with: pip install rasterio"
        ) from e


    # Fixed paths

    SEN12MS_ROOT = os.environ.get("SEN12MS_ROOT", "data/SEN12MS")

    OUTPUT_ROOT = "./RS-OmniEdit-Atmosphere"
    SOURCE_DIR = os.path.join(OUTPUT_ROOT, "source")
    TARGET_DIR = os.path.join(OUTPUT_ROOT, "target")
    REFERENCE_DIR = os.path.join(OUTPUT_ROOT, "reference")

    OUTPUT_TEXT_ONLY_JSON = os.path.join(OUTPUT_ROOT, "data_text_only.json")
    OUTPUT_IMAGE_REFERENCED_JSON = os.path.join(OUTPUT_ROOT, "data_image_referenced.json")
    INSTRUCTION_POOL_JSON = os.path.join(OUTPUT_ROOT, "instruction_pools_cloud.json")

    CONCURRENCY_LIMIT = 32

    OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")

    client = AsyncOpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
    )

    MODEL_NAME = os.environ.get("RS_UNIEDIT_LLM_MODEL", "Qwen/Qwen3.6-35B-A3B")


    # Generation config

    SEED = 42

    # 已有 instruction_pools_cloud.json 后可以改成 False
    REGEN_POOL = True

    # 最终 cloud 数据量
    TOTAL_TEXT_ONLY_RECORDS = 10000
    TOTAL_IMAGE_REFERENCED_RECORDS = 10000

    # 测试集大小
    TEST_TEXT_ONLY_RECORDS = 100
    TEST_IMAGE_REFERENCED_RECORDS = 100

    # train 自动由 total - test 得到
    TRAIN_TEXT_ONLY_RECORDS = TOTAL_TEXT_ONLY_RECORDS - TEST_TEXT_ONLY_RECORDS
    TRAIN_IMAGE_REFERENCED_RECORDS = TOTAL_IMAGE_REFERENCED_RECORDS - TEST_IMAGE_REFERENCED_RECORDS

    TRACKS = ["text_only", "image-referenced"]
    DIRECTIONS = ["add_cloud", "remove_cloud"]

    # 指令池设置
    INSTRUCTIONS_PER_REQUEST = 50
    REQUESTS_PER_TASK = 4

    COPY_IMAGES = True
    USE_ABSOLUTE_OUTPUT_PATH = False

    # 只删除 source/target/reference 下 SEN12MS_*.png，不影响 RRSHID 的雾任务图像
    CLEAR_EXISTING_SEN12MS_OUTPUT_IMAGES = True

    # season 抽样时尽量均衡
    BALANCE_SEASONS = True

    # test 池预留得大一点，方便 strict unique + reference 选择
    TEST_PAIR_POOL_SIZE = 2000


    # Sentinel-2 RGB config

    # SEN12MS/SEN12MS-CR S2 band order:
    # 0:B01, 1:B02, 2:B03, 3:B04
    S2_B02 = 1
    S2_B03 = 2
    S2_B04 = 3

    RGB_STRETCH_LOW_PERCENTILE = 2
    RGB_STRETCH_HIGH_PERCENTILE = 98
    RGB_GAMMA = 1 / 2.2


    # Constants

    TASK_NAME = "Atmosphere"
    SOURCE_DATASET = "SEN12MS-CR"

    SUBTASK_ADD_CLOUD = "clear_to_cloudy"
    SUBTASK_REMOVE_CLOUD = "cloudy_to_clear"

    IMG_EXTS = {".tif", ".tiff"}

    SEASON_CONFIGS = {
        "summer": {
            "clean_dir": "ROIs1868_summer_s2",
            "cloudy_dir": "ROIs1868_summer_s2_cloudy",
        },
        "winter": {
            "clean_dir": "ROIs2017_winter_s2",
            "cloudy_dir": "ROIs2017_winter_s2_cloudy",
        },
    }

    OUTPUT_FILE_REGISTRY: Dict[str, str] = {}


    # Basic utilities

    def ensure_dirs():
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
        os.makedirs(SOURCE_DIR, exist_ok=True)
        os.makedirs(TARGET_DIR, exist_ok=True)
        os.makedirs(REFERENCE_DIR, exist_ok=True)

        if CLEAR_EXISTING_SEN12MS_OUTPUT_IMAGES:
            for d in [SOURCE_DIR, TARGET_DIR, REFERENCE_DIR]:
                for p in Path(d).glob("SEN12MS_*.png"):
                    p.unlink()


    def safe_name(text: str) -> str:
        text = str(text)
        text = re.sub(r"[^\w\-\.]+", "_", text)
        text = re.sub(r"_+", "_", text)
        return text.strip("_")


    def output_path_for_json(path: str) -> str:
        if USE_ABSOLUTE_OUTPUT_PATH:
            return os.path.abspath(path)
        return os.path.relpath(path, OUTPUT_ROOT).replace("\\", "/")


    def list_images(folder: Path) -> List[Path]:
        if not folder.exists():
            return []
        return sorted([
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ])


    def normalize_instruction_for_dedup(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        text = text.strip(" .,!?:;\"'")
        return text


    def contains_any(text: str, keywords: List[str]) -> bool:
        return any(k in text for k in keywords)


    def make_sample_id(idx: int) -> str:
        return f"{idx:06d}"


    def split_half(n: int) -> Tuple[int, int]:
        a = n // 2
        b = n - a
        return a, b


    # Correct Sentinel-2 RGB reading

    def stretch_rgb_true_color(
        rgb: np.ndarray,
        low: float = RGB_STRETCH_LOW_PERCENTILE,
        high: float = RGB_STRETCH_HIGH_PERCENTILE,
        gamma: float = RGB_GAMMA,
    ) -> np.ndarray:
        """
        Convert Sentinel-2 RGB reflectance to uint8 visualization.

        Input:
          rgb: [H, W, 3], ordered as B04/B03/B02.

        Uses a joint percentile stretch over all RGB channels to reduce color shifts.
        """
        rgb = rgb.astype(np.float32)
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)

        valid_mask = np.isfinite(rgb) & (rgb > 0)
        valid_values = rgb[valid_mask]

        if valid_values.size == 0:
            return np.zeros(rgb.shape, dtype=np.uint8)

        p_low, p_high = np.percentile(valid_values, (low, high))

        if not np.isfinite(p_low) or not np.isfinite(p_high) or p_high <= p_low:
            return np.zeros(rgb.shape, dtype=np.uint8)

        x = (rgb - p_low) / (p_high - p_low)
        x = np.clip(x, 0.0, 1.0)
        x[~np.isfinite(x)] = 0.0

        if gamma is not None and gamma > 0:
            x = np.power(x, gamma)

        all_invalid = ~np.any(valid_mask, axis=2)
        x[all_invalid] = 0.0

        return (x * 255.0).astype(np.uint8)


    def get_rgb_image_from_s2_tif(path: str) -> Image.Image:
        """
        Read SEN12MS-CR Sentinel-2 .tif and return true-color RGB PIL image.

        rasterio.read() returns:
          s2_data: [C, H, W]

        True color:
          R = B04 = index 3
          G = B03 = index 2
          B = B02 = index 1
        """
        with rio.open(path) as src:
            s2_data = src.read()

        if s2_data.ndim != 3:
            raise ValueError(f"Expected S2 data shape [C,H,W], got {s2_data.shape}, path={path}")

        if s2_data.shape[0] <= S2_B04:
            raise ValueError(f"S2 data has too few bands for RGB: {s2_data.shape}, path={path}")

        rgb = np.stack(
            [
                s2_data[S2_B04],  # Red = B04
                s2_data[S2_B03],  # Green = B03
                s2_data[S2_B02],  # Blue = B02
            ],
            axis=2,
        )

        rgb_uint8 = stretch_rgb_true_color(rgb)
        return Image.fromarray(rgb_uint8)


    def save_rgb_png(
        src_path: str,
        dst_dir: str,
        season: str,
        domain: str,
        sample_id: str,
    ) -> str:
        """
        Save as:
          SEN12MS_summer_clear_000001.png
          SEN12MS_summer_cloudy_000001.png
          SEN12MS_winter_clear_000001.png
          SEN12MS_winter_cloudy_000001.png
        """
        src = Path(src_path).resolve()
        src_abs = str(src)

        stem = safe_name(f"SEN12MS_{season}_{domain}_{sample_id}")
        dst_path = Path(dst_dir) / f"{stem}.png"
        dst_abs = str(dst_path.resolve())

        if dst_abs in OUTPUT_FILE_REGISTRY:
            if OUTPUT_FILE_REGISTRY[dst_abs] == src_abs:
                return output_path_for_json(str(dst_path))

            idx = 2
            while True:
                dst_path_2 = Path(dst_dir) / f"{stem}_{idx}.png"
                dst_abs_2 = str(dst_path_2.resolve())

                if dst_abs_2 not in OUTPUT_FILE_REGISTRY:
                    OUTPUT_FILE_REGISTRY[dst_abs_2] = src_abs
                    if COPY_IMAGES:
                        img = get_rgb_image_from_s2_tif(src_abs)
                        img.save(str(dst_path_2))
                    return output_path_for_json(str(dst_path_2))

                if OUTPUT_FILE_REGISTRY[dst_abs_2] == src_abs:
                    return output_path_for_json(str(dst_path_2))

                idx += 1

        OUTPUT_FILE_REGISTRY[dst_abs] = src_abs

        if COPY_IMAGES and not dst_path.exists():
            img = get_rgb_image_from_s2_tif(src_abs)
            img.save(str(dst_path))

        return output_path_for_json(str(dst_path))


    # SEN12MS pair discovery

    def normalize_key_from_relative_path(rel_path: Path) -> str:
        """
        Make clean S2 and cloudy S2 file keys match.

        Examples:
          xxx_s2_001.tif        -> xxx_s2_001
          xxx_s2_cloudy_001.tif -> xxx_s2_001
        """
        s = str(rel_path.with_suffix("")).replace("\\", "/").lower()

        s = s.replace("s2_cloudy", "s2")
        s = s.replace("s2-cloudy", "s2")
        s = s.replace("cloudy_s2", "s2")
        s = s.replace("cloudy-s2", "s2")

        s = re.sub(r"[_\-]+cloudy", "", s)
        s = re.sub(r"cloudy[_\-]+", "", s)

        s = re.sub(r"/+", "/", s)
        s = re.sub(r"\s+", "_", s)

        return s


    def build_file_map(base_dir: Path) -> Dict[str, Path]:
        images = list_images(base_dir)
        result = {}

        for p in images:
            rel = p.relative_to(base_dir)
            key = normalize_key_from_relative_path(rel)
            if key not in result:
                result[key] = p

        return result


    def discover_sen12ms_pairs() -> List[dict]:
        root = Path(SEN12MS_ROOT).expanduser().resolve()
        all_pairs = []

        for season, cfg in SEASON_CONFIGS.items():
            clean_dir = root / cfg["clean_dir"]
            cloudy_dir = root / cfg["cloudy_dir"]

            if not clean_dir.exists():
                print(f"[Warning] Missing clean dir for {season}: {clean_dir}")
                continue

            if not cloudy_dir.exists():
                print(f"[Warning] Missing cloudy dir for {season}: {cloudy_dir}")
                continue

            clean_map = build_file_map(clean_dir)
            cloudy_map = build_file_map(cloudy_dir)

            common_keys = sorted(set(clean_map.keys()) & set(cloudy_map.keys()))

            print(
                f"[Info] {season}: clean={len(clean_map)}, cloudy={len(cloudy_map)}, "
                f"matched_pairs={len(common_keys)}"
            )

            for key in common_keys:
                all_pairs.append({
                    "season": season,
                    "pair_key": safe_name(key),
                    "clean_path": str(clean_map[key]),
                    "cloudy_path": str(cloudy_map[key]),
                })

        all_pairs = sorted(all_pairs, key=lambda x: (x["season"], x["pair_key"]))

        for idx, p in enumerate(all_pairs, start=1):
            p["sample_id"] = make_sample_id(idx)

        return all_pairs


    # Sampling pools

    def sample_balanced_by_season(
        pairs: List[dict],
        n: int,
        rng: random.Random,
    ) -> Tuple[List[dict], List[dict]]:
        pairs = list(pairs)

        if not BALANCE_SEASONS:
            rng.shuffle(pairs)
            return pairs[:n], pairs[n:]

        buckets = defaultdict(list)
        for p in pairs:
            buckets[p["season"]].append(p)

        for season in buckets:
            rng.shuffle(buckets[season])

        seasons = sorted(buckets.keys())
        selected = []

        base = n // max(len(seasons), 1)
        remainder = n % max(len(seasons), 1)

        for i, season in enumerate(seasons):
            target = base + (1 if i < remainder else 0)
            take = min(target, len(buckets[season]))
            selected.extend(buckets[season][:take])
            buckets[season] = buckets[season][take:]

        remaining = []
        for season in seasons:
            remaining.extend(buckets[season])

        if len(selected) < n:
            rng.shuffle(remaining)
            need = n - len(selected)
            selected.extend(remaining[:need])
            remaining = remaining[need:]

        rng.shuffle(selected)
        rng.shuffle(remaining)

        return selected, remaining


    def split_train_test_pair_pools(all_pairs: List[dict]) -> Tuple[List[dict], List[dict]]:
        """
        Select a test pair pool first (balanced by season), and use the rest as train pool.
        """
        rng = random.Random(SEED)
        pairs = list(all_pairs)
        rng.shuffle(pairs)

        test_pool, remaining = sample_balanced_by_season(
            pairs,
            min(TEST_PAIR_POOL_SIZE, len(pairs)),
            rng,
        )

        train_pool = remaining

        print(f"[Info] test pair pool size: {len(test_pool)}")
        print(f"[Info] train pair pool size: {len(train_pool)}")

        return train_pool, test_pool


    def assign_split_to_pairs(pairs: List[dict], split_name: str) -> List[dict]:
        out = []
        for p in pairs:
            q = dict(p)
            q["split"] = split_name
            out.append(q)
        return out


    # Cloud instruction validation

    def is_valid_instruction_for_cloud(
        instruction: str,
        direction: str,
        track: str,
    ) -> bool:
        if not isinstance(instruction, str):
            return False

        ins = re.sub(r"\s+", " ", instruction.strip())
        if not ins:
            return False

        low = ins.lower()

        if len(ins) < 25 or len(ins) > 240:
            return False

        banned_phrases = [
            "core action",
            "checked",
            "avoid vague",
            "requirement",
            "requirements",
            "note:",
            "instruction:",
            "goal:",
            "output",
            "json",
            "markdown",
            "self-evaluation",
            "self evaluation",
            "as an ai",
            "i cannot",
            "here are",
            "valid instruction",
            "bad example",
            "good example",
        ]

        if any(x in low for x in banned_phrases):
            return False

        banned_chars = [":", "*", "/", "`", "{", "}", "[", "]"]
        if any(ch in ins for ch in banned_chars):
            return False

        if "?" in ins:
            return False

        banned_topics = [
            "haze",
            "hazy",
            "fog",
            "foggy",
            "dehaze",
            "mist",
            "misty",
            "season",
            "spring",
            "summer",
            "autumn",
            "fall",
            "winter",
            "resolution",
            "super-resolution",
            "super resolution",
            "upscale",
            "downsample",
            "viewpoint",
            "angle",
            "sensor",
            "sentinel",
            "landsat",
            "modis",
            "classification",
            "segmentation",
            "mask",
            "label map",
        ]

        if any(x in low for x in banned_topics):
            return False

        cloud_terms = [
            "cloud",
            "clouds",
            "cloudy",
            "cloud cover",
            "cloud-covered",
            "overcast",
        ]

        if not contains_any(low, cloud_terms):
            return False

        if direction == "add_cloud":
            add_verbs = [
                "add",
                "introduce",
                "apply",
                "render",
                "simulate",
                "make",
                "turn",
                "create",
                "give",
                "cover",
                "place",
            ]
            if not contains_any(low, add_verbs):
                return False

        elif direction == "remove_cloud":
            remove_verbs = [
                "remove",
                "clear",
                "restore",
                "recover",
                "reduce",
                "clean",
                "reveal",
            ]
            if not contains_any(low, remove_verbs):
                return False

        else:
            return False

        rs_terms = [
            "satellite",
            "aerial",
            "remote sensing",
            "overhead",
            "land-cover",
            "land cover",
            "roads",
            "buildings",
            "vegetation",
            "terrain",
            "water bodies",
            "geographic",
            "scene layout",
            "object positions",
            "ground scene",
        ]

        if not contains_any(low, rs_terms):
            return False

        preserve_terms = [
            "preserve",
            "keep",
            "maintain",
            "retain",
            "unchanged",
            "without changing",
            "without altering",
            "same",
        ]

        if not contains_any(low, preserve_terms):
            return False

        if track == "text_only":
            if "reference" in low:
                return False

        elif track == "image-referenced":
            if "reference" not in low:
                return False

        else:
            return False

        return True


    def deduplicate_and_filter_cloud_instructions(
        instructions: List[str],
        direction: str,
        track: str,
    ) -> List[str]:
        cleaned = []
        seen = set()

        for ins in instructions:
            ins = str(ins).strip()
            ins = re.sub(r"\s+", " ", ins)
            ins = ins.strip().strip('"').strip("'").strip(",")

            if not is_valid_instruction_for_cloud(ins, direction=direction, track=track):
                continue

            key = normalize_instruction_for_dedup(ins)

            if key in seen:
                continue

            cleaned.append(ins)
            seen.add(key)

        return cleaned


    # Cloud instruction pool generation

    def get_cloud_scenario(direction: str, track: str) -> str:
        if direction == "add_cloud":
            base = (
                "Add realistic cloud cover to a clear RGB remote sensing image while keeping "
                "the same land-cover layout, roads, buildings, vegetation, terrain, water bodies, "
                "ground scene, and object positions unchanged."
            )
        else:
            base = (
                "Remove cloud cover from a cloudy RGB remote sensing image and recover a clear view "
                "while keeping the same land-cover layout, roads, buildings, vegetation, terrain, "
                "water bodies, ground scene, and object positions unchanged."
            )

        if track == "image-referenced":
            base += (
                " The instruction should naturally mention using a reference image only as guidance "
                "for the target cloud or clear-sky appearance."
            )
        else:
            base += " The instruction should not mention any reference image."

        return base


    def get_cloud_track_requirement(track: str) -> str:
        if track == "text_only":
            return "Do not mention any reference image."
        if track == "image-referenced":
            return (
                "Mention the reference image naturally as guidance for the target cloud or clear-sky appearance, "
                "but make clear that the source scene layout should stay unchanged."
            )
        raise ValueError(f"Unknown track: {track}")


    def get_cloud_direction_requirement(direction: str) -> str:
        if direction == "add_cloud":
            return (
                "Every instruction must ask to add, introduce, apply, render, simulate, make, "
                "create, place, or cover the scene with clouds."
            )
        if direction == "remove_cloud":
            return (
                "Every instruction must ask to remove, clear, reduce, restore, recover, "
                "or reveal the ground scene from clouds."
            )
        raise ValueError(f"Unknown direction: {direction}")


    def fallback_cloud_instructions(direction: str, track: str) -> List[str]:
        if direction == "add_cloud":
            base = [
                "Add realistic clouds to this satellite image while keeping the land-cover layout unchanged.",
                "Introduce cloud cover into this aerial image without altering roads, buildings, or vegetation.",
                "Render this overhead scene with clouds while preserving the terrain and object positions.",
                "Apply cloud cover to this remote sensing image while maintaining the same geographic structure.",
                "Make this satellite scene appear cloudy while keeping the ground layout the same.",
            ]
        else:
            base = [
                "Remove the clouds from this satellite image while keeping the land-cover layout unchanged.",
                "Clear the cloud cover from this aerial image without altering roads, buildings, or vegetation.",
                "Restore a cloud-free view of this overhead scene while preserving the terrain and object positions.",
                "Recover the clear ground scene in this remote sensing image while maintaining the same geographic structure.",
                "Reduce the cloud cover in this satellite scene while keeping the ground layout the same.",
            ]

        if track == "image-referenced":
            base = [
                "Use the reference image as guidance for the target atmospheric appearance, and " + x[0].lower() + x[1:]
                for x in base
            ]

        return base


    async def build_cloud_instruction_pool_one_request(
        direction: str,
        track: str,
    ) -> List[str]:
        scenario = get_cloud_scenario(direction, track)
        track_requirement = get_cloud_track_requirement(track)
        direction_requirement = get_cloud_direction_requirement(direction)

        prompt = f"""You are a remote sensing image editing data construction assistant.

    Task: Generate a list of {INSTRUCTIONS_PER_REQUEST} diverse, natural, and imperative image editing instructions in English.

    Scenario: {scenario}

    Requirements:
    1. Use varied imperative expressions that sound like real user editing requests.
    2. The instructions must be direct editing commands.
    3. Each instruction must be one natural sentence.
    4. {direction_requirement}
    5. {track_requirement}
    6. Mention remote sensing concepts naturally, such as satellite image, aerial image, overhead view, land-cover layout, roads, buildings, vegetation, water bodies, terrain, geographic structure, ground scene, or object positions.
    7. The instruction must say that the underlying scene layout or object positions should be preserved.
    8. Do not use labels, headings, checklist language, explanations, or self-evaluation.
    9. Do not use phrases such as Core Action, Checked, Requirement, Note, Goal, or Instruction.
    10. Do not use slash alternatives. Choose one natural wording.
    11. Do not mention haze, fog, seasons, resolution changes, viewpoint changes, sensors, masks, segmentation maps, or classification maps.
    12. Output strictly in JSON format containing a list named instructions.

    Output format:
    {{
      "instructions": [
        "Remove the clouds from this satellite image while keeping the land-cover layout unchanged.",
        "Add realistic clouds to this aerial image without moving roads, buildings, or vegetation."
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
            instructions = result.get("instructions", [])

            if not isinstance(instructions, list):
                return []

            return [
                x.strip()
                for x in instructions
                if isinstance(x, str) and x.strip()
            ]

        except Exception:
            return []


    async def generate_one_cloud_instruction_pool(
        semaphore: asyncio.Semaphore,
        direction: str,
        track: str,
    ) -> Tuple[str, List[str], dict]:
        key = f"{track}|cloud|{direction}"
        print(f"[Info] Generating cloud instruction pool: {key}")

        async def single_request():
            async with semaphore:
                return await build_cloud_instruction_pool_one_request(
                    direction=direction,
                    track=track,
                )

        results = await asyncio.gather(
            *[single_request() for _ in range(REQUESTS_PER_TASK)]
        )

        merged = []
        for res in results:
            merged.extend(res)

        cleaned = deduplicate_and_filter_cloud_instructions(
            merged,
            direction=direction,
            track=track,
        )

        seen = set(normalize_instruction_for_dedup(x) for x in cleaned)
        fallback_used = 0

        for ins in fallback_cloud_instructions(direction, track):
            if not is_valid_instruction_for_cloud(ins, direction=direction, track=track):
                continue

            key_norm = normalize_instruction_for_dedup(ins)

            if key_norm not in seen:
                cleaned.append(ins)
                seen.add(key_norm)
                fallback_used += 1

        stats = {
            "key": key,
            "requested": INSTRUCTIONS_PER_REQUEST * REQUESTS_PER_TASK,
            "raw_returned": len(merged),
            "valid_unique": len(cleaned),
            "fallback_used": fallback_used,
        }

        return key, cleaned, stats


    async def generate_cloud_instruction_pools() -> Tuple[Dict[str, List[str]], List[dict]]:
        if os.path.exists(INSTRUCTION_POOL_JSON) and not REGEN_POOL:
            print(f"[Info] Loading existing cloud instruction pool: {INSTRUCTION_POOL_JSON}")

            with open(INSTRUCTION_POOL_JSON, "r", encoding="utf-8") as f:
                pools = json.load(f)

            pool_stats = []

            for key, values in sorted(pools.items()):
                try:
                    track, _, direction = key.split("|")
                except ValueError:
                    continue

                cleaned = deduplicate_and_filter_cloud_instructions(
                    values,
                    direction=direction,
                    track=track,
                )

                pools[key] = cleaned

                pool_stats.append({
                    "key": key,
                    "requested": "loaded",
                    "raw_returned": len(values),
                    "valid_unique": len(cleaned),
                    "fallback_used": 0,
                })

            return pools, pool_stats

        semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
        tasks = []

        for direction in DIRECTIONS:
            for track in TRACKS:
                tasks.append(
                    generate_one_cloud_instruction_pool(
                        semaphore=semaphore,
                        direction=direction,
                        track=track,
                    )
                )

        results = await asyncio.gather(*tasks)

        pools = {}
        pool_stats = []

        for key, instructions, stats in results:
            if len(instructions) == 0:
                raise RuntimeError(f"Cloud instruction pool is empty after filtering: {key}")

            pools[key] = instructions
            pool_stats.append(stats)

        with open(INSTRUCTION_POOL_JSON, "w", encoding="utf-8") as f:
            json.dump(pools, f, ensure_ascii=False, indent=2)

        print(f"[Info] Saved cloud instruction pool to: {INSTRUCTION_POOL_JSON}")

        return pools, pool_stats


    class InstructionSampler:
        def __init__(self, pools: Dict[str, List[str]], seed: int = 42):
            self.pools = {}
            self.indices = {}
            self.rng = random.Random(seed)

            for key, values in pools.items():
                values = list(values)

                if len(values) == 0:
                    raise RuntimeError(f"Empty instruction pool: {key}")

                self.rng.shuffle(values)
                self.pools[key] = values
                self.indices[key] = 0

        def sample(self, key: str) -> str:
            if key not in self.pools:
                raise KeyError(f"Missing instruction pool: {key}")

            idx = self.indices[key]

            if idx >= len(self.pools[key]):
                self.rng.shuffle(self.pools[key])
                idx = 0

            instruction = self.pools[key][idx]
            self.indices[key] = idx + 1
            return instruction


    # Reference and record construction

    def build_reference_index(pairs: List[dict]) -> Dict[Tuple[str, str, str], List[dict]]:
        index = defaultdict(list)

        for p in pairs:
            split = p["split"]
            season = p["season"]
            index[(split, season, "clear")].append(p)
            index[(split, season, "cloudy")].append(p)

        return index


    def choose_reference_pair(
        ref_index: Dict[Tuple[str, str, str], List[dict]],
        split: str,
        season: str,
        target_domain: str,
        current_sample_id: str,
        used_images: set,
    ) -> Optional[dict]:
        candidates = list(ref_index.get((split, season, target_domain), []))
        random.shuffle(candidates)

        for p in candidates:
            if p["sample_id"] == current_sample_id:
                continue

            candidate_path = p["clean_path"] if target_domain == "clear" else p["cloudy_path"]
            if candidate_path in used_images:
                continue

            return p

        return None


    def get_direction_fields(p: dict, direction: str) -> Tuple[str, str, str, str, str]:
        if direction == "add_cloud":
            return (
                p["clean_path"],
                p["cloudy_path"],
                "clear",
                "cloudy",
                SUBTASK_ADD_CLOUD,
            )

        if direction == "remove_cloud":
            return (
                p["cloudy_path"],
                p["clean_path"],
                "cloudy",
                "clear",
                SUBTASK_REMOVE_CLOUD,
            )

        raise ValueError(f"Unknown direction: {direction}")


    def make_cloud_record(
        p: dict,
        direction: str,
        track: str,
        ref_index: Dict[Tuple[str, str, str], List[dict]],
        sampler: InstructionSampler,
        used_images: set,
    ) -> Optional[dict]:
        source_path, target_path, source_domain, target_domain, subtask = get_direction_fields(p, direction)

        # strict unique for all splits
        if source_path in used_images or target_path in used_images:
            return None

        reference_path = ""
        reference_sample_id = ""

        if track == "image-referenced":
            ref_pair = choose_reference_pair(
                ref_index=ref_index,
                split=p["split"],
                season=p["season"],
                target_domain=target_domain,
                current_sample_id=p["sample_id"],
                used_images=used_images,
            )

            if ref_pair is None:
                return None

            reference_sample_id = ref_pair["sample_id"]
            reference_path = ref_pair["clean_path"] if target_domain == "clear" else ref_pair["cloudy_path"]

            if reference_path == source_path or reference_path == target_path:
                return None

            if reference_path in used_images:
                return None

        pair_id = safe_name(f"SEN12MS_{p['split']}_{p['season']}_{p['sample_id']}_{subtask}_{track}")

        source_image = save_rgb_png(
            src_path=source_path,
            dst_dir=SOURCE_DIR,
            season=p["season"],
            domain=source_domain,
            sample_id=p["sample_id"],
        )

        target_image = save_rgb_png(
            src_path=target_path,
            dst_dir=TARGET_DIR,
            season=p["season"],
            domain=target_domain,
            sample_id=p["sample_id"],
        )

        if track == "image-referenced":
            reference_image = save_rgb_png(
                src_path=reference_path,
                dst_dir=REFERENCE_DIR,
                season=p["season"],
                domain=target_domain,
                sample_id=reference_sample_id,
            )
        else:
            reference_image = ""

        pool_key = f"{track}|cloud|{direction}"
        instruction = sampler.sample(pool_key)

        if not is_valid_instruction_for_cloud(
            instruction,
            direction=direction,
            track=track,
        ):
            raise RuntimeError(f"Invalid sampled cloud instruction: {instruction}")

        used_images.add(source_path)
        used_images.add(target_path)
        if reference_path:
            used_images.add(reference_path)

        return {
            "pair_id": pair_id,
            "source_dataset": SOURCE_DATASET,
            "split": p["split"],
            "task": TASK_NAME,
            "subtask": subtask,
            "track": track,
            "source_image": source_image,
            "target_image": target_image,
            "reference_image": reference_image,
            "instruction": instruction,
        }


    # Strict unique build logic

    def group_pairs_by_split_and_season(pairs: List[dict]) -> Dict[Tuple[str, str], List[dict]]:
        grouped = defaultdict(list)
        for p in pairs:
            grouped[(p["split"], p["season"])].append(p)

        for k in grouped:
            random.shuffle(grouped[k])

        return grouped


    def pop_next_unused_pair(
        queues: Dict[Tuple[str, str], List[dict]],
        queue_ptrs: Dict[Tuple[str, str], int],
        split_name: str,
        season: str,
        used_images: set,
    ) -> Optional[dict]:
        key = (split_name, season)
        q = queues.get(key, [])
        idx = queue_ptrs.get(key, 0)

        while idx < len(q):
            p = q[idx]
            idx += 1

            # a source-target pair requires both clean and cloudy unused
            if p["clean_path"] in used_images or p["cloudy_path"] in used_images:
                continue

            queue_ptrs[key] = idx
            return p

        queue_ptrs[key] = idx
        return None


    def build_records_for_combo(
        split_name: str,
        season: str,
        track: str,
        direction: str,
        target_n: int,
        queues: Dict[Tuple[str, str], List[dict]],
        queue_ptrs: Dict[Tuple[str, str], int],
        ref_index: Dict[Tuple[str, str, str], List[dict]],
        sampler: InstructionSampler,
        used_images: set,
    ) -> List[dict]:
        records = []

        desc = f"Building {split_name} | {season} | {track} | {direction}"
        pbar = tqdm(total=target_n, desc=desc, leave=False)

        while len(records) < target_n:
            p = pop_next_unused_pair(
                queues=queues,
                queue_ptrs=queue_ptrs,
                split_name=split_name,
                season=season,
                used_images=used_images,
            )

            if p is None:
                break

            rec = make_cloud_record(
                p=p,
                direction=direction,
                track=track,
                ref_index=ref_index,
                sampler=sampler,
                used_images=used_images,
            )

            if rec is not None:
                records.append(rec)
                pbar.update(1)

        pbar.close()

        if len(records) < target_n:
            print(
                f"[Warning] {split_name}|{season}|{track}|{direction}: "
                f"target={target_n}, actual={len(records)}"
            )

        return records


    def build_strict_cloud_records(
        all_pairs: List[dict],
        pools: Dict[str, List[str]],
    ) -> Tuple[List[dict], List[dict], set, List[dict]]:
        """
        Build all cloud records with strict unique image usage for both train and test.
        """
        train_pool_raw, test_pool_raw = split_train_test_pair_pools(all_pairs)

        train_pairs = assign_split_to_pairs(train_pool_raw, "train")
        test_pairs = assign_split_to_pairs(test_pool_raw, "test")

        all_selected_pairs = train_pairs + test_pairs
        ref_index = build_reference_index(all_selected_pairs)
        sampler = InstructionSampler(pools, seed=SEED)

        queues = group_pairs_by_split_and_season(all_selected_pairs)
        queue_ptrs = defaultdict(int)
        used_images = set()

        text_only_records = []
        image_referenced_records = []
        build_stats = []

        # quotas
        test_text_add, test_text_remove = split_half(TEST_TEXT_ONLY_RECORDS)
        test_img_add, test_img_remove = split_half(TEST_IMAGE_REFERENCED_RECORDS)

        train_text_add, train_text_remove = split_half(TRAIN_TEXT_ONLY_RECORDS)
        train_img_add, train_img_remove = split_half(TRAIN_IMAGE_REFERENCED_RECORDS)

        quota_table = [
            ("test",  "text_only",        "add_cloud",    test_text_add),
            ("test",  "text_only",        "remove_cloud", test_text_remove),
            ("test",  "image-referenced", "add_cloud",    test_img_add),
            ("test",  "image-referenced", "remove_cloud", test_img_remove),

            ("train", "text_only",        "add_cloud",    train_text_add),
            ("train", "text_only",        "remove_cloud", train_text_remove),
            ("train", "image-referenced", "add_cloud",    train_img_add),
            ("train", "image-referenced", "remove_cloud", train_img_remove),
        ]

        # test first, then train, so train cannot leak into test
        for split_name, track, direction, total_target in quota_table:
            season_a, season_b = split_half(total_target)

            season_targets = {
                "summer": season_a,
                "winter": season_b,
            }

            combo_records = []

            for season in ["summer", "winter"]:
                target_n = season_targets[season]
                recs = build_records_for_combo(
                    split_name=split_name,
                    season=season,
                    track=track,
                    direction=direction,
                    target_n=target_n,
                    queues=queues,
                    queue_ptrs=queue_ptrs,
                    ref_index=ref_index,
                    sampler=sampler,
                    used_images=used_images,
                )
                combo_records.extend(recs)

            if len(combo_records) != total_target:
                raise RuntimeError(
                    f"Failed to build exact quota for {split_name}|{track}|{direction}: "
                    f"target={total_target}, actual={len(combo_records)}"
                )

            if track == "text_only":
                text_only_records.extend(combo_records)
            else:
                image_referenced_records.extend(combo_records)

            build_stats.append({
                "split": split_name,
                "track": track,
                "direction": direction,
                "target": total_target,
                "actual": len(combo_records),
                "summer": season_targets["summer"],
                "winter": season_targets["winter"],
            })

        if len(text_only_records) != TOTAL_TEXT_ONLY_RECORDS:
            raise RuntimeError(
                f"text_only count mismatch: expected {TOTAL_TEXT_ONLY_RECORDS}, got {len(text_only_records)}"
            )

        if len(image_referenced_records) != TOTAL_IMAGE_REFERENCED_RECORDS:
            raise RuntimeError(
                f"image-referenced count mismatch: expected {TOTAL_IMAGE_REFERENCED_RECORDS}, got {len(image_referenced_records)}"
            )

        return text_only_records, image_referenced_records, used_images, build_stats


    # Append to existing fog JSON

    def load_json_list(path: str) -> List[dict]:
        if not os.path.exists(path):
            return []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise RuntimeError(f"Expected a list JSON: {path}")

        return data


    def remove_existing_sen12ms_records(records: List[dict]) -> List[dict]:
        remove_names = {"SEN12MS", "SEN12MS-CR"}
        return [r for r in records if r.get("source_dataset") not in remove_names]


    def append_cloud_to_existing_jsons(
        cloud_text_records: List[dict],
        cloud_image_records: List[dict],
    ) -> Tuple[List[dict], List[dict]]:
        old_text = load_json_list(OUTPUT_TEXT_ONLY_JSON)
        old_image = load_json_list(OUTPUT_IMAGE_REFERENCED_JSON)

        old_text = remove_existing_sen12ms_records(old_text)
        old_image = remove_existing_sen12ms_records(old_image)

        combined_text = old_text + cloud_text_records
        combined_image = old_image + cloud_image_records

        return combined_text, combined_image


    # Validation and statistics

    def validate_one_record(record: dict):
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

        if list(record.keys()) != required_keys:
            raise RuntimeError(f"Invalid JSON keys in record: {record.get('pair_id', 'UNKNOWN')}")

        if record["task"] != TASK_NAME:
            raise RuntimeError(f"Invalid task name: {record['task']}")


    def direction_from_subtask(subtask: str) -> Optional[str]:
        if subtask == SUBTASK_ADD_CLOUD:
            return "add_cloud"
        if subtask == SUBTASK_REMOVE_CLOUD:
            return "remove_cloud"
        return None


    def validate_track_records(records: List[dict], expected_track: str):
        pair_ids = [r["pair_id"] for r in records]
        duplicate_pair_ids = [k for k, v in Counter(pair_ids).items() if v > 1]

        if duplicate_pair_ids:
            raise RuntimeError(f"Duplicate pair_id found in {expected_track}: {duplicate_pair_ids[:10]}")

        for r in records:
            validate_one_record(r)

            if r["track"] != expected_track:
                raise RuntimeError(f"Track mismatch: expected {expected_track}, got {r['track']}")

            if expected_track == "text_only" and r["reference_image"] != "":
                raise RuntimeError(f"text_only record has non-empty reference_image: {r['pair_id']}")

            if expected_track == "image-referenced" and r["reference_image"] == "":
                raise RuntimeError(f"image-referenced record has empty reference_image: {r['pair_id']}")

            if r.get("source_dataset") == SOURCE_DATASET:
                direction = direction_from_subtask(r["subtask"])
                if direction is not None:
                    if not is_valid_instruction_for_cloud(
                        r["instruction"],
                        direction=direction,
                        track=expected_track,
                    ):
                        raise RuntimeError(f"Invalid cloud instruction in {r['pair_id']}: {r['instruction']}")

            for key in ["source_image", "target_image", "reference_image"]:
                if r[key]:
                    abs_path = os.path.join(OUTPUT_ROOT, r[key])
                    if not os.path.exists(abs_path):
                        raise RuntimeError(f"Missing output image file: {abs_path}")


    def validate_counts(records: List[dict], expected_dataset: str):
        subset = [r for r in records if r.get("source_dataset") == expected_dataset]
        return subset


    def print_instruction_pool_stats(pool_stats: List[dict]):
        print("\n========== Cloud Instruction Pool Statistics ==========")
        print(
            f"Each cloud pool requests {REQUESTS_PER_TASK} × {INSTRUCTIONS_PER_REQUEST} "
            f"= {REQUESTS_PER_TASK * INSTRUCTIONS_PER_REQUEST} raw instructions."
        )

        for s in sorted(pool_stats, key=lambda x: x["key"]):
            print(
                f"{s['key']}: "
                f"requested={s['requested']}, "
                f"raw_returned={s['raw_returned']}, "
                f"valid_unique={s['valid_unique']}, "
                f"fallback_used={s['fallback_used']}"
            )


    def print_track_stats(records: List[dict], title: str):
        print(f"\n========== {title} ==========")
        print(f"Total records: {len(records)}")

        for field in ["source_dataset", "split", "task", "subtask", "track"]:
            print(f"\n[{field}]")
            counter = Counter(r[field] for r in records)
            for k, v in sorted(counter.items()):
                print(f"{k}: {v}")


    def print_cloud_only_stats(records: List[dict], title: str):
        cloud_records = [r for r in records if r.get("source_dataset") == SOURCE_DATASET]
        print_track_stats(cloud_records, title)


    def print_build_stats(build_stats: List[dict], used_images: set):
        print("\n========== Cloud Build Statistics ==========")
        print(f"TOTAL_TEXT_ONLY_RECORDS: {TOTAL_TEXT_ONLY_RECORDS}")
        print(f"TOTAL_IMAGE_REFERENCED_RECORDS: {TOTAL_IMAGE_REFERENCED_RECORDS}")
        print(f"TEST_TEXT_ONLY_RECORDS: {TEST_TEXT_ONLY_RECORDS}")
        print(f"TEST_IMAGE_REFERENCED_RECORDS: {TEST_IMAGE_REFERENCED_RECORDS}")
        print(f"TRAIN_TEXT_ONLY_RECORDS: {TRAIN_TEXT_ONLY_RECORDS}")
        print(f"TRAIN_IMAGE_REFERENCED_RECORDS: {TRAIN_IMAGE_REFERENCED_RECORDS}")
        print(f"Seasons used: {list(SEASON_CONFIGS.keys())}")
        print("Image usage policy: strict unique for both train and test.")
        print("RGB conversion: Sentinel-2 true color B04/B03/B02 from rasterio [C,H,W].")

        for s in build_stats:
            print(
                f"{s['split']} | {s['track']} | {s['direction']}: "
                f"target={s['target']}, actual={s['actual']}, "
                f"summer={s['summer']}, winter={s['winter']}"
            )

        print(f"Unique original images used across all cloud records: {len(used_images)}")


    # Main

    async def main():
        random.seed(SEED)
        ensure_dirs()

        print(f"[Info] SEN12MS_ROOT: {SEN12MS_ROOT}")
        print(f"[Info] OUTPUT_ROOT: {OUTPUT_ROOT}")
        print(f"[Info] SOURCE_DIR: {SOURCE_DIR}")
        print(f"[Info] TARGET_DIR: {TARGET_DIR}")
        print(f"[Info] REFERENCE_DIR: {REFERENCE_DIR}")
        print(f"[Info] OUTPUT_TEXT_ONLY_JSON: {OUTPUT_TEXT_ONLY_JSON}")
        print(f"[Info] OUTPUT_IMAGE_REFERENCED_JSON: {OUTPUT_IMAGE_REFERENCED_JSON}")
        print(f"[Info] INSTRUCTION_POOL_JSON: {INSTRUCTION_POOL_JSON}")
        print(f"[Info] MODEL_NAME: {MODEL_NAME}")
        print(f"[Info] CONCURRENCY_LIMIT: {CONCURRENCY_LIMIT}")
        print(f"[Info] TOTAL_TEXT_ONLY_RECORDS: {TOTAL_TEXT_ONLY_RECORDS}")
        print(f"[Info] TOTAL_IMAGE_REFERENCED_RECORDS: {TOTAL_IMAGE_REFERENCED_RECORDS}")
        print(f"[Info] TEST_TEXT_ONLY_RECORDS: {TEST_TEXT_ONLY_RECORDS}")
        print(f"[Info] TEST_IMAGE_REFERENCED_RECORDS: {TEST_IMAGE_REFERENCED_RECORDS}")
        print(f"[Info] SEASONS: {list(SEASON_CONFIGS.keys())}")
        print("[Info] RGB logic: R=B04, G=B03, B=B02 from rasterio output [C,H,W].")
        print("[Info] Image usage policy: strict unique for both train and test.")

        all_pairs = discover_sen12ms_pairs()
        if not all_pairs:
            raise RuntimeError("No SEN12MS clean/cloudy pairs found. Please check SEN12MS_ROOT and folder structure.")

        print(f"[Info] Total matched SEN12MS clean/cloudy pairs: {len(all_pairs)}")

        pools, pool_stats = await generate_cloud_instruction_pools()

        cloud_text_records, cloud_image_records, used_images, build_stats = build_strict_cloud_records(
            all_pairs=all_pairs,
            pools=pools,
        )

        combined_text_records, combined_image_records = append_cloud_to_existing_jsons(
            cloud_text_records=cloud_text_records,
            cloud_image_records=cloud_image_records,
        )

        validate_track_records(combined_text_records, "text_only")
        validate_track_records(combined_image_records, "image-referenced")

        # strict count checks for cloud subset only
        new_text_cloud = [r for r in cloud_text_records if r["source_dataset"] == SOURCE_DATASET]
        new_img_cloud = [r for r in cloud_image_records if r["source_dataset"] == SOURCE_DATASET]

        if len(new_text_cloud) != TOTAL_TEXT_ONLY_RECORDS:
            raise RuntimeError(
                f"New cloud text_only count mismatch: expected {TOTAL_TEXT_ONLY_RECORDS}, got {len(new_text_cloud)}"
            )

        if len(new_img_cloud) != TOTAL_IMAGE_REFERENCED_RECORDS:
            raise RuntimeError(
                f"New cloud image-referenced count mismatch: expected {TOTAL_IMAGE_REFERENCED_RECORDS}, got {len(new_img_cloud)}"
            )

        if Counter(r["split"] for r in new_text_cloud)["test"] != TEST_TEXT_ONLY_RECORDS:
            raise RuntimeError("Cloud text_only test count mismatch.")

        if Counter(r["split"] for r in new_img_cloud)["test"] != TEST_IMAGE_REFERENCED_RECORDS:
            raise RuntimeError("Cloud image-referenced test count mismatch.")

        with open(OUTPUT_TEXT_ONLY_JSON, "w", encoding="utf-8") as f:
            json.dump(combined_text_records, f, ensure_ascii=False, indent=2)

        with open(OUTPUT_IMAGE_REFERENCED_JSON, "w", encoding="utf-8") as f:
            json.dump(combined_image_records, f, ensure_ascii=False, indent=2)

        print_build_stats(build_stats, used_images)
        print_instruction_pool_stats(pool_stats)

        print_cloud_only_stats(cloud_text_records, "New SEN12MS Cloud Text-Only Statistics")
        print_cloud_only_stats(cloud_image_records, "New SEN12MS Cloud Image-Referenced Statistics")

        print_track_stats(combined_text_records, "Combined Text-Only JSON Statistics")
        print_track_stats(combined_image_records, "Combined Image-Referenced JSON Statistics")

        print(f"\n[Done] Appended SEN12MS cloud records to: {OUTPUT_TEXT_ONLY_JSON}")
        print(f"[Done] Appended SEN12MS cloud records to: {OUTPUT_IMAGE_REFERENCED_JSON}")

    asyncio.run(main())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RS-OmniEdit atmosphere records.")
    parser.add_argument(
        "component",
        choices=["fog", "cloud", "all"],
        nargs="?",
        default="all",
        help="Atmosphere component to build. Default: all.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.component in {"fog", "all"}:
        run_fog()
    if args.component in {"cloud", "all"}:
        run_cloud()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
