import os
import json
import random
import asyncio
import shutil
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
from openai import AsyncOpenAI

try:
    import tacoreader.v1 as tacoreader
    import rasterio as rio
except ImportError:
    print("Please install dependencies: pip install 'tacoreader<1.0' rasterio")
    raise


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

SEN2_ROOT = os.environ.get("SEN2_ROOT", "data/SEN2NAIPv2")
OLI2MSI_ROOT = os.environ.get("OLI2MSI_ROOT", "data/oli2msi")

OUTPUT_ROOT = "./RS-OmniEdit-Resolution"

SOURCE_DIR = os.path.join(OUTPUT_ROOT, "source")
TARGET_DIR = os.path.join(OUTPUT_ROOT, "target")
OUTPUT_JSON = os.path.join(OUTPUT_ROOT, "data.json")

TEST_NUM_SEN2 = 100
TEST_NUM_OLI2MSI = 100

REBUILD_OUTPUT = True

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")

client = AsyncOpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY
)

MODEL_NAME = os.environ.get("RS_UNIEDIT_LLM_MODEL", "Qwen/Qwen3.6-35B-A3B")


async def build_instruction_pool(direction_desc, n_requests=10):
    """
    生成通用分辨率编辑指令池。
    不区分 SEN2NAIPv2 / OLI2MSI。
    """
    async def one_call():
        prompt = f"""You are a remote sensing image editing data construction assistant.

Task:
Generate a list of 50 diverse, natural, and imperative image editing instructions in English.

Scenario:
{direction_desc}

Requirements:
1. Use varied verbs and expressions.
2. Must be imperative, like a human asking for an image edit.
3. Do NOT use specific sensor names, such as Sentinel-2, NAIP, Landsat, OLI, MSI, etc.
4. Do NOT use exact resolution numbers, such as 10m, 1m, 30m, 3x, etc.
5. Use general natural terms such as:
   - satellite image
   - aerial image
   - high resolution
   - low resolution
   - clearer
   - sharper
   - more detailed
   - lower quality
   - blurrier
   - more pixelated
6. Output strictly in JSON format containing a list named "instructions".

Output format:
{{
  "instructions": [
    "Make this satellite image sharper and more detailed.",
    "Degrade the quality of this aerial view."
  ]
}}
"""
        try:
            resp = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                response_format={"type": "json_object"}
            )
            text = resp.choices[0].message.content.strip()
            data = json.loads(text)
            instructions = data.get("instructions", [])
            if not isinstance(instructions, list):
                return []
            return [x.strip() for x in instructions if isinstance(x, str) and x.strip()]
        except Exception as e:
            print(f"Instruction generation failed: {e}")
            return []

    results = await asyncio.gather(*[one_call() for _ in range(n_requests)])

    pool = []
    for r in results:
        pool.extend(r)

    banned_terms = [
        "sentinel", "sentinel-2", "naip", "landsat", "oli", "msi",
        "10m", "1m", "30m", "3x", "2x", "4x",
        "dataset", "metadata"
    ]

    clean_pool = []
    seen = set()

    for ins in pool:
        low = ins.lower()
        if any(term in low for term in banned_terms):
            continue
        if ins not in seen:
            seen.add(ins)
            clean_pool.append(ins)

    return clean_pool


async def build_all_instruction_pools():
    print("🧠 正在构建通用分辨率指令池...")

    sr_desc = "Convert a low-resolution remote sensing image into a clearer, sharper, and more detailed high-resolution image."
    deg_desc = "Degrade a high-resolution remote sensing image into a blurrier, less detailed, lower-resolution image."

    sr_pool_task = build_instruction_pool(sr_desc, n_requests=10)
    deg_pool_task = build_instruction_pool(deg_desc, n_requests=10)

    sr_pool, deg_pool = await asyncio.gather(sr_pool_task, deg_pool_task)

    if not sr_pool or not deg_pool:
        raise RuntimeError("Instruction pool generation failed. Check the vLLM endpoint, model path, port, and response_format support.")

    print(f"Super-resolution instruction count: {len(sr_pool)}")
    print(f"Degradation instruction count: {len(deg_pool)}")

    return {
        "sr": sr_pool,
        "deg": deg_pool
    }


# 输出目录
def reset_output_dirs():
    if REBUILD_OUTPUT and os.path.exists(OUTPUT_ROOT):
        print(f"🧹 清空旧输出目录: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)

    os.makedirs(SOURCE_DIR, exist_ok=True)
    os.makedirs(TARGET_DIR, exist_ok=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def pil_save(img, path):
    img.save(path, format="PNG", optimize=False)


# 图像读取与可视化
def stretch_to_uint8(rgb):
    """
    将 H,W,3 的遥感 RGB 数据拉伸到 uint8。
    只做数值可视化，不改变图像尺寸。
    """
    rgb = np.nan_to_num(rgb)
    orig_dtype = rgb.dtype
    rgb = rgb.astype(np.float32)

    minv = float(rgb.min())
    maxv = float(rgb.max())

    # 反射率型 0~1 浮点遥感图。
    # 不能直接乘 255，否则常见的 0.03~0.15 会整体发黑。
    # 这里改成按通道百分位拉伸，更接近常见遥感可视化。
    if minv >= 0 and maxv <= 1.5:
        out = np.zeros_like(rgb, dtype=np.float32)
        for c in range(3):
            ch = rgb[:, :, c]
            lo = np.percentile(ch, 2)
            hi = np.percentile(ch, 98)
            if hi <= lo:
                out[:, :, c] = np.clip(ch * 255.0, 0, 255)
            else:
                out[:, :, c] = (ch - lo) / (hi - lo) * 255.0
        return np.clip(out, 0, 255).astype(np.uint8)

    # 已经是 0-255 图像。只对整数型直接保留，
    # 避免把 0~1 的 float 误判为 0~255 后整体截成全黑。
    if np.issubdtype(orig_dtype, np.integer) and minv >= 0 and maxv <= 255:
        return np.clip(rgb, 0, 255).astype(np.uint8)

    # 遥感高动态范围，用百分位拉伸
    out = np.zeros_like(rgb, dtype=np.float32)
    for c in range(3):
        ch = rgb[:, :, c]
        lo = np.percentile(ch, 2)
        hi = np.percentile(ch, 98)

        if hi <= lo:
            out[:, :, c] = np.clip(ch, 0, 255)
        else:
            out[:, :, c] = (ch - lo) / (hi - lo) * 255.0

    return np.clip(out, 0, 255).astype(np.uint8)


def choose_rgb_bands(arr, sensor_hint=""):
    """
    输入:
        arr: C,H,W

    输出:
        rgb: H,W,3

    注意：
    这里只选择 RGB 通道，不改变 H/W 尺寸。
    """
    if arr.ndim != 3:
        raise ValueError(f"Unexpected array shape: {arr.shape}")

    c, h, w = arr.shape

    if c == 1:
        gray = arr[0]
        return np.stack([gray, gray, gray], axis=-1)

    if c < 3:
        raise ValueError(f"Need at least 3 channels, got shape: {arr.shape}")

    # 默认取前三个通道。
    # 如果后面发现 OLI2MSI 颜色明显不对，可以改成 [2, 1, 0]。
    idx = [0, 1, 2]

    rgb = arr[idx].transpose(1, 2, 0)
    return rgb


def read_tif_as_pil(path, sensor_hint=""):
    """
    读取 tif 并转成 PIL RGB。
    不 resize，不 crop，不 padding，保持原始 H/W 尺寸。
    """
    with rio.open(path) as src:
        arr = src.read()

    rgb = choose_rgb_bands(arr, sensor_hint=sensor_hint)
    rgb_uint8 = stretch_to_uint8(rgb)

    return Image.fromarray(rgb_uint8)


# SEN2NAIPv2 收集与读取
_TACO_CACHE = {}


def get_taco_dataset(taco_path):
    if taco_path not in _TACO_CACHE:
        _TACO_CACHE[taco_path] = tacoreader.load(taco_path)
    return _TACO_CACHE[taco_path]


def collect_sen2naipv2_pairs():
    """
    只收集 SEN2NAIPv2 的 crosssensor 子集。
    """
    taco_files = []

    for f in os.listdir(SEN2_ROOT):
        if not f.endswith(".taco"):
            continue
        if "crosssensor" in f:
            taco_files.append(os.path.join(SEN2_ROOT, f))

    taco_files = sorted(taco_files)

    print(f"SEN2NAIPv2 crosssensor taco 文件数: {len(taco_files)}")

    pairs = []

    for taco_path in taco_files:
        try:
            dataset = get_taco_dataset(taco_path)
            length = len(dataset)
        except Exception as e:
            print(f"Warning: 跳过无法读取的 taco 文件: {taco_path}, error={e}")
            continue

        base_name = os.path.basename(taco_path)

        for i in range(length):
            pairs.append({
                "source_dataset": "SEN2NAIPv2",
                "backend": "sen2naipv2",
                "taco_path": taco_path,
                "index": i,
                "raw_pair_key": f"{base_name}::{i}"
            })

    print(f"SEN2NAIPv2 可用 pair 数: {len(pairs)}")
    return pairs


def read_sen2naipv2_pair(pair_info):
    """
    默认:
    - read(0): LR
    - read(1): HR

    不改变 LR / HR 原始尺寸。
    """
    dataset = get_taco_dataset(pair_info["taco_path"])
    item = dataset.read(pair_info["index"])

    lr_file = item.read(0)
    hr_file = item.read(1)

    lr_img = read_tif_as_pil(lr_file, sensor_hint=os.path.basename(lr_file))
    hr_img = read_tif_as_pil(hr_file, sensor_hint=os.path.basename(hr_file))

    return lr_img, hr_img


# OLI2MSI 收集与读取
def match_tif_pairs(lr_dir, hr_dir):
    """
    根据相同文件名匹配 LR / HR。
    """
    if not os.path.isdir(lr_dir):
        print(f"Warning: LR 文件夹不存在: {lr_dir}")
        return []
    if not os.path.isdir(hr_dir):
        print(f"Warning: HR 文件夹不存在: {hr_dir}")
        return []

    lr_files = [
        f for f in os.listdir(lr_dir)
        if f.lower().endswith((".tif", ".tiff"))
    ]
    hr_files = [
        f for f in os.listdir(hr_dir)
        if f.lower().endswith((".tif", ".tiff"))
    ]

    lr_map = {
        os.path.splitext(f)[0]: os.path.join(lr_dir, f)
        for f in lr_files
    }
    hr_map = {
        os.path.splitext(f)[0]: os.path.join(hr_dir, f)
        for f in hr_files
    }

    common_keys = sorted(set(lr_map.keys()) & set(hr_map.keys()))

    pairs = []
    for key in common_keys:
        pairs.append((lr_map[key], hr_map[key], key))

    return pairs


def collect_oli2msi_pairs():
    """
    把 OLI2MSI 的 train 和 test 文件夹都收集进来，
    后面我们自己统一划分 100 个 test。
    """
    train_lr = os.path.join(OLI2MSI_ROOT, "train_lr")
    train_hr = os.path.join(OLI2MSI_ROOT, "train_hr")
    test_lr = os.path.join(OLI2MSI_ROOT, "test_lr")
    test_hr = os.path.join(OLI2MSI_ROOT, "test_hr")

    train_pairs = match_tif_pairs(train_lr, train_hr)
    test_pairs = match_tif_pairs(test_lr, test_hr)

    pairs = []

    for lr_path, hr_path, key in train_pairs:
        pairs.append({
            "source_dataset": "OLI2MSI",
            "backend": "oli2msi",
            "origin_split": "train",
            "lr_path": lr_path,
            "hr_path": hr_path,
            "raw_pair_key": f"train::{key}"
        })

    for lr_path, hr_path, key in test_pairs:
        pairs.append({
            "source_dataset": "OLI2MSI",
            "backend": "oli2msi",
            "origin_split": "test",
            "lr_path": lr_path,
            "hr_path": hr_path,
            "raw_pair_key": f"test::{key}"
        })

    print(f"OLI2MSI 可用 pair 数: {len(pairs)}")
    print(f"   - train pair: {len(train_pairs)}")
    print(f"   - test pair:  {len(test_pairs)}")

    return pairs


def read_oli2msi_pair(pair_info):
    """
    读取 OLI2MSI LR / HR。
    不改变 LR / HR 原始尺寸。
    """
    lr_img = read_tif_as_pil(pair_info["lr_path"], sensor_hint="oli2msi_lr")
    hr_img = read_tif_as_pil(pair_info["hr_path"], sensor_hint="oli2msi_hr")

    return lr_img, hr_img


# 划分 train / test
def split_train_test(sen2_pairs, oli_pairs):
    random.shuffle(sen2_pairs)
    random.shuffle(oli_pairs)

    if len(sen2_pairs) < TEST_NUM_SEN2:
        raise ValueError(f"SEN2NAIPv2 数量不足，无法划分 {TEST_NUM_SEN2} 个 test。")

    if len(oli_pairs) < TEST_NUM_OLI2MSI:
        raise ValueError(f"OLI2MSI 数量不足，无法划分 {TEST_NUM_OLI2MSI} 个 test。")

    test_sen2 = sen2_pairs[:TEST_NUM_SEN2]
    train_sen2 = sen2_pairs[TEST_NUM_SEN2:]

    test_oli = oli_pairs[:TEST_NUM_OLI2MSI]
    train_oli = oli_pairs[TEST_NUM_OLI2MSI:]

    train_pairs = train_sen2 + train_oli
    test_pairs = test_sen2 + test_oli

    random.shuffle(train_pairs)
    random.shuffle(test_pairs)

    print("\n📦 Split Stats")
    print(f"Train total: {len(train_pairs)}")
    print(f"Test total:  {len(test_pairs)}")
    print(f"  - Test SEN2NAIPv2: {len(test_sen2)}")
    print(f"  - Test OLI2MSI:    {len(test_oli)}")

    return train_pairs, test_pairs


def make_balanced_directions(n):
    """
    每个 pair 只生成一个方向。
    整体方向尽量均衡。
    """
    n_sr = n // 2
    n_deg = n - n_sr

    directions = ["lr_to_hr"] * n_sr + ["hr_to_lr"] * n_deg
    random.shuffle(directions)

    return directions


# 导出数据
def read_pair(pair_info):
    if pair_info["backend"] == "sen2naipv2":
        return read_sen2naipv2_pair(pair_info)
    elif pair_info["backend"] == "oli2msi":
        return read_oli2msi_pair(pair_info)
    else:
        raise ValueError(f"Unknown backend: {pair_info['backend']}")


def export_split(pairs, split_name, instruction_pools, start_index=0):
    if split_name not in ["train", "test"]:
        raise ValueError("split_name must be train or test")

    directions = make_balanced_directions(len(pairs))
    records = []

    skipped = 0

    for local_idx, pair_info in enumerate(tqdm(pairs, desc=f"Exporting {split_name}")):
        global_idx = start_index + len(records)
        direction = directions[local_idx]

        try:
            lr_img, hr_img = read_pair(pair_info)

            file_name = f"{global_idx:06d}.png"

            if direction == "lr_to_hr":
                source_img = lr_img
                target_img = hr_img
                instruction = random.choice(instruction_pools["sr"])
            else:
                source_img = hr_img
                target_img = lr_img
                instruction = random.choice(instruction_pools["deg"])

            source_path = os.path.join(SOURCE_DIR, file_name)
            target_path = os.path.join(TARGET_DIR, file_name)

            pil_save(source_img, source_path)
            pil_save(target_img, target_path)

            record = {
                "pair_id": f"res_{global_idx:06d}",
                "task": "resolution",
                "track": "text_only",
                "split": split_name,
                "source_dataset": pair_info["source_dataset"],
                "direction": direction,
                "source_image": f"source/{file_name}",
                "target_image": f"target/{file_name}",
                "instruction": instruction,
                "raw_pair_key": pair_info["raw_pair_key"],

                # 记录原始尺寸，方便后面检查
                "source_size": [source_img.size[0], source_img.size[1]],
                "target_size": [target_img.size[0], target_img.size[1]]
            }

            records.append(record)

        except Exception:
            skipped += 1
            continue

    print(f"{split_name} 导出完成: {len(records)} 条，跳过 {skipped} 条。")

    return records, start_index + len(records)


def print_stats(records, name):
    ds_count = defaultdict(int)
    dir_count = defaultdict(int)
    size_count = defaultdict(int)

    for r in records:
        ds_count[r["source_dataset"]] += 1
        dir_count[r["direction"]] += 1
        size_key = f'{r["source_size"]}->{r["target_size"]}'
        size_count[size_key] += 1

    print(f"\n===== {name} Stats =====")
    print(f"Total: {len(records)}")

    print("By source_dataset:")
    for k, v in sorted(ds_count.items()):
        print(f"  {k}: {v}")

    print("By direction:")
    for k, v in sorted(dir_count.items()):
        print(f"  {k}: {v}")

    print("Top size mappings:")
    for k, v in sorted(size_count.items(), key=lambda x: -x[1])[:10]:
        print(f"  {k}: {v}")


# 主流程
async def main():
    reset_output_dirs()

    # 1. 构建统一通用指令池
    instruction_pools = await build_all_instruction_pools()

    # 2. 收集两个数据集的 pair
    sen2_pairs = collect_sen2naipv2_pairs()
    oli_pairs = collect_oli2msi_pairs()

    print("\n📊 Dataset Pair Count")
    print(f"SEN2NAIPv2: {len(sen2_pairs)}")
    print(f"OLI2MSI:    {len(oli_pairs)}")
    print(f"Total:      {len(sen2_pairs) + len(oli_pairs)}")

    # 3. 每个数据集各划分 100 个测试样本
    train_pairs, test_pairs = split_train_test(sen2_pairs, oli_pairs)

    # 4. 导出图片和 JSON
    train_records, next_idx = export_split(
        train_pairs,
        split_name="train",
        instruction_pools=instruction_pools,
        start_index=0
    )

    test_records, next_idx = export_split(
        test_pairs,
        split_name="test",
        instruction_pools=instruction_pools,
        start_index=next_idx
    )

    # 5. 保存 JSON
    all_records = train_records + test_records
    save_json(all_records, OUTPUT_JSON)

    # 6. 打印统计
    print_stats(train_records, "Train")
    print_stats(test_records, "Test")
    print_stats(all_records, "All")

    print("\n🎉 分辨率任务数据集构建完成！")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"Output root: {OUTPUT_ROOT}")
    print("\n注意：本脚本不会 resize 图像。source 和 target 可能尺寸不同，这是按原始 LR/HR 尺寸保存的结果。")


if __name__ == "__main__":
    asyncio.run(main())
