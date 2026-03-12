import argparse
import glob
import json
import os
import random
import shutil
from pathlib import Path
from typing import List

import torch
from accelerate.utils import set_seed
from PIL import Image
from tqdm import tqdm
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor


def center_crop(im: Image.Image) -> Image.Image:
    width, height = im.size
    min_dim = min(width, height)
    left = int((width - min_dim) / 2)
    top = int((height - min_dim) / 2)
    right = left + min_dim
    bottom = top + min_dim
    return im.crop((left, top, right, bottom))


def load_im_from_path(im_path: str, image_resolution: int) -> Image.Image:
    image = Image.open(im_path).convert("RGB")
    image = center_crop(image)
    image = image.resize((image_resolution, image_resolution), Image.LANCZOS)
    return image


def derive_pair_path(p0: Path) -> Path:
    """
    将 xxx_0.png -> xxx_1.png
    只替换文件名末尾的 _0，避免污染目录名（如 _work_00000 之类）。
    """
    if not p0.stem.endswith("_0"):
        raise ValueError(f"Not a *_0 file: {p0.name}")
    stem1 = p0.stem[:-2] + "_1"  # 去掉末尾的 _0 换成 _1
    return p0.with_name(stem1 + p0.suffix)


@torch.inference_mode()
def caption_one_image(
    processor: LlavaNextProcessor,
    model: LlavaNextForConditionalGeneration,
    device: str,
    im: Image.Image,
    max_new_tokens: int = 50,
) -> str:
    # 你原来的 prompt 保留
    prompt = "[INST] <image>\nDescribe the image using five phrases and separate the phrases using commas.[/INST]"

    # 关键修复：必须用 keyword 参数，避免把 prompt 当成 image source
    inputs = processor(
        text=prompt,
        images=im,
        return_tensors="pt",
    )

    # inputs 是 BatchEncoding：把 tensor 全部搬到 GPU
    inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    out_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    # 和你原逻辑一致：取 [/INST] 后的回答部分
    if "[/INST]" in out_text:
        out_text = out_text.split("[/INST]")[-1].strip()

    return out_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True, type=str, help="path to the image dir")
    parser.add_argument("--json_path", required=True, type=str, help="path to the caption output dir")
    args, extras = parser.parse_known_args()

    set_seed(42)

    image_resolution = 768
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 初始化模型
    model_dir = "/root/autodl-tmp/FreeMorph/models/llava-v1.6-mistral-7b-hf"

    processor = LlavaNextProcessor.from_pretrained(
        model_dir,
        local_files_only=True
    )

    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True
    )
    model.eval()
    model.to(device)

    # 收集所有图片，优先只处理 *_0.(png/jpg/jpeg/webp)
    img_dir = Path(args.image_path)
    if not img_dir.exists():
        raise FileNotFoundError(f"--image_path not found: {args.image_path}")

    exts = {".png", ".jpg", ".jpeg", ".webp"}
    all_files = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])

    json_output: List[dict] = []
    exp_id = 0

    for p0 in tqdm(all_files, desc="caption pairs"):
        # 只处理 *_0 文件，避免 enumerate 里混入 *_1 导致 exp_id/配对混乱
        if not p0.stem.endswith("_0"):
            continue

        try:
            p1 = derive_pair_path(p0)
        except Exception as e:
            print(f"[WARN] skip {p0.name}: {e}")
            continue

        if not p1.exists():
            print(f"[WARN] missing pair image: {p1}")
            continue

        try:
            im0 = load_im_from_path(str(p0), image_resolution=image_resolution)
            im1 = load_im_from_path(str(p1), image_resolution=image_resolution)
        except Exception as e:
            print(f"[WARN] image load failed: {p0.name} / {p1.name}: {e}")
            continue

        try:
            prompt1 = caption_one_image(processor, model, device, im0, max_new_tokens=50)
            prompt2 = caption_one_image(processor, model, device, im1, max_new_tokens=50)
        except Exception as e:
            print(f"[WARN] caption model failed on {p0.name}: {e}")
            continue

        json_output.append(
            {
                "exp_id": exp_id,
                "image_paths": [str(p0), str(p1)],
                "prompts": [prompt1, prompt2],
            }
        )
        exp_id += 1

    os.makedirs(args.json_path, exist_ok=True)
    out_file = os.path.join(args.json_path, "caption.json")

    # 维持你原来的 jsonl 写法（每行一个 json）
    with open(out_file, "w", encoding="utf-8") as f:
        for item in json_output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[DONE] wrote {len(json_output)} items to {out_file}")


if __name__ == "__main__":
    main()
