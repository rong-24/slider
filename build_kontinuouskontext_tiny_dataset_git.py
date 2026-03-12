# 这一版是git版，主要用于git上，真正的文件在/root/autodl-tmp/kk/kontinuouskontext/data/build_kontinuouskontext_tiny_dataset.py

import os
import io
import json
import glob
import random
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import pyarrow.parquet as pq
import lpips
num_interp = 5 # 7 --- IGNORE --- FreeMorph 输出的中间帧数量，默认为5，过多可能导致LPIPS计算过慢


# -------------------------
# 0) 配置区
# -------------------------
@dataclass
class Cfg:
    parquet_glob: str = "/root/autodl-tmp/dataset/ffhq512_data/data/*.parquet"
    out_dir: str = "/root/autodl-tmp/kontinuouskontext_ffhq_debug"

    qwen_path: str = "/root/autodl-tmp/FreeMorph/models/qwen"
    flux_path: str = "/root/autodl-tmp/kontinuous-kontext/models/black-forest-labs/FLUX.1-Kontext-dev"
    freemorph_root: str = "/root/autodl-tmp/FreeMorph"

    n_samples: int = 5
    seed: int = 0
    resolution: int = 512

    use_freemorph: bool = True
    qwen_max_new_tokens: int = 64
    safeworkdir_no__0_substring: bool = True

    # 新增：可视化与方向
    draw_text_on_stack: bool = True
    text_panel_height: int = 150
    auto_reverse_direction: bool = True

    # 新增：调试时可保存单张图
    save_individual_images: bool = False


# -------------------------
# 1) 通用工具
# -------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s))

def resize_to(img: Image.Image, res: int) -> Image.Image:
    return img.resize((res, res), Image.BICUBIC)

def to_pil_from_parquet_image_cell(cell) -> Image.Image:
    if isinstance(cell, Image.Image):
        return cell.convert("RGB")

    if isinstance(cell, (bytes, bytearray)):
        return Image.open(io.BytesIO(cell)).convert("RGB")

    if isinstance(cell, dict):
        b = cell.get("bytes", None)
        if b is not None:
            return Image.open(io.BytesIO(b)).convert("RGB")
        p = cell.get("path", None)
        if p and os.path.exists(p):
            return Image.open(p).convert("RGB")

    raise TypeError(f"Unrecognized image cell type: {type(cell)}")

def hstack(images: List[Image.Image]) -> Image.Image:
    w, h = images[0].size
    canvas = Image.new("RGB", (w * len(images), h))
    for i, im in enumerate(images):
        canvas.paste(im, (i * w, 0))
    return canvas

def safe_filename_text(s: str, max_len: int = 180) -> str:
    s = s.replace("/", "_").replace("\\", "_").replace(":", "_")
    s = s.replace("\n", " ").replace("\r", " ").strip()
    s = "_".join(s.split())
    return s[:max_len]

def wrap_text(text: str, max_chars: int = 80) -> List[str]:
    text = text.strip()
    if not text:
        return [""]
    words = text.split()
    lines = []
    cur = []
    cur_len = 0
    for w in words:
        add_len = len(w) if not cur else len(w) + 1
        if cur_len + add_len <= max_chars:
            cur.append(w)
            cur_len += add_len
        else:
            lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
    if cur:
        lines.append(" ".join(cur))
    return lines

def get_default_font(size: int = 20):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()

def draw_text_panel(
    stack: Image.Image,
    lines: List[str],
    panel_height: int = 150,
    bg=(20, 20, 20),
    fg=(240, 240, 240),
) -> Image.Image:
    w, h = stack.size
    canvas = Image.new("RGB", (w, h + panel_height), color=bg)
    canvas.paste(stack, (0, panel_height))

    draw = ImageDraw.Draw(canvas)
    font_title = get_default_font(22)
    font_text = get_default_font(18)

    y = 10
    for i, line in enumerate(lines):
        font = font_title if i == 0 else font_text
        draw.text((12, y), line, fill=fg, font=font)
        y += 28 if i == 0 else 24

    return canvas


# -------------------------
# 2) Qwen：生成 edit instruction
# -------------------------
def build_qwen_generator(qwen_path: str, device: str = "cuda", max_new_tokens: int = 64):
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as QwenModel
    except Exception:
        from transformers import AutoModelForVision2Seq as QwenModel

    print(f"[QWEN] loading processor/model from: {qwen_path}")
    processor = AutoProcessor.from_pretrained(qwen_path, trust_remote_code=True)
    model = QwenModel.from_pretrained(
        qwen_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    categories = [
        "stylization",
        "scene_reimagination",
        "environment_change",
        "material_change",
        "appearance_change",
        "attribute_modification",
        "shape_morphing",
    ]

    def gen_instruction(img: Image.Image) -> Tuple[str, str]:
        cat = random.choice(categories)
        prompt = (
            "You are a visual editor instruction generator.\n"
            f"Category: {cat}\n"
            "Given the input, produce ONE editing instruction that causes a clearly visible change.\n"
            "The change must be obvious to a human viewer at a glance, not just subtle enhancement.\n"
            "Prefer changing color, style, shape, material, accessories, background, or overall appearance.\n"
            "Avoid weak instructions like 'enhance texture', 'improve lighting', 'add glow', or other barely noticeable refinements.\n"
            "Constraints:\n"
            "- Do not mention 'image' or 'photo'.\n"
            "- Be specific to objects in the input.\n"
            "- Keep the main subject recognizable.\n"
            "- Output ONLY the instruction text, no explanation.\n"
        )

        if hasattr(processor, "apply_chat_template"):
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True)
        else:
            inputs = processor(images=img, text=prompt, return_tensors="pt")

        inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        if "input_ids" in inputs:
            gen_ids = generated[:, inputs["input_ids"].shape[1]:]
        else:
            gen_ids = generated

        out_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        out_text = out_text.replace("\n", " ").strip()

        if not out_text:
            out_text = "Edit the main subject in a plausible way"

        return cat, out_text

    return gen_instruction


# -------------------------
# 3) Flux：生成 full edit（端点）
# -------------------------
def build_flux_editor(flux_path: str):
    print(f"[FLUX] loading Flux Kontext from: {flux_path}")
    try:
        from diffusers import FluxKontextPipeline
        pipe = FluxKontextPipeline.from_pretrained(flux_path, torch_dtype=torch.bfloat16).to("cuda")
    except Exception:
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(flux_path, torch_dtype=torch.bfloat16).to("cuda")

    pipe.set_progress_bar_config(disable=True)

    def edit(img: Image.Image, instruction: str) -> Image.Image:
        with torch.no_grad():
            try:
                out = pipe(image=img, prompt=instruction)
            except TypeError:
                out = pipe(prompt=instruction, image=img, num_images_per_prompt=1)

        if hasattr(out, "images") and out.images:
            return out.images[0].convert("RGB")
        if isinstance(out, list) and out:
            return out[0].convert("RGB")
        return img

    return edit


# -------------------------
# 4) FreeMorph：caption + freemorph + 切分输出帧
# -------------------------
def prepare_freemorph_pair(work_dir: str, src: Image.Image, edit: Image.Image) -> str:
    pair_dir = os.path.join(work_dir, "image_pairs")
    ensure_dir(pair_dir)

    p0 = os.path.join(pair_dir, "pair1_0.png")
    p1 = os.path.join(pair_dir, "pair1_1.png")
    src.save(p0)
    edit.save(p1)
    return pair_dir

def load_caption_jsonl(caption_json: str) -> List[Dict]:
    items = []
    with open(caption_json, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def run_caption_py(freemorph_root: str, pair_dir: str, work_dir: str) -> str:
    """
    caption.py 的 --json_path 是“目录”，会写 work_dir/caption.json
    """
    caption_cmd = ["python", "caption.py", "--image_path", pair_dir, "--json_path", work_dir]
    print("[FREEMORPH] Running caption.py:", " ".join(caption_cmd))
    r = subprocess.run(caption_cmd, cwd=freemorph_root, capture_output=True, text=True)

    if r.returncode != 0:
        print("[FREEMORPH] caption.py stdout:\n", r.stdout)
        print("[FREEMORPH] caption.py stderr:\n", r.stderr)
        raise RuntimeError("caption.py failed")

    caption_json = os.path.join(work_dir, "caption.json")
    if (not os.path.exists(caption_json)) or (os.path.getsize(caption_json) == 0):
        print("[FREEMORPH] caption.py stdout:\n", r.stdout)
        print("[FREEMORPH] caption.py stderr:\n", r.stderr)
        raise RuntimeError(f"caption.json missing/empty: {caption_json}")
    return caption_json

def split_freemorph_grid(grid_path: str) -> List[Image.Image]:
    """
    FreeMorph 常用 make_grid/save_image 保存拼图，默认 padding=2，
    输出尺寸一般满足：
      W = cols*tile + (cols+1)*pad
      H = rows*tile + (rows+1)*pad
    这里自动推断 pad/tile/rows/cols 并切分。
    """
    im = Image.open(grid_path).convert("RGB")
    W, H = im.size

    best = None
    for pad in range(0, 33):
        for rows in range(1, 9):
            numerator = H - (rows + 1) * pad
            if numerator <= 0:
                continue
            if numerator % rows != 0:
                continue
            tile = numerator // rows
            if tile <= 0:
                continue

            if (W - pad) <= 0:
                continue
            denom = tile + pad
            if (W - pad) % denom != 0:
                continue
            cols = (W - pad) // denom
            if cols <= 0:
                continue

            if W == cols * tile + (cols + 1) * pad and H == rows * tile + (rows + 1) * pad:
                best = (pad, tile, rows, cols)
                break
        if best is not None:
            break

    if best is None:
        raise RuntimeError(f"Cannot parse grid size {W}x{H}. Please check FreeMorph output settings.")

    pad, tile, rows, cols = best
    frames = []
    for r in range(rows):
        for c in range(cols):
            x0 = pad + c * (tile + pad)
            y0 = pad + r * (tile + pad)
            x1 = x0 + tile
            y1 = y0 + tile
            frames.append(im.crop((x0, y0, x1, y1)))

    return frames

def run_freemorph_sequence(freemorph_root: str, work_dir: str, caption_json: str) -> List[Image.Image]:
    """
    在 work_dir 里运行 freemorph.py，保证输出落在 work_dir/eval_results/freemorph
    """
    freemorph_py = os.path.join(freemorph_root, "freemorph.py")
    cmd = ["python", freemorph_py, "--json_path", caption_json]
    print("[FREEMORPH] Running freemorph.py:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)

    if r.returncode != 0:
        print("[FREEMORPH] freemorph.py stdout:\n", r.stdout)
        print("[FREEMORPH] freemorph.py stderr:\n", r.stderr)
        raise RuntimeError("freemorph.py failed")

    out_dir = os.path.join(work_dir, "eval_results", "freemorph")
    pngs = sorted(glob.glob(os.path.join(out_dir, "*.png")))
    if not pngs:
        print("[FREEMORPH] freemorph.py stdout:\n", r.stdout)
        print("[FREEMORPH] freemorph.py stderr:\n", r.stderr)
        raise RuntimeError(f"No outputs found under {out_dir}")

    latest = max(pngs, key=lambda p: os.path.getmtime(p))
    frames = split_freemorph_grid(latest)

    if len(frames) < 3:
        raise RuntimeError(f"Too few frames parsed from grid: {len(frames)} ({latest})")

    return frames


# -------------------------
# 5) LPIPS
# -------------------------
def lpips_distance(lpips_model, a: Image.Image, b: Image.Image) -> float:
    import torchvision.transforms.functional as TF
    ta = TF.to_tensor(a).unsqueeze(0) * 2 - 1
    tb = TF.to_tensor(b).unsqueeze(0) * 2 - 1
    ta = ta.to("cuda")
    tb = tb.to("cuda")
    with torch.no_grad():
        d = lpips_model(ta, tb)
    return float(d.item())

def compute_lpips_triangle(lpips_model, start: Image.Image, mid: Image.Image, end: Image.Image) -> float:
    dl = lpips_distance(lpips_model, start, mid)
    dr = lpips_distance(lpips_model, mid, end)
    dd = lpips_distance(lpips_model, start, end)
    if dd == 0:
        return 0.0
    return float((dl + dr - dd) / dd)

def evaluate_lpips_sequences(lpips_model, imgs: List[Image.Image]) -> Tuple[List[float], List[float]]:
    first = [lpips_distance(lpips_model, imgs[i], imgs[i + 1]) for i in range(len(imgs) - 1)]
    tri = [compute_lpips_triangle(lpips_model, imgs[i], imgs[i + 1], imgs[i + 2]) for i in range(len(imgs) - 2)]
    return first, tri

def infer_direction_with_lpips(lpips_model, src: Image.Image, edit: Image.Image, inters: List[Image.Image]) -> Tuple[str, Dict[str, float]]:
    """
    判断当前 inters 是否更像 src -> ... -> edit
    返回:
      - "forward" : 当前顺序更合理
      - "reverse" : 当前顺序更像 edit -> ... -> src
      - "uncertain": 证据不足
    """
    if not inters:
        return "uncertain", {}

    first = inters[0]
    last = inters[-1]

    d_first_src = lpips_distance(lpips_model, first, src)
    d_first_edit = lpips_distance(lpips_model, first, edit)
    d_last_src = lpips_distance(lpips_model, last, src)
    d_last_edit = lpips_distance(lpips_model, last, edit)

    forward_score = (d_first_src + d_last_edit)
    reverse_score = (d_first_edit + d_last_src)

    metrics = {
        "d_first_src": d_first_src,
        "d_first_edit": d_first_edit,
        "d_last_src": d_last_src,
        "d_last_edit": d_last_edit,
        "forward_score": forward_score,
        "reverse_score": reverse_score,
    }

    margin = abs(forward_score - reverse_score)
    if margin < 1e-4:
        return "uncertain", metrics
    if forward_score < reverse_score:
        return "forward", metrics
    return "reverse", metrics


# -------------------------
# 6) 可视化
# -------------------------
def save_individuals_if_needed(cfg: Cfg, out_dir: str, sample_idx: int, src: Image.Image, inter5: List[Image.Image], edit: Image.Image):
    if not cfg.save_individual_images:
        return
    single_dir = os.path.join(out_dir, "single_frames", f"{sample_idx:05d}")
    ensure_dir(single_dir)
    src.save(os.path.join(single_dir, "src.png"))
    for i, im in enumerate(inter5):
        im.save(os.path.join(single_dir, f"mid_{i:02d}.png"))
    edit.save(os.path.join(single_dir, "edit.png"))

def build_annotated_stack(
    cfg: Cfg,
    stack_imgs: List[Image.Image],
    sample_idx: int,
    category: str,
    instruction: str,
    freemorph_ok: bool,
    direction_before: str,
    direction_after: str,
    freemorph_prompts: Optional[List[str]] = None,
) -> Image.Image:
    stack = hstack(stack_imgs)

    if not cfg.draw_text_on_stack:
        return stack

    title = f"sample={sample_idx:05d} | category={category} | freemorph_success={freemorph_ok}"
    line2 = f"qwen_instruction: {instruction}"
    line3 = f"direction: before={direction_before} -> after={direction_after} | expected: src -> mids -> edit"

    lines = [title]
    lines.extend(wrap_text(line2, max_chars=110))
    lines.extend(wrap_text(line3, max_chars=110))

    if freemorph_prompts is not None and len(freemorph_prompts) == 2:
        line4 = f"freemorph_caption_src: {freemorph_prompts[0]}"
        line5 = f"freemorph_caption_edit: {freemorph_prompts[1]}"
        lines.extend(wrap_text(line4, max_chars=110))
        lines.extend(wrap_text(line5, max_chars=110))

    return draw_text_panel(
        stack=stack,
        lines=lines,
        panel_height=cfg.text_panel_height,
    )


# -------------------------
# 7) 主流程
# -------------------------
def main():
    cfg = Cfg()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    ensure_dir(cfg.out_dir)
    out_sampling = os.path.join(cfg.out_dir, "sampling_data")
    ensure_dir(out_sampling)

    print("[INIT] parquet_glob:", cfg.parquet_glob)
    parquet_files = sorted(glob.glob(cfg.parquet_glob))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files matched: {cfg.parquet_glob}")
    print("[INIT] num parquet files:", len(parquet_files))

    gen_instruction = build_qwen_generator(cfg.qwen_path, device="cuda", max_new_tokens=cfg.qwen_max_new_tokens)
    flux_edit = build_flux_editor(cfg.flux_path)
    lpips_model = lpips.LPIPS(net="vgg").to("cuda").eval()

    meta: List[Dict] = []
    sample_idx = 0

    for pf in parquet_files:
        print(f"\n[PARQUET] reading: {pf}")
        table = pq.read_table(pf, columns=["image"])
        print(f"[PARQUET] rows: {table.num_rows}")

        for r in range(table.num_rows):
            if sample_idx >= cfg.n_samples:
                break

            img_cell = table["image"][r].as_py()
            try:
                src = to_pil_from_parquet_image_cell(img_cell)
            except Exception as e:
                print(f"[WARN] skip row {r} due to image decode error: {e}")
                continue

            src = resize_to(center_crop_square(src), cfg.resolution)

            # 1) Qwen 指令
            cat, instruction = gen_instruction(src)
            print(f"[{sample_idx}] category={cat}, instruction={instruction}")

            # 2) Flux full edit
            edit = flux_edit(src, instruction)
            edit = resize_to(center_crop_square(edit), cfg.resolution)

            # 3) FreeMorph 序列
            freemorph_ok = False
            freemorph_prompts = None
            direction_before = "not_checked"
            direction_after = "not_checked"
            direction_metrics = {}
            inter5: List[Image.Image]

            if cfg.use_freemorph:
                if cfg.safeworkdir_no__0_substring:
                    work_dir = os.path.join(cfg.out_dir, f"_wk{sample_idx:05d}")
                else:
                    work_dir = os.path.join(cfg.out_dir, f"_work_{sample_idx:05d}")

                ensure_dir(work_dir)

                try:
                    pair_dir = prepare_freemorph_pair(work_dir, src, edit)
                    caption_json = run_caption_py(cfg.freemorph_root, pair_dir, work_dir)

                    caption_items = load_caption_jsonl(caption_json)
                    if len(caption_items) > 0:
                        freemorph_prompts = caption_items[0].get("prompts", None)

                    frames = run_freemorph_sequence(cfg.freemorph_root, work_dir, caption_json)
                    frames = [resize_to(center_crop_square(f), cfg.resolution) for f in frames]

                    if len(frames) != num_interp:
                        idxs = np.linspace(0, len(frames) - 1, num_interp).round().astype(int).tolist()
                        frames = [frames[i] for i in idxs]

                    inter5 = frames

                    direction_before, direction_metrics = infer_direction_with_lpips(lpips_model, src, edit, inter5)
                    direction_after = direction_before

                    if cfg.auto_reverse_direction and direction_before == "reverse":
                        inter5 = inter5[::-1]
                        direction_after = "forward(auto_reversed)"

                    freemorph_ok = True

                except Exception as e:
                    print(f"[{sample_idx}] FreeMorph failed: {e} -> fallback to linear blend.")
                    iinter_frames = [
                        Image.blend(src, edit, k / (num_interp - 1)).convert("RGB")
                        for k in range(num_interp)
                    ]
                    freemorph_ok = False
                    direction_before = "fallback_linear_blend"
                    direction_after = "forward"

            else:
                inter_frames = [
                    Image.blend(src, edit, k / (num_interp - 1)).convert("RGB")
                    for k in range(num_interp)
                ]
                freemorph_ok = False
                direction_before = "linear_blend_only"
                direction_after = "forward"

            # 4) 拼 stack
            stack_imgs = [src] + inter5 + [edit]

            stack = build_annotated_stack(
                cfg=cfg,
                stack_imgs=stack_imgs,
                sample_idx=sample_idx,
                category=cat,
                instruction=instruction,
                freemorph_ok=freemorph_ok,
                direction_before=direction_before,
                direction_after=direction_after,
                freemorph_prompts=freemorph_prompts,
            )

            image_name = f"image_{sample_idx:05d}.png"
            stack_name = image_name.replace(".png", f"_nsamples_{num_interp}.png")
            stack_path = os.path.join(out_sampling, stack_name)
            stack.save(stack_path)

            #下面为了验证 stack_imgs[-1] 和 edit 是否真的相同，单独保存 edit
            edit_path = os.path.join(out_sampling, f"edit_{sample_idx:05d}.png")
            edit.save(edit_path)
            print(f"[{sample_idx}] saved pure flux edit -> {edit_path}")

            stack_imgs = [src] + inter5 + [edit]
            stack = hstack(stack_imgs)

            # 验证列表最后一张
            same_mem = np.array_equal(np.array(stack_imgs[-1]), np.array(edit))
            print(f"[{sample_idx}] stack_imgs[-1] == edit : {same_mem}")

            stack_path = os.path.join(out_sampling, stack_name)
            stack.save(stack_path)
            print(f"[{sample_idx}] saved stack -> {stack_path}")

            # print(f"[{sample_idx}] saved stack -> {stack_path}")

            save_individuals_if_needed(cfg, cfg.out_dir, sample_idx, src, inter5, edit)

            # 5) LPIPS
            lpips_seq, lpips_tri = evaluate_lpips_sequences(lpips_model, stack_imgs)
            lpips_kontext_edit = lpips_distance(lpips_model, src, edit)
            lpips_edit_inversion = lpips_distance(lpips_model, edit, inter5[-1])
            lpips_inversion_edit = lpips_distance(lpips_model, inter5[0], inter5[-1])

            meta.append({
                "image_name": image_name,
                "extended_image_name": f"{image_name[:-4]}_{safe_filename_text(cat)}|{safe_filename_text(instruction)}.png",
                "category": cat,
                "edit_instruction": instruction,
                "freemorph_prompts": freemorph_prompts,
                "direction_before_fix": direction_before,
                "direction_after_fix": direction_after,
                "direction_metrics": direction_metrics,
                "lpips_kontext_edit": lpips_kontext_edit,
                "lpips_edit_inversion": lpips_edit_inversion,
                "lpips_inversion_edit": lpips_inversion_edit,
                "lpips_sequence": lpips_seq,
                "lpips_sequence_triangle": lpips_tri,
                "freemorph_success": freemorph_ok,
            })

            sample_idx += 1

        if sample_idx >= cfg.n_samples:
            break

    json_path = os.path.join(cfg.out_dir, "sample_data_scores_w_scores.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n[DONE] wrote json:", json_path)
    print("[DONE] dataset root:", cfg.out_dir)
    print("[DONE] sampling_data:", out_sampling)
    print(f"[DONE] total samples: {len(meta)}")


if __name__ == "__main__":
    main()