# #!/usr/bin/env python3
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # ✅ 放在 torch 前
# import json
# import argparse
# from pathlib import Path
# from PIL import Image
# from transformers import AutoModelForCausalLM
# import torch

# def load_deqa_model():
#     return AutoModelForCausalLM.from_pretrained(
#         "zhiyuanyou/DeQA-Score-Mix3",
#         trust_remote_code=True,
#         attn_implementation="eager",
#         torch_dtype=torch.float16,
#         device_map="auto",
#     )

# def score_images(model, image_paths):
#     scores = []
#     for p in image_paths:
#         image = Image.open(p).convert("RGB")
#         score = model.score([image])[0]
#         scores.append(float(score))
#         torch.cuda.empty_cache()  # 保险释放
#     return scores

# def collect_images_from_folder(root: Path):
#     """
#     查找路径下每个子文件夹中 samples/0000.png 图片的路径。
#     返回一个列表。
#     """
#     image_paths = []
#     subfolders = sorted([p for p in root.iterdir() if p.is_dir()])
#     for folder in subfolders:
#         img_path = folder / "samples" / "0000.png"
#         if img_path.exists():
#             image_paths.append(img_path)
#         else:
#             print(f"[WARN] Missing image: {img_path}")
#     return image_paths

# def main():
#     parser = argparse.ArgumentParser(
#         description="Compare average DeQA scores between ori_img and opt_img folders"
#     )
#     parser.add_argument("--ori-img", type=str, required=True, help="Path to original image folder")
#     parser.add_argument("--opt-img", type=str, required=True, help="Path to optimized image folder")
#     parser.add_argument("--hf-home", type=str, default=None, help="Optional: HF_HOME cache dir")
#     parser.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES")
#     args = parser.parse_args()

#     if args.hf_home:
#         os.environ["HF_HOME"] = args.hf_home
#     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

#     print("Loading DeQA model...")
#     model = load_deqa_model()

#     results = {}

#     for label, folder_path in [("ori_img", args.ori_img), ("opt_img", args.opt_img)]:
#         folder = Path(folder_path)
#         print(f"\n→ Scanning {label} folder: {folder}")
#         image_paths = collect_images_from_folder(folder)
#         if not image_paths:
#             print(f"[SKIP] No valid images in {folder}")
#             continue

#         print(f"Scoring {len(image_paths)} images...")
#         scores = score_images(model, image_paths)
#         avg_score = sum(scores) / len(scores)
#         results[label] = avg_score
#         print(f"[RESULT] {label} average DeQA score: {avg_score:.4f}")

#     # 比较结果
#     if "ori_img" in results and "opt_img" in results:
#         delta = results["opt_img"] - results["ori_img"]
#         print(f"\n[COMPARE] opt_img - ori_img = {delta:.4f}")
#         if delta > 0:
#             print("✅ Optimized image quality improved.")
#         elif delta < 0:
#             print("⚠️ Optimized image quality decreased.")
#         else:
#             print("🔁 No change in quality.")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import json
import argparse
from pathlib import Path
from PIL import Image
from transformers import AutoModelForCausalLM
import torch

def load_deqa_model():
    return AutoModelForCausalLM.from_pretrained(
        "zhiyuanyou/DeQA-Score-Mix3",
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch.float16,
        device_map="auto",
    )

def score_images(model, image_paths):
    scores = []
    for p in image_paths:
        image = Image.open(p).convert("RGB")
        score = model.score([image])[0]
        scores.append(float(score))
        torch.cuda.empty_cache()
    return scores

def collect_images_from_folder(root: Path):
    image_paths = []
    subfolders = sorted([p for p in root.iterdir() if p.is_dir()])
    for folder in subfolders:
        img_path = folder / "samples" / "0000.png"
        if img_path.exists():
            image_paths.append(img_path)
        else:
            print(f"[WARN] Missing image: {img_path}")
    return image_paths

def main():
    parser = argparse.ArgumentParser(
        description="Score only opt_img folder using DeQA"
    )
    parser.add_argument("--opt-img", type=str, required=True, help="Path to optimized image folder")
    parser.add_argument("--hf-home", type=str, default=None, help="Optional: HF_HOME cache dir")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print("Loading DeQA model...")
    model = load_deqa_model()

    label = "opt_img"
    folder = Path(args.opt_img)
    print(f"\n→ Scanning {label} folder: {folder}")
    image_paths = collect_images_from_folder(folder)
    if not image_paths:
        print(f"[SKIP] No valid images in {folder}")
        return

    print(f"Scoring {len(image_paths)} images...")
    scores = score_images(model, image_paths)
    avg_score = sum(scores) / len(scores)
    print(f"[RESULT] {label} average DeQA score: {avg_score:.4f}")

if __name__ == "__main__":
    main()

