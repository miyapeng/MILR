#!/usr/bin/env python3
import argparse
import os
import glob
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from transformers import AutoProcessor, AutoConfig, GitForCausalLM

# —— 内部配置区 —— 
GIT_CKPT_PATH = "/media/raid/workspace/miyapeng/.cache/models/git-large-vqav2"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_PATTERN = "*.png"
VERBOSE       = True
# ————————————————————

class ColorEvaluator:
    def __init__(self, git_ckpt_path):
        self.processor = AutoProcessor.from_pretrained(git_ckpt_path)
        config = AutoConfig.from_pretrained(git_ckpt_path)
        self.model = GitForCausalLM(config)
        ckpt = torch.load(os.path.join(git_ckpt_path, 'pytorch_model.bin'), map_location='cpu')
        self.model.load_state_dict(ckpt, strict=False)

        self.yes_token_id = self.processor.tokenizer.encode('yes')[1]
        self.no_token_id  = self.processor.tokenizer.encode('no')[1]

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.model.to(DEVICE)

    def evaluate_color(self, image_path, color_object_pairs, verbose=False):
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            return {'success': False, 'error': str(e), 'avg_score': 0.0, 'details': []}

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
        results, scores = [], []

        for color, obj, count in color_object_pairs:
            prompt = f"Are there {count} {color} {obj}s?" if count and count > 1 else f"Is this a {color} {obj}?"
            input_ids = self.processor(text=prompt, add_special_tokens=False).input_ids
            input_ids = [self.processor.tokenizer.cls_token_id] + input_ids
            input_ids = torch.tensor(input_ids).unsqueeze(0).to(DEVICE)

            logits = self.model(pixel_values=pixel_values, input_ids=input_ids).logits[:, -1]
            probs = torch.softmax(logits, dim=1)
            py = probs[0, self.yes_token_id].item()
            pn = probs[0, self.no_token_id].item()
            score = py / (py + pn) if (py + pn) > 0 else 0.0

            results.append({
                'color': color,
                'object': obj,
                'count': count,
                'prompt': prompt,
                'score': score,
                'is_match': score > 0.5
            })
            scores.append(score)

            if verbose:
                print(f"[{os.path.basename(image_path)}] '{prompt}' -> {score:.4f}")

        avg_score = float(np.mean(scores)) if scores else 0.0
        return {'success': True, 'avg_score': avg_score, 'details': results}

def parse_jsonl_file(jsonl_path):
    prompts = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            objects = [
                (obj.get('color', ''), obj.get('class', ''), obj.get('count', None))
                for obj in data.get('include', [])
            ]
            prompts.append({
                'prompt': data.get('prompt', ''),
                'objects': objects,
                'tag': data.get('tag', '')
            })
    return prompts

def evaluate_all_images(evaluator, base_dir, jsonl_file, image_pattern=IMAGE_PATTERN, verbose=VERBOSE):
    prompts = parse_jsonl_file(jsonl_file)
    print(f"加载了 {len(prompts)} 条 prompt。")

    subdirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    print(f"发现 {len(subdirs)} 个子目录。")

    summary = {
        "total_directories": len(subdirs),
        "prompts_count": len(prompts),
        "directories": {}
    }

    for subdir in tqdm(subdirs, desc="处理子目录"):
        subdir_path = os.path.join(base_dir, subdir)
        output_path = os.path.join(subdir_path, "git_evaluation.json")
        image_paths = sorted(glob.glob(os.path.join(subdir_path, image_pattern)))

        if not image_paths:
            print(f"⚠️ 跳过 {subdir}，未找到图像")
            continue

        try:
            idx = int(subdir) - 1
            prompt_idx = idx % len(prompts)
            prompt_info = prompts[prompt_idx]
        except:
            print(f"⚠️ 跳过 {subdir}，目录名无法转换为整数")
            continue

        if verbose:
            print(f"\n🔹 {subdir} 使用 prompt[{prompt_idx}]: {prompt_info['prompt']}")

        dir_result = {
            "directory": subdir,
            "prompt": prompt_info["prompt"],
            "tag": prompt_info["tag"],
            "prompt_index": prompt_idx,
            "images": {},
            "average_score": 0.0
        }

        total, count = 0.0, 0
        for img_path in image_paths:
            result = evaluator.evaluate_color(img_path, prompt_info["objects"], verbose=False)
            dir_result["images"][os.path.basename(img_path)] = result
            if result["success"]:
                total += result["avg_score"]
                count += 1

        if count > 0:
            dir_result["average_score"] = round(total / count, 4)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dir_result, f, ensure_ascii=False, indent=2)

        summary["directories"][subdir] = {
            "prompt": dir_result["prompt"],
            "tag": dir_result["tag"],
            "prompt_index": dir_result["prompt_index"],
            "average_score": dir_result["average_score"]
        }

    # 保存总汇
    summary_path = os.path.join(base_dir, "git_evaluation_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 所有评估完成，总结保存至：{summary_path}")

def main():
    parser = argparse.ArgumentParser(description="图像颜色属性批量评估器")
    parser.add_argument('--base_dir', required=True, help='图像目录根路径（包含子目录）')
    parser.add_argument('--jsonl_file', required=True, help='prompt JSONL 文件路径')
    parser.add_argument('--gpu', type=str, default="0", help='使用的 GPU ID（如 0、1）')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # ✅ 设置环境变量

    print(f"📦 初始化模型: {GIT_CKPT_PATH}")
    print(f"📂 基础路径: {args.base_dir}")
    evaluator = ColorEvaluator(GIT_CKPT_PATH)

    evaluate_all_images(
        evaluator=evaluator,
        base_dir=args.base_dir,
        jsonl_file=args.jsonl_file,
        image_pattern=IMAGE_PATTERN,
        verbose=VERBOSE
    )

if __name__ == "__main__":
    main()
