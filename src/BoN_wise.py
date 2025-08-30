#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from janus.models import MultiModalityCausalLM, VLChatProcessor
from rewards.gpt4o_reward import GPT4oReward


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", type=str, help="Path to JSON file (array) with fields: Prompt, prompt_id, ...")
    parser.add_argument("--model", type=str, default="deepseek-ai/Janus-Pro-7B", help="Janus model name or path.")
    parser.add_argument("--outdir", type=str, required=True, help="Directory to write results to (will create samples/).")
    parser.add_argument("--img_size", type=int, default=384, help="Generated image size.")
    parser.add_argument("--parallel_size", type=int, default=10, help="Number of parallel samples (N for BoN).")
    parser.add_argument("--image_token_num_per_image", type=int, default=576, help="Token length per image.")
    parser.add_argument("--cfg_weight", type=float, default=5.0, help="Classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    # GPT-4o reward
    parser.add_argument("--gpt4o_api_key", type=str, required=True, help="API key for GPT-4o reward.")
    parser.add_argument("--gpt4o_model", type=str, default="gpt-4o-2024-11-20", help="GPT-4o model name.")
    return parser.parse_args()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.inference_mode()
def generate_image(model, processor, prompt, parallel_size, cfg_weight, image_token_num_per_image, img_size):
    input_ids = processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids).cuda()

    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.long, device=input_ids.device)
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = processor.pad_id

    inputs_embeds = model.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.long, device=input_ids.device)

    outputs = None
    for i in range(image_token_num_per_image):
        outputs = model.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=outputs.past_key_values if outputs is not None else None,
        )
        hidden_states = outputs.last_hidden_state
        logits = model.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(-1)
        next_token = torch.cat([next_token, next_token], dim=1).view(-1)
        img_embeds = model.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(1)

    decoded = model.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size // 16, img_size // 16],
    )
    decoded = decoded.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    decoded = np.clip((decoded + 1) / 2 * 255, 0, 255).astype(np.uint8)
    return decoded


def main(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(opt.seed)

    # Load model & processor
    processor = VLChatProcessor.from_pretrained(opt.model)
    model = MultiModalityCausalLM.from_pretrained(opt.model, trust_remote_code=True)
    model = model.to(torch.bfloat16 if device.type == "cuda" else torch.float32).to(device).eval()

    # Reward model: GPT-4o
    reward_model = GPT4oReward(
        api_key=opt.gpt4o_api_key,
        model=opt.gpt4o_model,
    )

    # Load JSON array
    with open(opt.json_file, "r", encoding="utf-8") as f:
        examples = json.load(f)
        assert isinstance(examples, list), "JSON must be a list of objects."

    # Prepare output dirs
    base_path = opt.outdir
    samples_dir = os.path.join(base_path, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    # Optional: log scores
    results_log = os.path.join(base_path, "results.jsonl")
    log_fp = open(results_log, "a", encoding="utf-8")

    for example in tqdm(examples, desc="BoN"):
        prompt_text = example.get("Prompt", "")
        prompt_id = example.get("prompt_id", None)
        if prompt_id is None:
            # fallback: index if missing; but better to require prompt_id
            raise ValueError("Each JSON item must contain 'prompt_id'.")

        # Build Janus chat prompt
        conversation = [
            {"role": "<|User|>", "content": prompt_text},
            {"role": "<|Assistant|>", "content": ""},
        ]
        formatted_prompt = processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=processor.sft_format,
            system_prompt=""
        ) + processor.image_start_tag

        # Generate N images
        images = generate_image(
            model, processor, formatted_prompt,
            parallel_size=opt.parallel_size,
            cfg_weight=opt.cfg_weight,
            image_token_num_per_image=opt.image_token_num_per_image,
            img_size=opt.img_size,
        )

        # Select best by max reward
        best_score = None
        best_img = None

        for i, img_np in enumerate(images):
            img_pil = Image.fromarray(img_np)
            # 计算分数（传整个 example，里边包含 Prompt/Category 等上下文）
            score = reward_model.get_reward(img_pil, example)

            if (best_score is None) or (score > best_score):
                best_score = score
                best_img = img_pil

        # Save best image as <prompt_id>.png
        filename = f"{prompt_id}.png"
        save_path = os.path.join(samples_dir, filename)
        best_img.save(save_path)

        # log
        log_fp.write(json.dumps({
            "prompt_id": prompt_id,
            "prompt": prompt_text,
            "best_score": float(best_score),
            "save_path": save_path
        }, ensure_ascii=False) + "\n")

    log_fp.close()
    print(f"Done. Best-of-N images saved under: {samples_dir}")
    print(f"Scores logged to: {results_log}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
