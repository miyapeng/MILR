import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
import argparse
import json
import random
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor

# Ensure reproducibility
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_file", type=str, help="Path to JSONL metadata file.")
    parser.add_argument("--model", type=str, default="deepseek-ai/Janus-Pro-7B", help="Janus model name or path.")
    parser.add_argument("--outdir", type=str, default="/media/raid/workspace/miyapeng/Multimodal-LatentSeek/src/geneval_results/janus_pro_think", help="Directory to write results to.")
    parser.add_argument("--parallel_size", type=int, default=4, help="Number of parallel images to generate.")
    parser.add_argument("--max_text_tokens", type=int, default=512, help="Max tokens for text enhancement.")
    parser.add_argument("--image_token_num", type=int, default=576, help="Number of image tokens per image.")
    parser.add_argument("--img_size", type=int, default=384, help="Generated image resolution.")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for decoding.")
    parser.add_argument("--cfg_weight", type=float, default=5.0, help="CFG guidance weight.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


@torch.inference_mode()
def generate_images(input_text: str,
                    model: MultiModalityCausalLM,
                    processor: VLChatProcessor,
                    device: torch.device,
                    parallel_size: int,
                    max_text_tokens: int,
                    image_token_num: int,
                    img_size: int,
                    patch_size: int,
                    cfg_weight: float):
    """
    Inline original_generation: returns a list of PIL Images of length parallel_size.
    """
    tokenizer = processor.tokenizer
    stop_words = {"</s>", "<|im_end|>", "<|endoftext|>", tokenizer.eos_token}

    # -----------------------------
    # Part 1: Text Enhancement
    # -----------------------------
    cot_prompt = (
        'You are asked to generate an image based on this prompt: "{}"\n'
        'Provide a brief, precise visualization of all elements in the prompt. Your description should:\n'
        '1. Include every object mentioned in the prompt\n'
        '2. Specify visual attributes (color, number, shape, texture) if specified in the prompt\n'
        '3. Clarify relationships (e.g., spatial) between objects if specified in the prompt\n'
        '4. Be concise (50 words or less)\n'
        "5. Focus only on what's explicitly stated in the prompt\n"
        '6. Do not elaborate beyond the attributes or relationships specified in the prompt\n'
        'Do not miss objects. Output your visualization directly without explanation:'
    )
    formatted = cot_prompt.format(input_text)
    conv = [{"role":"User","content":formatted}, {"role":"Assistant","content":""}]
    sys_prompt = 'You are a helpful assistant that receives an image prompt and generate a visualization.'
    sft = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conv, sft_format=processor.sft_format, system_prompt=sys_prompt)

    inputs = tokenizer([sft], return_tensors="pt", padding=True, add_special_tokens=True)
    current_ids = inputs.input_ids.to(device)
    text_ids = []

    for _ in range(max_text_tokens):
        out = model.language_model.model(current_ids, output_hidden_states=True)
        h = out.last_hidden_state[:, -1]
        if h.grad is not None: h.grad.zero_()
        # record hidden state
        # detach + grad
        h_det = h.detach().requires_grad_(True)
        logits = model.language_model.lm_head(h_det)
        nxt = torch.argmax(logits, dim=-1)
        token = tokenizer.decode(nxt.item(), skip_special_tokens=False)
        if token in stop_words:
            break
        text_ids.append(nxt.item())
        current_ids = torch.cat([current_ids, nxt.unsqueeze(0)], dim=-1)

    text_input_ids = current_ids.cpu()
    enhanced_text = tokenizer.decode(text_ids, skip_special_tokens=True)
    print(f"Enhanced text: {enhanced_text}")
    # -----------------------------
    # Part 2: Image Generation
    # -----------------------------
    image_prompt = f"{input_text}. {enhanced_text}"
    print(f"Image prompt: {image_prompt}")
    conv2 = [{"role":"User","content":image_prompt}, {"role":"Assistant","content":""}]
    sft2 = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conv2, sft_format=processor.sft_format, system_prompt="")
    prompt_ids = tokenizer([sft2], return_tensors="pt", padding=True, add_special_tokens=True).input_ids.to(device)
    # append image start
    img_tag_id = tokenizer.encode(processor.image_start_tag)[1]
    prompt_ids = torch.cat([prompt_ids, prompt_ids.new_full((1,1), img_tag_id)], dim=1)
    # expand for parallel + uncond
    prompt_ids = prompt_ids.repeat(parallel_size, 1)
    # cond/uncond inputs_embeds
    cond_emb = model.language_model.get_input_embeddings()(prompt_ids)
    pad_emb = model.language_model.get_input_embeddings()(prompt_ids.new_full((1,1), processor.pad_id))
    uncond = cond_emb.clone()
    uncond[:,1:-1] = pad_emb
    inputs_embeds = torch.repeat_interleave(cond_emb, 2, dim=0)
    inputs_embeds[1::2] = uncond
    # image token generation
    gen_tokens = torch.zeros((parallel_size, image_token_num), dtype=torch.long, device=device)
    past = None
    for i in range(image_token_num):
        out2 = model.language_model.model(
            inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past)
        past = out2.past_key_values
        h2 = out2.last_hidden_state[:, -1, :]
        logits2 = model.gen_head(h2)
        lc = logits2[0::2]; lu = logits2[1::2]
        logits2 = lu + cfg_weight*(lc-lu)
        probs2 = torch.softmax(logits2, dim=-1)
        nt = torch.multinomial(probs2, num_samples=1)
        gen_tokens[:, i] = nt.squeeze(-1)
        nt2 = nt.repeat(1,2).view(-1)
        emb2 = model.prepare_gen_img_embeds(nt2)
        inputs_embeds = emb2.unsqueeze(1)

    dec = model.gen_vision_model.decode_code(
        gen_tokens.to(torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0,2,3,1)
    dec = np.clip((dec+1)/2*255,0,255).astype(np.uint8)
    # return list of PIL images
    pil_imgs = [Image.fromarray(dec[j]) for j in range(parallel_size)]
    return pil_imgs, enhanced_text, image_prompt


def main():
    opt = parse_args()
    seed_all(opt.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = VLChatProcessor.from_pretrained(opt.model)
    model = MultiModalityCausalLM.from_pretrained(opt.model, trust_remote_code=True)
    model = model.to(torch.bfloat16).to(device).eval()

    with open(opt.metadata_file) as f:
        metas = [json.loads(line) for line in f]

    for idx, meta in enumerate(tqdm(metas, desc="Geneval")):
        imgs, enhanced_text, image_prompt = generate_images(
            input_text=meta['prompt'], model=model, processor=processor,
            device=device, parallel_size=opt.parallel_size,
            max_text_tokens=opt.max_text_tokens,
            image_token_num=opt.image_token_num,
            img_size=opt.img_size, patch_size=opt.patch_size,
            cfg_weight=opt.cfg_weight
        )
        outdir = os.path.join(opt.outdir, f"{idx:05d}")
        samples_dir = os.path.join(outdir, "samples")
        os.makedirs(samples_dir, exist_ok=True)
        for j, im in enumerate(imgs):
            im.save(os.path.join(samples_dir, f"{j:04}.png"))
        with open(os.path.join(outdir, "metadata.jsonl"), "w") as fp:
            json.dump(meta, fp)
        # 3) 保存增强文本和最终 prompt
        info = {
            "enhanced_text": enhanced_text,
            "image_prompt": image_prompt
        }
        with open(os.path.join(outdir, "prompts.json"), "w") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
