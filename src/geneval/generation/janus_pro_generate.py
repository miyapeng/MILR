import argparse
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_file", type=str, help="JSONL file containing prompts")
    parser.add_argument("--model_path", type=str, default="deepseek-ai/Janus-Pro-7B")
    parser.add_argument("--outdir", type=str, default="/media/raid/workspace/miyapeng/Multimodal-LatentSeek/src/geneval_results/janus-pro")
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--image_token_num_per_image", type=int, default=576)
    parser.add_argument("--parallel_size", type=int, default=16)
    parser.add_argument("--cfg_weight", type=float, default=5.0)
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser.parse_args()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def generate_image(index, prompt, model, processor, args):
    tokenizer = processor.tokenizer
    sft_format = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=[{"role": "<|User|>", "content": prompt}, {"role": "<|Assistant|>", "content": ""}],
        sft_format=processor.sft_format,
        system_prompt="",
    )
    full_prompt = sft_format + processor.image_start_tag
    input_ids = tokenizer.encode(full_prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((args.parallel_size * 2, len(input_ids)), dtype=torch.long).cuda()
    for i in range(args.parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = processor.pad_id

    inputs_embeds = model.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((args.parallel_size, args.image_token_num_per_image), dtype=torch.long).cuda()

    outputs = None
    for i in range(args.image_token_num_per_image):
        outputs = model.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=outputs.past_key_values if i != 0 else None
        )
        hidden_states = outputs.last_hidden_state
        logits = model.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        logits = logit_uncond + args.cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / args.temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)
        next_token = torch.cat([next_token.unsqueeze(dim=1)] * 2, dim=1).view(-1)
        img_embeds = model.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = model.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[args.parallel_size, 8, args.img_size // args.patch_size, args.img_size // args.patch_size]
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
    return dec


def main(opt):
    os.makedirs(opt.outdir, exist_ok=True)
    with open(opt.metadata_file) as f:
        metadatas = [json.loads(l) for l in f]

    processor = VLChatProcessor.from_pretrained(opt.model_path)
    model = AutoModelForCausalLM.from_pretrained(opt.model_path, trust_remote_code=True)
    model = model.to(torch.bfloat16).cuda().eval()

    for index, metadata in enumerate(tqdm(metadatas)):
        seed_everything(opt.seed)
        outpath = os.path.join(opt.outdir, f"{index:05}")
        os.makedirs(outpath, exist_ok=True)

        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        imgs = generate_image(index, metadata["prompt"], model, processor, opt)
        for i in range(min(opt.n_samples, imgs.shape[0])):
            Image.fromarray(imgs[i]).save(os.path.join(outpath, f"{i:05}.png"))


if __name__ == "__main__":
    args = parse_args()
    main(args)
