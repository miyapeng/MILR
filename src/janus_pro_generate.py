import argparse
import json
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pytorch_lightning import seed_everything
from janus.models import MultiModalityCausalLM, VLChatProcessor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_file", type=str, help="Path to JSONL metadata file.")
    parser.add_argument("--model", type=str, default="deepseek-ai/Janus-Pro-7B", help="Janus model name or path.")
    parser.add_argument("--outdir", type=str, default="/media/raid/workspace/miyapeng/Multimodal-LatentSeek/src/geneval_results/janus-pro", help="Directory to write results to.")
    parser.add_argument("--img_size", type=int, default=384, help="Generated image size.")
    parser.add_argument("--parallel_size", type=int, default=4, help="Number of parallel samples.")
    parser.add_argument("--image_token_num_per_image", type=int, default=576, help="Token length per image.")
    parser.add_argument("--cfg_weight", type=float, default=5.0, help="Classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()

@torch.inference_mode()
def generate_image(model, processor, prompt, parallel_size, cfg_weight, image_token_num_per_image, img_size):
    input_ids = processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.long).cuda()
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = processor.pad_id

    inputs_embeds = model.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.long).cuda()

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
    device = torch.device("cuda")
    seed_everything(opt.seed)

    processor = VLChatProcessor.from_pretrained(opt.model)
    model = MultiModalityCausalLM.from_pretrained(opt.model, trust_remote_code=True)
    model = model.to(torch.bfloat16).cuda().eval()

    with open(opt.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]

    for index, metadata in enumerate(metadatas):
        prompt = metadata["prompt"]
        conversation = [
            {"role": "<|User|>", "content": prompt},
            {"role": "<|Assistant|>", "content": ""},
        ]
        formatted_prompt = processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=processor.sft_format,
            system_prompt=""
        ) + processor.image_start_tag

        outpath = os.path.join(opt.outdir, f"{index:05d}")
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)

        print(f"[{index}/{len(metadatas)}] Generating for prompt: {prompt}")
        images = generate_image(
            model, processor, formatted_prompt,
            parallel_size=opt.parallel_size,
            cfg_weight=opt.cfg_weight,
            image_token_num_per_image=opt.image_token_num_per_image,
            img_size=opt.img_size,
        )

        for i, img in enumerate(images):
            Image.fromarray(img).save(os.path.join(sample_path, f"{i:04}.png"))
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as f:
            json.dump(metadata, f)



if __name__ == "__main__":
    args = parse_args()
    main(args)
