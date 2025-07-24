from numpy.random import f
import torch
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
import PIL.Image
from PIL import Image
import random
from typing import List, Dict, Tuple
import argparse
import copy

from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor

def seed_all(seed):
    """Set all random seeds to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # The two lines below are known to cause slowdowns, but ensure reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_all(42)
def generate_image(prompt_inputs: Dict[str, torch.Tensor],
                   device: torch.device,
                   tokenizer,
                   parallel_size: int = 1,
                   model: MultiModalityCausalLM = None,
                   image_token_num: int = 576,
                   cfg_weight: float = 5.0,
                   temperature: float = 1.0,
                   img_size: int = 384,
                   patch_size: int = 16,
                   ):
    """
    Generate only one image.

    Args:
        prompt_inputs (Dict[str, torch.Tensor]): The input tensors for the image generation.
        device (torch.device): The device to run the model on.
        tokenizer: The tokenizer used for encoding the prompt.
        parallel_size (int): The number of images to generate in parallel.
        model (MultiModalityCausalLM): The loaded MultiModalityCausalLM model.
        image_token_num (int): The number of image tokens to generate.
        cfg_weight (float): The weight for Classifier-Free Guidance.
        temperature (float): The sampling temperature for image token generation.
        img_size (int): The size of the generated image (height and width).
        patch_size (int): The size of the patches used in the image generation model.

    Returns:
        visual_img (np.ndarray): The generated image as a numpy array.
        image_hidden_states_step (List[torch.Tensor]): Hidden states for each token in the image generation phase.
        generated_image_tokens (torch.Tensor): The tensor of generated image tokens.
        image_prompt_ids (torch.Tensor): The input IDs of the prompt used for image generation.
    """
    image_prompt_ids = prompt_inputs["input_ids"].to(device)
    attention_mask = prompt_inputs["attention_mask"].to(device)

    image_start_token_id = tokenizer.encode(vl_chat_processor.image_start_tag)[1]
    image_prompt_ids = torch.cat([image_prompt_ids, image_prompt_ids.new_full((image_prompt_ids.size(0), 1), image_start_token_id)], dim=1)
    attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.size(0), 1))], dim=1)

    image_prompt_ids = image_prompt_ids.repeat(parallel_size, 1)
    attention_mask = attention_mask.repeat(parallel_size, 1)

    cond_inputs_embeds = model.language_model.get_input_embeddings()(image_prompt_ids)
    pad_input_embeds = model.language_model.get_input_embeddings()(image_prompt_ids.new_full((1, 1), vl_chat_processor.pad_id))

    uncond_inputs_embeds = cond_inputs_embeds.clone()
    uncond_inputs_embeds[:,1:-1] = pad_input_embeds

    inputs_embeds_img = torch.repeat_interleave(cond_inputs_embeds, 2, dim=0)
    inputs_embeds_img[1::2] = uncond_inputs_embeds

    attention_mask_img = torch.repeat_interleave(attention_mask, 2, dim=0)
    attention_mask_img[1::2] = torch.ones_like(attention_mask_img[1::2])

    generated_image_tokens = torch.zeros((parallel_size, image_token_num), dtype=torch.int, device=device)
    image_hidden_states_step = []
    current_img_embeds = inputs_embeds_img
    past_key_values = None

    for k in range(image_token_num):
        with torch.no_grad():
            outputs = model.language_model.model(
                inputs_embeds=current_img_embeds,
                use_cache=True,
                past_key_values=past_key_values,
                attention_mask=attention_mask_img
            )
            past_key_values = outputs.past_key_values
            hidden_states = outputs.last_hidden_state

            image_hidden_states_step.append(hidden_states[:, -1, :].clone().cpu())

            logits = model.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)

            probs = torch.softmax(logits/temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_image_tokens[:, k] = next_token.squeeze(dim=-1)
            next_token = next_token.repeat(1, 2).view(-1)
            current_img_embeds = model.prepare_gen_img_embeds(next_token).unsqueeze(1)
            attention_mask_img = torch.cat([attention_mask_img, attention_mask_img.new_ones((attention_mask_img.size(0), 1), dtype=torch.int)], dim=1)

    with torch.no_grad():
        dec = model.gen_vision_model.decode_code(generated_image_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])

    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec
    return visual_img, image_hidden_states_step, generated_image_tokens.cpu(), image_prompt_ids.cpu()


def original_generation(
    input_text: str,
    model: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    device: torch.device,
    parallel_size: int = 1,
    temperature: float = 1.0,
    cfg_weight: float = 5.0,
    max_reasoning_steps: int = 4,
    image_token_num: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
) -> Tuple[PIL.Image.Image, List[PIL.Image.Image], List[torch.Tensor], torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Generates a sequence of intermediate images as reasoning steps and a final image using the Janus-Pro model.
    Each reasoning image is generated one by one and used as prompt for the next step.
    Returns intermediate hidden states for each image generation phase.

    Args:
        input_text: The user's original text prompt.
        model: The loaded MultiModalityCausalLM model.
        vl_chat_processor: The loaded VLChatProcessor, which includes the tokenizer.
        device: The computation device (e.g., torch.device('cuda')).
        parallel_size: the number of generated images in parallel.
        temperature: The sampling temperature for image token generation.
        cfg_weight: The weight for Classifier-Free Guidance.
        max_reasoning_steps: Number of intermediate reasoning images to generate.
        image_token_num: The number of image tokens to generate.
        img_size: The size of the generated image (height and width).
        patch_size: The size of the patches used in the image generation model.

    Returns:
        A tuple containing:
        - answer (PIL.Image.Image): The final generated image.
        - reasoning_images (List[PIL.Image.Image]): List of intermediate reasoning images.
        - image_hidden_states_list (List[torch.Tensor]): Hidden states for each token in the final image generation phase.
        - image_prompt_ids (torch.Tensor): The input IDs of the prompt used for image generation.
        - reasoning_hidden_states (List[torch.Tensor]): Hidden states for each reasoning image step.
        - generated_image_tokens (torch.Tensor): The tensor of generated image tokens for the final image.
        - image_gen_prompt (str): The prompt used for image generation.
    """
    tokenizer = vl_chat_processor.tokenizer

    reasoning_images = []
    reasoning_hidden_states = []

    ##### Image Reasoning Path #####
    for step in range(max_reasoning_steps):
        # Build conversation with previous images
        if step == 0:
            image_gen_prompt = f"{input_text} (Step {step+1} of reasoning)"
            img_gen_conversation = [{"role": "User", "content": image_gen_prompt}, {"role": "Assistant", "content": ""}]
        else:
            # Include previous reasoning images in the conversation
            conversation_content = [{"type": "text", "text": f"{input_text} (Step {step+1} of reasoning, building on previous steps)"}]
            
            # Add all previous reasoning images to the conversation
            for _, prev_img in enumerate(reasoning_images):
                conversation_content.append({"type": "image", "image": prev_img})
            
            img_gen_conversation = [
                {"role": "User", "content": conversation_content}, 
                {"role": "Assistant", "content": ""}
            ]

        sft_image_prompt = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=img_gen_conversation, sft_format=vl_chat_processor.sft_format, system_prompt=""
        )

        prompt_inputs = tokenizer(
            text=[sft_image_prompt], return_tensors="pt", padding=True, padding_side="right", add_special_tokens=True
        )
        visual_img, image_hidden_states_step, generated_image_tokens, image_prompt_ids = generate_image(
            prompt_inputs=prompt_inputs,
            device=device,
            tokenizer=tokenizer,
            parallel_size=parallel_size,
            model=model,
            image_token_num=image_token_num,
            cfg_weight=cfg_weight,
            temperature=temperature,
            img_size=img_size,
            patch_size=patch_size
        )

        reasoning_images.append(Image.fromarray(visual_img[0]))
        reasoning_hidden_states.append(image_hidden_states_step)

    ######## Final Image Generation ########

    # using all reasoning images as context
    conversation_content = [{"type": "text", "text": f"{input_text} (Final image, incorporating all reasoning steps)"}]
    
    # Add all reasoning images to the final conversation
    for i, reasoning_img in enumerate(reasoning_images):
        conversation_content.append({"type": "image", "image": reasoning_img})
    
    img_gen_conversation = [
        {"role": "User", "content": conversation_content}, 
        {"role": "Assistant", "content": ""}
    ]
    
    image_gen_prompt = f"{input_text} (Final image, incorporating all reasoning steps)"
    sft_image_prompt = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=img_gen_conversation, sft_format=vl_chat_processor.sft_format, system_prompt=""
    )

    prompt_inputs = tokenizer(
        text=[sft_image_prompt], return_tensors="pt", padding=True, padding_side="right", add_special_tokens=True
    )
    visual_img, image_hidden_states_list, generated_image_tokens, image_prompt_ids = generate_image(
        prompt_inputs=prompt_inputs,
        device=device,
        tokenizer=tokenizer,
        parallel_size=parallel_size,
        model=model,
        image_token_num=image_token_num,
        cfg_weight=cfg_weight,
        temperature=temperature,
        img_size=img_size,
        patch_size=patch_size
    )


    answer = Image.fromarray(visual_img[0])

    return answer, reasoning_images, image_hidden_states_list, image_prompt_ids.cpu(), reasoning_hidden_states, generated_image_tokens.cpu(), image_gen_prompt

model_path = "/home/plm/Janus-Pro-7B/"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
     model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
input_text = "a photo of two toothbrushes."
answer, reasoning_images, image_hidden_states_list, image_prompt_ids, reasoning_hidden_states, generated_image_tokens, image_gen_prompt = original_generation(
    input_text=input_text,
    model=vl_gpt,
    vl_chat_processor=vl_chat_processor,
    device=torch.device("cuda"),
    parallel_size=1,
    temperature=1.0,
    cfg_weight=5.0,
    max_reasoning_steps=4,
    image_token_num=576,
    img_size=384,
    patch_size=16,
)
print(f"Generated image size: {answer.size}")
