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

def original_generation(
    input_text: str,
    model: MultiModalityCausalLM, # Should be of type MultiModalityCausalLM
    vl_chat_processor: VLChatProcessor,       # Should be of type VLChatProcessor
    optimize_mode: str,
    device: torch.device,
    parallel_size: int = 1,
    temperature: float = 1.0,
    cfg_weight: float = 5.0,
    max_text_tokens: int = 512,
    image_token_num: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
) -> Tuple[PIL.Image.Image, List[torch.Tensor], torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Generates an image using the Janus-Pro model with text enhancement and returns intermediate hidden states for both text and image generation phases.

    This function follows the logic of enhancing a text prompt using a Chain-of-Thought (CoT)
    approach, capturing the hidden states during text generation, and then using the enhanced
    prompt to generate an image, while also capturing the hidden states of the image generation process.

    Args:
        input_text: The user's original text prompt.
        model: The loaded MultiModalityCausalLM model.
        vl_chat_processor: The loaded VLChatProcessor, which includes the tokenizer.
        device: The computation device (e.g., torch.device('cuda')).
        parallel_size: the number of generated images in parallel.
        temperature: The sampling temperature for image token generation.
        cfg_weight: The weight for Classifier-Free Guidance.
        max_text_tokens: The maximum number of tokens to generate during the text enhancement phase.
        image_token_num: The number of image tokens to generate.
        img_size: The size of the generated image (height and width).
        patch_size: The size of the patches used in the image generation model.

    Returns:
        A tuple containing:
        - answer (PIL.Image.Image): The final generated image.
        - text_hidden_states_list (List[torch.Tensor]): A list of hidden states for each token generated during the **text** enhancement phase.
        - text_final_input_ids (torch.Tensor): The final input IDs after **text** enhancement.
        - image_hidden_states_list (List[torch.Tensor]): A list of hidden states for each token generated during the **image** generation phase.
        - image_prompt_ids (torch.Tensor): The input IDs of the prompt used for **image** generation.
        - generated_image_tokens (torch.Tensor): The tensor of generated **image** tokens.
    """
    tokenizer = vl_chat_processor.tokenizer
    stop_words = {"</s>", "<|im_end|>", "<|endoftext|>", tokenizer.eos_token}

    # The CoT prompt is now hardcoded inside the function.
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

    # ========================================================================
    # Part 1: Text Enhancement (Semantic-CoT Generation) & Hidden State Extraction
    # ========================================================================

    formatted_cot_prompt = cot_prompt.format(input_text)
    conversation = [{"role": "User", "content": formatted_cot_prompt}, {"role": "Assistant", "content": ""}]
    system_prompt = 'You are a helpful assistant that receives an image prompt and generate a visualization of the prompt.'
    sft_prompt = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation, sft_format=vl_chat_processor.sft_format, system_prompt=system_prompt
    )

    inputs = tokenizer(
        [sft_prompt], return_tensors="pt", padding=True, padding_side="right", add_special_tokens=True
    )
    current_input_ids = inputs.input_ids.to(device)

    text_hidden_states_list = []
    generated_text_ids = []

    for _ in range(max_text_tokens):
        # generate normal outputs
        with torch.no_grad():
            outputs = model.language_model.model(current_input_ids, output_hidden_states=True)
        last_hidden_state = outputs[0][:, -1]  # [B, hidden_dim]
        text_hidden_states_list.append(last_hidden_state.clone())
        # detach + requires_grad
        last_hidden_state = last_hidden_state.detach()
        last_hidden_state.requires_grad = True
        if last_hidden_state.grad is not None:
            last_hidden_state.grad.zero_()
        # generate token
        with torch.no_grad():
            logits = model.language_model.lm_head(last_hidden_state)
            next_token_id = torch.argmax(logits, dim=-1)  # [1, 1]
            new_token = tokenizer.decode(next_token_id.item(), skip_special_tokens=False)
            generated_text_ids.append(next_token_id.item())
            # check if finished
            if new_token in stop_words:
                break

            current_input_ids = torch.cat([current_input_ids, next_token_id.unsqueeze(0)], dim=-1)

    # final answer
    text_final_input_ids = current_input_ids.clone().clone().cpu()
    enhanced_text = tokenizer.decode(generated_text_ids, skip_special_tokens=True)
    print(enhanced_text)

    # ========================================================================
    # Part 2: Image Generation & Hidden State Extraction
    # ========================================================================
    if optimize_mode == "image":
        image_gen_prompt = f"{input_text}"
    else:
        image_gen_prompt = f"{input_text}. {enhanced_text}"

    img_gen_conversation = [{"role": "User", "content": image_gen_prompt}, {"role": "Assistant", "content": ""}]
    sft_image_prompt = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=img_gen_conversation, sft_format=vl_chat_processor.sft_format, system_prompt=""
    )
    print(f"Prompt: {sft_image_prompt}\nSemantic-CoT: {enhanced_text}")

    prompt_inputs = tokenizer(
        text=[sft_image_prompt], return_tensors="pt", padding=True, padding_side="right", add_special_tokens=True
    )
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
    image_hidden_states_list = []
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
            
            image_hidden_states_list.append(hidden_states[:, -1, :].clone().cpu())
            
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

    # os.makedirs('generated_samples', exist_ok=True)
    # for i in range(parallel_size):
    #     save_path = os.path.join('generated_samples', "img_{}.jpg".format(i))
    #     PIL.Image.fromarray(visual_img[i]).save(save_path)
    answer = Image.fromarray(visual_img[0])

    return answer, text_hidden_states_list, text_final_input_ids, image_hidden_states_list, inputs_embeds_img.cpu(), image_gen_prompt