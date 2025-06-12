# Janus-Pro optimized generation: latent optimization for prompt (text) and/or image hidden states
# Provides three distinct branches: "text", "image", and "both".
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4" 

import torch
import numpy as np
import PIL.Image

from ori_generation_janus import original_generation
from janus.models import MultiModalityCausalLM, VLChatProcessor
from transformers import AutoModelForCausalLM
from rewards.reward import RewardModel

@torch.inference_mode()
def generate_image_from_prompt(
    mmgpt,
    vl_chat_processor,
    user_prompt: str,
    temperature: float = 1.0,
    cfg_weight: float = 5.0,
    image_token_num: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    save_path: str = None
):
    print("user_prompt:", user_prompt)
    
    # === 构造对话 prompt ===
    conversation = [
        {"role": "<|User|>", "content": user_prompt},
        {"role": "<|Assistant|>", "content": ""}
    ]
    prompt = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt=""
    ) + vl_chat_processor.image_start_tag

    # === Tokenize ===
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    parallel_size = 1  # 只生成一张图
    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id  # unconditional

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num), dtype=torch.int).cuda()

    outputs = None
    for i in range(image_token_num):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=outputs.past_key_values if i > 0 else None,
        )
        hidden_states = outputs.last_hidden_state  # [2, seq, hidden]

        logits = mmgpt.gen_head(hidden_states[:, -1, :])  # [2, vocab]
        logit_cond = logits[0::2]
        logit_uncond = logits[1::2]
        fused_logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)

        probs = torch.softmax(fused_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]
        generated_tokens[:, i] = next_token.squeeze(-1)

        next_token = torch.cat([next_token, next_token], dim=1).view(-1)  # [2]
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(1)

    # === 图像解码 ===
    decoded = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size]
    )
    decoded = decoded.detach().to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    decoded = np.clip((decoded + 1) / 2 * 255, 0, 255).astype(np.uint8)

    image = PIL.Image.fromarray(decoded[0])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image.save(save_path)

    return image

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
def optimized_generation(
    reward_model,
    image: PIL.Image,
    data,
    model: MultiModalityCausalLM,                          # MultiModalityCausalLM
    vl_chat_processor: VLChatProcessor,                     # VLChatProcessor
    device: torch.device,
    # input_text: str,                         
    text_hidden_states_list: list,   # List[Tensor] from CoT
    text_final_input_ids: torch.Tensor,
    image_hidden_states_list: list,  # List[Tensor] from image gen
    image_prompt_ids: torch.Tensor,
    generated_image_tokens: torch.Tensor,
    start_index=0,
    max_text_steps=10,
    max_image_steps=10,
    lr=0.03,
    grad_clip=None,
    k=0.1,
    reward_threshold=0.1,
    max_text_tokens=512,
    image_token_num=576,
    img_size=384,
    patch_size=16,
    optimize_mode="both",  # must be one of: "text", "image", "both"
):
    """
    Optimize latent states for Janus-Pro image generation with three explicit branches:
      - "text": only optimize CoT text hidden states, then generate image once
      - "image": generate image from original CoT and only optimize image hidden states
      - "both": perform text optimization first, then image optimization

    Returns:
        final_image: PIL.Image
        reward_history: List[float]
    """
    # Validate mode
    assert optimize_mode in ("text", "image", "both"), "optimize_mode must be 'text', 'image', or 'both'"

    tokenizer = vl_chat_processor.tokenizer
    stop_words.append(tokenizer.eos_token)
    reward_history = []
    # start from original image tokens
    initial_reward = reward_model.get_reward(image, data)
    print(f"-- Initial Image Reward: {initial_reward}")
    reward_history.append(initial_reward)
    current_reward = initial_reward
    tokens = generated_image_tokens.clone()

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

    formatted_cot_prompt = cot_prompt.format(data['prompt'])
    conversation = [{"role": "User", "content": formatted_cot_prompt}, {"role": "Assistant", "content": ""}]
    system_prompt = 'You are a helpful assistant that receives an image prompt and generate a visualization of the prompt.'
    sft_prompt = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation, sft_format=vl_chat_processor.sft_format, system_prompt=system_prompt
    )

    # === Text-only branch ===
    if optimize_mode == "text":
        total = len(text_hidden_states_list)
        update_length = min(int(k * total), total)
        if update_length <= 0:
            print("Update Length Zero!!!")
            return None, reward_history, total, 0, update_length

        # 1. 构建前缀 base_input_ids
        inputs = tokenizer([sft_prompt], return_tensors="pt", padding=True, padding_side="right", add_special_tokens=True).to(device)
        base_input_ids = inputs.input_ids.clone()

        # 2. 构建 base_ids（前缀 + start_index），作为优化起点
        base_ids = text_final_input_ids[:, :base_input_ids.shape[-1] + start_index].to(device)
        original_seq = base_ids[0].tolist()

        # 3. 构建需要优化的隐藏状态
        optimized_states = torch.nn.Parameter(torch.stack(
            [s.clone().detach().requires_grad_(True)
            for s in text_hidden_states_list[start_index:min(start_index + update_length, len(text_hidden_states_list))]
            ])
        )
        optimizer = torch.optim.Adam([optimized_states], lr=lr)

        new_img = None
        tokens = None

        for i in range(max_text_steps):
            if current_reward > reward_threshold:
                break

            optimizer.zero_grad()

            logits = model.language_model.lm_head(optimized_states)
            probs = torch.softmax(logits, dim=-1) + 1e-8
            next_token_ids = torch.argmax(probs, dim=-1)                   
            next_token_ids = next_token_ids.squeeze(-1)
            log_pi = torch.log(probs[torch.arange(update_length), 0, next_token_ids] + 1e-10)

            loss = - current_reward * log_pi.sum()
            print(f"-- Text Branch Loss: {loss.item()}")
            loss.backward(retain_graph=True)
            if grad_clip:
                torch.nn.utils.clip_grad_norm_([optimized_states], grad_clip)
            optimizer.step()

            # === 更新序列并重新评估 reward ===
            generated_seq = []
            generated_seq.extend(original_seq)
            with torch.no_grad():
                updated_token_ids = torch.argmax(model.language_model.lm_head(optimized_states), dim=-1)
                updated_token_ids = updated_token_ids.squeeze(-1)
                generated_seq.extend(updated_token_ids.tolist())

                # 拼接为新的 input_ids
                updated_input_ids = torch.cat([base_ids, updated_token_ids.unsqueeze(0)], dim=-1)

                # 后续 token 自动生成直到 eos 或 max_text_tokens
                cnt = 0
                while True:
                    outputs = model.language_model.model(updated_input_ids, output_hidden_states=True)
                    last_hidden = outputs[0][:, -1]
                    logits = model.language_model.lm_head(last_hidden)
                    next_token_id = torch.argmax(logits, dim=-1)
                    token_str = tokenizer.decode(next_token_id.item(), skip_special_tokens=False)

                    generated_seq.append(next_token_id.item())
                    updated_input_ids = torch.cat([updated_input_ids, next_token_id.unsqueeze(0)], dim=-1)
                    cnt += 1
                    if token_str in stop_words:
                        break
                    if cnt > max_text_tokens:
                        break

            new_token_ids = generated_seq[len(original_seq):]
            new_generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
            print(f"New Optimized Answer: {new_generated_text}")
            del outputs, last_hidden, next_token_id, token_str
            del logits, updated_token_ids, updated_input_ids
            torch.cuda.empty_cache()

            image_gen_prompt = f"{data['prompt']}. {new_generated_text}"
            # 生成图像 → 评估 reward
            save_path = f"/media/raid/workspace/miyapeng/Multimodal-LatentSeek/src/opt_images/optimized_image_{i}.png"
            new_img = generate_image_from_prompt(model, vl_chat_processor, image_gen_prompt, save_path=save_path)
            new_reward = reward_model.get_reward(new_img, data)
            print(f"-- Text Branch New Reward: {new_reward}")
            reward_history.append(new_reward)
            current_reward = new_reward

        # 最终返回（你可以选择返回 new_img、tokens 等）
        return new_img, reward_history, total, len(generated_seq), update_length

    # Branch: image-only
    elif optimize_mode == "image":
        total2 = len(image_hidden_states_list)
        img_upd = min(int(k * total2), total2)
        if img_upd > 0:
            i_states = torch.stack([
                s.clone().detach().requires_grad_(True)
                for s in image_hidden_states_list[:img_upd]
            ])
            i_states = torch.nn.Parameter(i_states)
            opt2 = torch.optim.Adam([i_states], lr=lr)
            for _ in range(max_image_steps):
                if current_reward > reward_threshold:
                    break
                opt2.zero_grad()
                logits2 = model.gen_head(i_states)
                lps = torch.log_softmax(logits2, dim=-1)
                nxt = torch.argmax(lps, dim=-1)
                seq = torch.cat([tokens[:start_index], nxt.cpu()])
                r2 = reward_model.get_reward(image, data)
                loss2 = -r2
                loss2.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_([i_states], grad_clip)
                opt2.step()
                tokens = seq
                current_reward = r2
                reward_history.append(r2)
    # Branch: both
    else:  # "both"
        # first text, then image
        # reuse text branch logic to update text_hidden_states_list and tokens
        # then reuse image branch on updated tokens
        # Text optimization
        total = len(text_hidden_states_list)
        text_upd = min(int(k * total), total)
        if text_upd > 0:
            states = torch.stack([
                s.clone().detach().requires_grad_(True)
                for s in text_hidden_states_list[start_index:start_index + text_upd]
            ])
            states = torch.nn.Parameter(states)
            opt = torch.optim.Adam([states], lr=lr)
            base_ids = text_final_input_ids[:, :start_index + 1].to(device)
            for _ in range(max_text_steps):
                if current_reward > reward_threshold:
                    break
                opt.zero_grad()
                seq = base_ids[0].tolist()
                for h in states:
                    lid = torch.argmax(model.language_model.lm_head(h), dim=-1).item()
                    seq.append(lid)
                toks = generate_image_from_text(seq)
                r = reward_model.get_reward(question, toks)
                (-r).backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_([states], grad_clip)
                opt.step()
                tokens = toks; current_reward = r; reward_history.append(r)
        # Image optimization on final tokens
        total2 = len(image_hidden_states_list)
        img_upd = min(int(k * total2), total2)
        if img_upd > 0:
            i_states = torch.stack([
                s.clone().detach().requires_grad_(True)
                for s in image_hidden_states_list[:img_upd]
            ])
            i_states = torch.nn.Parameter(i_states)
            opt2 = torch.optim.Adam([i_states], lr=lr)
            for _ in range(max_image_steps):
                if current_reward > reward_threshold:
                    break
                opt2.zero_grad()
                logits2 = model.gen_head(i_states)
                lps = torch.log_softmax(logits2, dim=-1)
                nxt = torch.argmax(lps, dim=-1)
                seq2 = torch.cat([tokens[:start_index], nxt.cpu()])
                r2 = reward_model.get_reward(question, seq2)
                (-r2).backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_([i_states], grad_clip)
                opt2.step()
                tokens = seq2; current_reward = r2; reward_history.append(r2)

    # Decode final tokens to image
    final_img = vl_chat_processor.decode_image(tokens.numpy())
    return final_img, reward_history

model_path = "deepseek-ai/Janus-Pro-7B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
input_text = "a photo of a purple suitcase and an orange pizza"
data = {"tag": "color_attr", "include": [{"class": "suitcase", "count": 1, "color": "purple"}, {"class": "pizza", "count": 1, "color": "orange"}], "prompt": "a photo of a purple suitcase and an orange pizza"}

reward_model = RewardModel(
    model_path="/media/raid/workspace/miyapeng/Multimodal-LatentSeek/src/rewards/<OBJECT_DETECTOR_FOLDER>",
    object_names_path="/media/raid/workspace/miyapeng/Multimodal-LatentSeek/src/rewards/object_names.txt",
    options={"clip_model": "ViT-L-14"}
)

answer, text_hidden_states_list, text_final_input_ids, image_hidden_states_list, image_prompt_ids, generated_image_tokens = original_generation(input_text, vl_gpt, vl_chat_processor, torch.device("cuda"))
torch.cuda.empty_cache()
new_img, reward_history, total, generated_seq, update_length = optimized_generation(reward_model, answer, data, vl_gpt, vl_chat_processor, torch.device("cuda"), text_hidden_states_list, text_final_input_ids, image_hidden_states_list, image_prompt_ids, generated_image_tokens, optimize_mode="text")