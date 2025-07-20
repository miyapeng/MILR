# Janus-Pro optimized generation: latent optimization for prompt (text) and/or image hidden states
# Provides three distinct branches: "text", "image", and "both".
import os

import torch
import numpy as np
import PIL.Image
import json

from process import save_image_and_metadata
from ori_generation_janus import original_generation
from janus.models import MultiModalityCausalLM, VLChatProcessor
from transformers import AutoModelForCausalLM

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
    image_prompt_embed: torch.Tensor,
    ori_image_prompt,
    start_index=0,
    max_text_steps=10,
    max_image_steps=10,
    max_both_steps=10,
    lr=0.01,
    grad_clip=None,
    text_k=0.1,
    image_k=0.01,
    reward_threshold=-0.1,
    max_text_tokens=512,
    image_token_num=576,
    img_size=384,
    patch_size=16,
    cfg_weight = 5.0,
    temperature = 1.0,
    optimize_mode="text",  # must be one of: "text", "image", "both"
    save_base_path: str = None,
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
        update_length = min(int(text_k * total), total)
        # 1. 构建前缀 base_input_ids
        inputs = tokenizer([sft_prompt], return_tensors="pt").to(device)
        base_input_ids = inputs.input_ids.clone()
        
        if update_length <= 0:
            print("Update Length Zero!!!")
            return None, reward_history, total, 0, update_length

        # 2. 构建 base_ids（前缀 + start_index），作为优化起点
        original_seq = []
        # the prompt
        original_seq.extend(text_final_input_ids[0][len(base_input_ids[-1]): len(base_input_ids[-1]) + start_index])

        # 3. 构建需要优化的隐藏状态
        optimized_states = torch.nn.Parameter(torch.stack(
            [s.clone().detach().requires_grad_(True)
            for s in text_hidden_states_list[start_index:min(start_index + update_length, len(text_hidden_states_list))]
            ])
        )
        optimizer = torch.optim.Adam([optimized_states], lr=lr)

        input_ids = text_final_input_ids[:, : len(base_input_ids[-1]) + start_index]
        base_input_ids = input_ids.clone()
        new_img = None
        generated_seq = []
        for i in range(max_text_steps):
            input_ids = base_input_ids.clone().to(device)
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
                updated_prob = torch.softmax(model.language_model.lm_head(optimized_states), dim=-1) + 1e-8
                updated_token_ids = torch.argmax(updated_prob, dim=-1)
                updated_token_ids = updated_token_ids.squeeze(-1)
                generated_seq.extend(updated_token_ids.tolist())

                # 拼接为新的 input_ids
                updated_input_ids = torch.cat([input_ids, updated_token_ids.unsqueeze(0)], dim=-1)

            with torch.no_grad():
                # 后续 token 自动生成直到 eos 或 max_text_tokens
                cnt = 0
                while True:
                    outputs = model.language_model.model(updated_input_ids, output_hidden_states=True)
                    last_hidden = outputs[0][:, -1]
                    logits = model.language_model.lm_head(last_hidden)
                    next_token_id = torch.argmax(logits, dim=-1)
                    token_str = tokenizer.decode(next_token_id.item())

                    generated_seq.append(next_token_id.item())
                    updated_input_ids = torch.cat([updated_input_ids, next_token_id.unsqueeze(0)], dim=-1)
                    cnt += 1
                    if token_str in stop_words:
                        break
                    if cnt > max_text_tokens:
                        break

            # new_token_ids = generated_seq[len(original_seq):]
            # new_generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
            # print(f"New Optimized Answer: {new_generated_text}")
            del outputs, last_hidden, next_token_id, token_str
            del logits, updated_token_ids, updated_input_ids
            torch.cuda.empty_cache()
            new_generated_text = tokenizer.decode(generated_seq,skip_special_tokens=True)
            print(f"New Optimized Answer: {new_generated_text}")

            image_gen_prompt = f"{data['prompt']}. {new_generated_text}"
            # 生成图像 → 评估 reward
            
            if save_base_path is not None:
                save_dir = os.path.join(save_base_path)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"optimized_image_{i}.png")

            new_img = generate_image_from_prompt(model, vl_chat_processor, image_gen_prompt, save_path=save_path)

            new_reward = reward_model.get_reward(new_img, data)
            if save_base_path is not None:
                os.makedirs(save_base_path, exist_ok=True)
                trace_file = os.path.join(save_base_path, "text_trace.jsonl")
                with open(trace_file, "a", encoding="utf-8") as f:
                    json.dump({
                        "step": i,
                        "generated_text": new_generated_text,
                        "image_gen_prompt": image_gen_prompt,
                        "reward": float(new_reward),
                        "loss": float(loss.item()),
                        "image_path": f"optimized_image_{i}.png"
                    }, f, ensure_ascii=False)
                    f.write("\n")
            print(f"-- Text Branch New Reward: {new_reward}")
            reward_history.append(new_reward)
            current_reward = new_reward

        # 最终返回（你可以选择返回 new_img、tokens 等）
        return new_img, reward_history, total, len(generated_seq), update_length
    
    elif optimize_mode == "image":
        total = len(image_hidden_states_list)
        update_length = min(int(image_k * total), total)
        if update_length <= 0:
            print("Update Length Zero!!!")
            return None, reward_history, total, 0, update_length

        # === Step 1: 直接使用已处理好得image prompt embedding ===
        image_prompt_embed = image_prompt_embed.to(device)  # [1, T]

        # === Step 2: 优化图像 hidden_states ===
        optimized_states = torch.nn.Parameter(torch.stack([
            s.clone().detach().to(device).requires_grad_(True)
            for s in image_hidden_states_list[start_index:start_index + update_length]
        ]))  # [update_len, H]
        optimizer = torch.optim.Adam([optimized_states], lr=lr)

        new_img = None
        for i in range(max_image_steps):
            if current_reward > reward_threshold:
                break

            optimizer.zero_grad()

            # === Step 1: 计算 loss（基于当前 optimized_states）===
            logits = model.gen_head(optimized_states)  # [update_length, 2, vocab]
            logit_cond = logits[:, 0, :]  # [update_length, vocab]
            logit_uncond = logits[:, 1, :]  # [update_length, vocab]
            fused_logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)  # [update_length, vocab]

            probs = torch.softmax(fused_logits / temperature, dim=-1) + 1e-8
            token_ids = torch.argmax(probs, dim=-1)  # [update_length]
            log_pi = torch.log(probs[torch.arange(update_length), token_ids] + 1e-10)

            loss = - current_reward * log_pi.sum()
            print(f"-- Image Branch Loss: {loss.item()}")
            loss.backward()
            optimizer.step()

            # === Step 3: 拼接 prompt_embeds + optimized_states，生成图像 token ===
            with torch.no_grad():
                logits = model.gen_head(optimized_states)  # updated states
                logit_cond = logits[:, 0, :]
                logit_uncond = logits[:, 1, :]
                fused_logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
                probs = torch.softmax(fused_logits / temperature, dim=-1)

                sampled_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [update_length]
                # 每个 token_id → repeat 成 [2] → prepare embedding
                repeated_ids = sampled_token_ids.repeat_interleave(2).view(-1, 2)
                flat_ids = repeated_ids.reshape(-1)  # [2 * update_length]

                optimized_token_embeds = model.prepare_gen_img_embeds(flat_ids).reshape(update_length, 2, -1)
                optimized_token_embeds = optimized_token_embeds.permute(1, 0, 2)  # [2, update_len, H]

                # 拼接到 image_prompt_embed 后形成最终 inputs_embeds_img
                inputs_embeds_img = torch.cat([image_prompt_embed, optimized_token_embeds], dim=1)  # [2, T_total, H]

                current_embedding = inputs_embeds_img
                generated_tokens = torch.zeros((1, image_token_num), dtype=torch.int).to(device)
                outputs = None
                for j in range(update_length, image_token_num):
                    outputs = model.language_model.model(
                        inputs_embeds=current_embedding,
                        use_cache=True,
                        past_key_values=outputs.past_key_values if j > update_length else None
                    )
                    last_hidden = outputs.last_hidden_state[:, -1, :]  # [1, H]
                    logits = model.gen_head(last_hidden)
                    logit_cond = logits[0::2, :]
                    logit_uncond = logits[1::2, :]
                    logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
                    next_token = torch.multinomial(torch.softmax(logits / temperature, dim=-1), num_samples=1)  # [1]
                    generated_tokens[:, j] = next_token.squeeze(dim=-1)

                    next_token = next_token.repeat(1, 2).view(-1)
                    next_embed = model.prepare_gen_img_embeds(next_token).unsqueeze(1)
                    current_embedding = next_embed
                    
                # 替换前 update_length 的 token 为优化结果
                generated_tokens[:, :update_length] = sampled_token_ids

                # === decode image ===
                decoded = model.gen_vision_model.decode_code(
                    generated_tokens.to(dtype=torch.int),
                    shape=[1, 8, img_size // patch_size, img_size // patch_size]
                )
                decoded = decoded.detach().to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
                decoded = np.clip((decoded + 1) / 2 * 255, 0, 255).astype(np.uint8)
                new_img = PIL.Image.fromarray(decoded[0])

            new_reward = reward_model.get_reward(new_img, data)
            print(f"-- Image Branch New Reward: {new_reward}")
            if save_base_path is not None:
                os.makedirs(save_base_path, exist_ok=True)
                trace_file = os.path.join(save_base_path, "image_trace.jsonl")
                with open(trace_file, "a", encoding="utf-8") as f:
                    json.dump({
                        "step": i,
                        "image_gen_prompt": ori_image_prompt,
                        "reward": float(new_reward),
                        "loss": float(loss.item()),
                        "image_path": f"optimized_image_{i}.png"
                    }, f, ensure_ascii=False)
                    f.write("\n")
                save_path = os.path.join(save_base_path, f"optimized_image_{i}.png")
                new_img.save(save_path)

            reward_history.append(new_reward)
            current_reward = new_reward

        return new_img, reward_history, total, image_token_num, update_length
    # Branch: both
    else:  # "both"
        print("-- Both branch: starting text optimization first...")
        # ========== Step 1: 文本优化逻辑 ==========
        total = len(text_hidden_states_list)
        update_length = min(int(text_k * total), total)
        inputs = tokenizer([sft_prompt], return_tensors="pt").to(device)
        base_input_ids = inputs.input_ids.clone()

        if update_length <= 0:
            print("Update Length Zero!!!")
            return None, reward_history, total, 0, update_length

        original_seq = []
        original_seq.extend(
            text_final_input_ids[0][len(base_input_ids[-1]) : len(base_input_ids[-1]) + start_index]
        )
        optimized_states = torch.nn.Parameter(
            torch.stack(
                [
                    s.clone().detach().requires_grad_(True)
                    for s in text_hidden_states_list[
                        start_index : min(start_index + update_length, len(text_hidden_states_list))
                    ]
                ]
            )
        )
        optimizer_text = torch.optim.Adam([optimized_states], lr=lr)

        # ========== Step 2: 图像优化逻辑 ==========
        total_img = len(image_hidden_states_list)
        img_update_length = min(int(image_k * total_img), total_img)
        if img_update_length <= 0:
            print("Image Update Length Zero!!!")
            return None, reward_history, total_img, 0, img_update_length

        optimized_states_img = torch.nn.Parameter(
            torch.stack(
                [
                    s.clone().detach().to(device).requires_grad_(True)
                    for s in image_hidden_states_list[start_index:start_index + img_update_length]
                ]
            )
        )
        optimizer_img = torch.optim.Adam([optimized_states_img], lr=lr)

        new_img = None
        generated_seq = []
        for i in range(max_both_steps):
            if current_reward > reward_threshold:
                break
            optimizer_text.zero_grad()

            logits = model.language_model.lm_head(optimized_states)
            probs = torch.softmax(logits, dim=-1) + 1e-8
            next_token_ids = torch.argmax(probs, dim=-1).squeeze(-1)
            log_pi = torch.log(
                probs[torch.arange(update_length), 0, next_token_ids] + 1e-10
            )
            text_loss = -current_reward * log_pi.sum()
            print(f"-- Both Text Opt Loss: {text_loss.item()}")
            text_loss.backward(retain_graph=True)
            if grad_clip:
                torch.nn.utils.clip_grad_norm_([optimized_states], grad_clip)
            optimizer_text.step()

            # 用更新后的 states 重新生成文本
            with torch.no_grad():
                updated_probs = torch.softmax(model.language_model.lm_head(optimized_states), dim=-1)
                updated_token_ids = torch.argmax(updated_probs, dim=-1).squeeze(-1)
                updated_input_ids = torch.cat([base_input_ids, updated_token_ids.unsqueeze(0)], dim=-1)

                # 持续生成直到 eos 或长度限制
                generated_seq = []
                generated_seq.extend(original_seq)
                generated_seq.extend(updated_token_ids.tolist())

                cnt = 0
                while True:
                    outputs = model.language_model.model(
                        updated_input_ids, output_hidden_states=True
                    )
                    last_hidden = outputs[0][:, -1, :]
                    next_token_id = torch.argmax(model.language_model.lm_head(last_hidden), dim=-1)
                    token_str = tokenizer.decode(next_token_id.item())
                    generated_seq.append(next_token_id.item())
                    updated_input_ids = torch.cat([updated_input_ids, next_token_id.unsqueeze(0)], dim=-1)
                    cnt += 1
                    if token_str in stop_words or cnt > max_text_tokens:
                        break

            new_generated_text = tokenizer.decode(generated_seq, skip_special_tokens=True)
            print(f"New Generated Text: {new_generated_text}")

            image_gen_prompt = f"{data['prompt']}. {new_generated_text}"

            # 用优化后的文本更新 image_prompt_embed
            # 重新 encode 优化后的文本prompt作为image分支输入
            img_gen_conversation = [{"role": "User", "content": image_gen_prompt}, {"role": "Assistant", "content": ""}]
            sft_image_prompt = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=img_gen_conversation, sft_format=vl_chat_processor.sft_format, system_prompt=""
            )
            print(f"Prompt: {sft_image_prompt}")
            prompt_inputs = tokenizer(
                text=[sft_image_prompt], return_tensors="pt", padding=True, padding_side="right", add_special_tokens=True
            )
            image_prompt_ids = prompt_inputs["input_ids"].to(device)
            image_start_token_id = tokenizer.encode(vl_chat_processor.image_start_tag)[1]
            image_prompt_ids = torch.cat([image_prompt_ids, image_prompt_ids.new_full((image_prompt_ids.size(0), 1), image_start_token_id)], dim=1).repeat(1, 1)
            cond_inputs_embeds = model.language_model.get_input_embeddings()(image_prompt_ids)
            pad_input_embeds = model.language_model.get_input_embeddings()(image_prompt_ids.new_full((1, 1), vl_chat_processor.pad_id))
            uncond_inputs_embeds = cond_inputs_embeds.clone()
            uncond_inputs_embeds[:,1:-1] = pad_input_embeds
            image_prompt_embed = torch.repeat_interleave(cond_inputs_embeds, 2, dim=0)
            image_prompt_embed[1::2] = uncond_inputs_embeds

            optimizer_img.zero_grad()

            # 跟image branch逻辑一样
            logits = model.gen_head(optimized_states_img)
            logit_cond = logits[:, 0, :]
            logit_uncond = logits[:, 1, :]
            fused_logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(fused_logits / temperature, dim=-1)
            token_ids = torch.argmax(probs, dim=-1)
            log_pi = torch.log(probs[torch.arange(img_update_length), token_ids] + 1e-10)

            image_loss = -current_reward * log_pi.sum()
            print(f"-- Both Image Opt Loss: {image_loss.item()}")
            image_loss.backward()
            optimizer_img.step()

            # 生成完整图像
            with torch.no_grad():
                logits = model.gen_head(optimized_states_img)  # updated states
                logit_cond = logits[:, 0, :]
                logit_uncond = logits[:, 1, :]
                fused_logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
                probs = torch.softmax(fused_logits / temperature, dim=-1)

                sampled_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [update_length]
                # 拼接prompt embedding与优化后的image token再解码
                optimized_token_embeds = model.prepare_gen_img_embeds(
                    sampled_token_ids.repeat_interleave(2).view(-1, 2).reshape(-1)
                ).reshape(img_update_length, 2, -1).permute(1, 0, 2)
                inputs_embeds_img = torch.cat([image_prompt_embed, optimized_token_embeds], dim=1)

                generated_tokens = torch.zeros((1, image_token_num), dtype=torch.int).to(device)
                generated_tokens[:, :img_update_length] = sampled_token_ids
                current_embeds = inputs_embeds_img

                for j in range(img_update_length, image_token_num):
                    img_outputs = model.language_model.model(
                        inputs_embeds=current_embeds,
                        use_cache=True,
                        past_key_values=img_outputs.past_key_values if j > img_update_length else None
                    )
                    last_hidden = img_outputs.last_hidden_state[:, -1, :]
                    logits = model.gen_head(last_hidden)
                    logit_cond = logits[0::2, :]
                    logit_uncond = logits[1::2, :]
                    fused_logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
                    next_token = torch.multinomial(torch.softmax(fused_logits / temperature, dim=-1), num_samples=1)
                    generated_tokens[:, j] = next_token.squeeze(dim=-1)
                    next_token = next_token.repeat(1, 2).view(-1)
                    next_embeds = model.prepare_gen_img_embeds(next_token).unsqueeze(1)
                    current_embeds = next_embeds

                decoded = model.gen_vision_model.decode_code(
                    generated_tokens.to(dtype=torch.int),
                    shape=[1, 8, img_size // patch_size, img_size // patch_size],
                )
                decoded = decoded.detach().to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
                decoded = np.clip((decoded + 1) / 2 * 255, 0, 255).astype(np.uint8)
                new_img = PIL.Image.fromarray(decoded[0])

            new_reward = reward_model.get_reward(new_img, data)
            print(f"-- Both branch new image reward: {new_reward}")
            reward_history.append(new_reward)
            current_reward = new_reward
            # ==================== Save trace ====================
            if save_base_path is not None:
                os.makedirs(save_base_path, exist_ok=True)
                trace_file = os.path.join(save_base_path, "both_trace.jsonl")
                # 当前生成的 image 文件路径
                img_save_path = os.path.join(save_base_path, f"optimized_image_{i}.png")
                #os.makedirs(save_dir, exist_ok=True)
                new_img.save(img_save_path)
                log_data = {
                    "step": i,
                    "reward": float(new_reward),
                    "text_loss": float(text_loss.item()),     # 前面 text branch 中损失
                    "image_loss": float(image_loss.item()),      # 当前 image branch 中损失
                    "optimized_text": new_generated_text,
                    "image_gen_prompt": image_gen_prompt,  # 记录完整 image prompt
                    "image_path": img_save_path,
                    "update_length": int(update_length),
                    "img_update_length": int(img_update_length),
                }
                with open(trace_file, "a", encoding="utf-8") as f:
                    json.dump(log_data, f, ensure_ascii=False)
                    f.write("\n")
                print(f"Trace saved: {trace_file}, image at: {img_save_path}")

        return new_img, reward_history, total_img+total, image_token_num+len(generated_seq), img_update_length+update_length

# model_path = "deepseek-ai/Janus-Pro-7B"
# vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)

# vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
#     model_path, trust_remote_code=True
# )
# vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
# input_text = "a photo of a purple suitcase and an orange pizza"
# data = {"tag": "color_attr", "include": [{"class": "suitcase", "count": 1, "color": "purple"}, {"class": "pizza", "count": 1, "color": "orange"}], "prompt": "a photo of a purple suitcase and an orange pizza"}

# reward_model = RewardModel(
#     model_path="/media/raid/workspace/miyapeng/Multimodal-LatentSeek/src/rewards/<OBJECT_DETECTOR_FOLDER>",
#     object_names_path="/media/raid/workspace/miyapeng/Multimodal-LatentSeek/src/rewards/object_names.txt",
#     options={"clip_model": "ViT-L-14"}
# )

# answer, text_hidden_states_list, text_final_input_ids, image_hidden_states_list, image_prompt_embed, generated_image_tokens = original_generation(input_text, vl_gpt, vl_chat_processor, torch.device("cuda"))
# new_img, reward_history, ori_length, generated_seq, update_length = optimized_generation(reward_model, answer, data, vl_gpt, vl_chat_processor, torch.device("cuda"), text_hidden_states_list, text_final_input_ids, image_hidden_states_list, image_prompt_embed, generated_image_tokens, optimize_mode="both", save_base_path="/media/raid/workspace/miyapeng/Multimodal-LatentSeek/src/geneval_results/test")