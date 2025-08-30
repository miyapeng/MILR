import os

import torch
import numpy as np
import PIL.Image
import json

from process import save_image_and_metadata
from ori_generation_janus import original_generation
from janus.models import MultiModalityCausalLM, VLChatProcessor
from transformers import AutoModelForCausalLM

from rewards.reward import RewardModel

@torch.no_grad()
def generate_image_from_prompt_and_states(
    model,
    vl_chat_processor,
    tokenizer,
    image_gen_prompt: str,                   # 原始文本 prompt
    optimized_states_img: torch.Tensor, # [update_len, 2, hidden]：你上游优化得到的前缀隐藏状态
    update_length_img: int,             # 前缀长度（与 optimized_states_img 对齐）
    image_token_num: int = 576,               # 整张图的 image token 总长度
    cfg_weight: float = 5.0,
    temperature: float = 1.0,
    img_size: int = 384,
    patch_size: int = 16,
    device: str | torch.device = "cuda",
    save_path: str | None = None,       # 可选：保存 PNG 的路径；None 则不落盘
):
    """
    一体化:Prompt -> 构建 cond/uncond image prompt embed -> 前缀段采样 -> 自回归补全 -> 解码图片
    返回: (final_img: PIL.Image.Image, generated_tokens: LongTensor[1, image_token_num])
    形状约定：
      - optimized_states_img: [update_len, 2, hidden] （第1维是步长，第2维是 cond/uncond 对偶）
      - 内部构建的 image_prompt_embed: [2, prompt_len, hidden]
    """

    device = torch.device(device)

    # -------------------------
    # A) 从 prompt 构建 image_prompt_embed（含 cond/uncond）
    # -------------------------
    img_conv = [{"role": "User", "content": image_gen_prompt}, {"role": "Assistant", "content": ""}]
    sft_image_prompt = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=img_conv, sft_format=vl_chat_processor.sft_format, system_prompt=""
    )

    prompt_inputs = tokenizer(
        text=[sft_image_prompt], return_tensors="pt", padding=True,
        padding_side="right", add_special_tokens=True
    )
    image_prompt_ids = prompt_inputs["input_ids"].to(device)

    # image_start_token 追加到末尾
    img_start_ids = tokenizer.encode(vl_chat_processor.image_start_tag)
    assert len(img_start_ids) >= 2, "image_start_tag 编码长度异常，检查 tokenizer/模板设置。"
    image_start_token_id = img_start_ids[1]
    image_prompt_ids = torch.cat(
        [image_prompt_ids, image_prompt_ids.new_full((image_prompt_ids.size(0), 1), image_start_token_id)], dim=1
    )

    # cond 路嵌入
    cond_inputs_embeds = model.language_model.get_input_embeddings()(image_prompt_ids)
    # uncond 路：仅保留首/末，中间替换为 pad
    pad_input_embeds = model.language_model.get_input_embeddings()(
        image_prompt_ids.new_full((1, 1), vl_chat_processor.pad_id)
    )
    uncond_inputs_embeds = cond_inputs_embeds.clone()
    uncond_inputs_embeds[:, 1:-1] = pad_input_embeds

    # 交错组成 batch=2（cond/uncond）
    image_prompt_embed = torch.repeat_interleave(cond_inputs_embeds, 2, dim=0)
    image_prompt_embed[1::2] = uncond_inputs_embeds
    # [2, prompt_len, hidden]

    # -------------------------
    # B) 前缀段：用 optimized_states_img 直接采样 update_length_img 个 token
    # -------------------------
    logits = model.gen_head(optimized_states_img.to(device))          # [update_len, 2, vocab]
    logit_cond = logits[:, 0, :]                                      # [update_len, vocab]
    logit_uncond = logits[:, 1, :]
    fused_logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
    probs = torch.softmax(fused_logits / temperature, dim=-1) + 1e-8

    sampled_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [update_len] long/int
    # sampled_token_ids = sampled_token_ids.to(torch.long)

    # 把前缀 token id 转回图像 token 的输入嵌入（cond/uncond 对偶）
    token_ids_for_embed = sampled_token_ids.repeat_interleave(2).view(-1, 2).reshape(-1)
    optimized_token_embeds = model.prepare_gen_img_embeds(token_ids_for_embed)          # [(update_len*2), hidden]
    optimized_token_embeds = optimized_token_embeds.reshape(update_length_img, 2, -1).permute(1, 0, 2)  # [2, update_len, hidden]

    # 初始输入 = prompt_embed + 前缀 token 的 embed
    inputs_embeds_img = torch.cat([image_prompt_embed.to(device), optimized_token_embeds], dim=1)  # [2, prompt_len+update_len, hidden]

    # 生成序列容器（long）
    generated_tokens = torch.zeros((1, image_token_num), dtype=torch.long, device=device)
    generated_tokens[:, :update_length_img] = sampled_token_ids

    # -------------------------
    # C) 自回归补全
    # -------------------------
    current_embeds = inputs_embeds_img   # 先喂完整上下文，后续只喂最后一步（依赖 past_key_values）
    img_outputs = None

    for j in range(update_length_img, image_token_num):
        img_outputs = model.language_model.model(
            inputs_embeds=current_embeds,
            use_cache=True,
            past_key_values=img_outputs.past_key_values if img_outputs is not None else None,
        )
        last_hidden = img_outputs.last_hidden_state[:, -1, :]  # [2, hidden]
        lg = model.gen_head(last_hidden)                       # [2, vocab]
        lc = lg[0::2, :]                                       # cond
        lu = lg[1::2, :]                                       # uncond
        lg = lu + cfg_weight * (lc - lu)

        probs_next = torch.softmax(lg / temperature, dim=-1)
        nxt_tok = torch.multinomial(probs_next, num_samples=1)         # [1, 1] long
        generated_tokens[:, j] = nxt_tok.squeeze(dim=-1)

        # 下一步仅喂“最后一个 token 的嵌入”（cond/uncond 对偶），依赖 KV cache 持续累积上下文
        nxt_ids = nxt_tok.repeat(1, 2).view(-1)                         # [2]
        nxt_embeds = model.prepare_gen_img_embeds(nxt_ids).unsqueeze(1) # [2, 1, hidden]
        current_embeds = nxt_embeds

    # -------------------------
    # D) 解码为图片
    # -------------------------
    decoded = model.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),  # 若 decode_code 需要 int，可保持与模型一致
        shape=[1, 8, img_size // patch_size, img_size // patch_size],
    )
    decoded = decoded.detach().to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    decoded = np.clip((decoded + 1) / 2 * 255, 0, 255).astype(np.uint8)
    final_img = PIL.Image.fromarray(decoded[0])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        final_img.save(save_path)

    return final_img

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
    max_text_steps=20,
    max_image_steps=20,
    max_both_steps=20,
    lr=0.01,
    grad_clip=None,
    text_k=0.2,
    image_k=0.02,
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
    # assert optimize_mode in ("text", "image", "both"), "optimize_mode must be 'text', 'image', or 'both'"

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

    if optimize_mode == "interleaved_text_image":
        print("-- Both (sequential): interleaved text and image...")

        # ====== 文本阶段（仅优化文本 hidden states，最多 max_text_steps 步）======
        total_text = len(text_hidden_states_list)
        update_length_text = min(int(text_k * total_text), total_text)
        if update_length_text <= 0:
            print("Text Update Length Zero!!!")
            return None, reward_history, total_text, 0, 0

        total_img = len(image_hidden_states_list)
        update_length_img = min(int(image_k * total_img), total_img)
        if update_length_img <= 0:
            print("Image Update Length Zero!!!")
            return last_img, reward_history, total_text, len(generated_seq), update_length_text

        # 准备文本优化所需前缀
        inputs = tokenizer([sft_prompt], return_tensors="pt").to(device)
        base_input_ids = inputs.input_ids.clone()

        # 原始序列前缀（prompt 后到 start_index 的 token）
        original_seq = []
        original_seq.extend(
            text_final_input_ids[0][len(base_input_ids[-1]) : len(base_input_ids[-1]) + start_index]
        )

        optimized_states_text = torch.nn.Parameter(
            torch.stack(
                [
                    s.clone().detach().requires_grad_(True)
                    for s in text_hidden_states_list[
                        start_index : min(start_index + update_length_text, len(text_hidden_states_list))
                    ]
                ]
            )
        )
        optimizer_text = torch.optim.Adam([optimized_states_text], lr=lr)
        
        optimized_states_img = torch.nn.Parameter(
            torch.stack(
                [
                    s.clone().detach().to(device).requires_grad_(True)
                    for s in image_hidden_states_list[start_index : start_index + update_length_img]
                ]
            )
        )
        optimizer_img = torch.optim.Adam([optimized_states_img], lr=lr)

        input_ids = text_final_input_ids[:, : len(base_input_ids[-1]) + start_index]
        base_input_ids = input_ids.clone().to(device)

        last_img = image  # 文本阶段每步会生成对应图片并评估 reward
        generated_seq = []

        final_img = last_img
        img_outputs = None

        # 若初始就已超过阈值，直接退出
        if current_reward > reward_threshold:
            print(f"Early stop before optimization: reward {current_reward:.6f} > {reward_threshold}")
            return last_img, reward_history, total_text, 0, 0

        for i in range(max_both_steps):
            optimizer_text.zero_grad()

            # 计算文本阶段 loss（基于当前 optimized_states_text）
            logits = model.language_model.lm_head(optimized_states_text)             # [update_len, vocab]
            probs = torch.softmax(logits, dim=-1) + 1e-8
            next_token_ids = torch.argmax(probs, dim=-1).squeeze(-1)                # [update_len]
            log_pi = torch.log(probs[torch.arange(update_length_text), 0, next_token_ids] + 1e-10)

            text_loss = - current_reward * log_pi.sum()
            print(f"-- Text Phase Loss[{i}]: {text_loss.item():.6f}")
            text_loss.backward(retain_graph=True)
            if grad_clip:
                torch.nn.utils.clip_grad_norm_([optimized_states_text], grad_clip)
            optimizer_text.step()

            # 用更新后的 states 展开文本，直到 stop word 或长度上限
            generated_seq = []
            generated_seq.extend(original_seq)
            with torch.no_grad():
                updated_probs = torch.softmax(model.language_model.lm_head(optimized_states_text), dim=-1)
                updated_token_ids = torch.argmax(updated_probs, dim=-1).squeeze(-1)
                generated_seq.extend(updated_token_ids.tolist())
                updated_input_ids = torch.cat([base_input_ids, updated_token_ids.unsqueeze(0)], dim=-1)

                steps = 0
                while True:
                    outputs = model.language_model.model(updated_input_ids, output_hidden_states=True)
                    last_hidden = outputs[0][:, -1]
                    nxt = torch.argmax(model.language_model.lm_head(last_hidden), dim=-1)
                    tok = tokenizer.decode(nxt.item())
                    generated_seq.append(nxt.item())
                    updated_input_ids = torch.cat([updated_input_ids, nxt.unsqueeze(0)], dim=-1)
                    steps += 1
                    if tok in stop_words or steps > max_text_tokens:
                        break

            new_generated_text = tokenizer.decode(generated_seq, skip_special_tokens=True)
            print(f"New Optimized Text[{i}]: {new_generated_text}")

            # 用最新文本生成图像并评估 reward（image 不做优化，只重生成）
            image_gen_prompt = f"{data['prompt']}. {new_generated_text}"
            save_path = None
            if save_base_path is not None:
                os.makedirs(save_base_path, exist_ok=True)
                save_path = os.path.join(save_base_path, f"optimized_image_text_{i}.png")

            new_img = generate_image_from_prompt_and_states(model, vl_chat_processor, tokenizer, image_gen_prompt, optimized_states_img, update_length_img,save_path=save_path)  ##optimize the origin image rather than the optimized image
            
            last_img = new_img

            new_reward = reward_model.get_reward(new_img, data)
            print(f"-- Text Phase New Reward[{i}]: {new_reward:.6f}")
            reward_history.append(new_reward)
            current_reward = new_reward

            # 记录 trace
            if save_base_path is not None:
                with open(os.path.join(save_base_path, "both_text_trace.jsonl"), "a", encoding="utf-8") as f:
                    json.dump({
                        "phase": "text",
                        "step": i,
                        "generated_text": new_generated_text,
                        "image_gen_prompt": image_gen_prompt,
                        "reward": float(new_reward),
                        "loss": float(text_loss.item()),
                        "image_path": f"optimized_image_text_{i}.png" if save_path else None
                    }, f, ensure_ascii=False)
                    f.write("\n")

            # 早停：文本阶段达阈值直接退出
            if current_reward > reward_threshold:
                print(f"Early Stop after Text Phase: reward {current_reward:.6f} > {reward_threshold}")
                return new_img, reward_history, total_text, len(generated_seq), update_length_text

            # ====== 图像阶段（仅优化图像 hidden states，最多 max_image_steps 步）======
            print("-- Switching to Image Phase with the last optimized text...")

            # 根据最新文本重建 image_prompt_embed（与原 both 分支一致）
            img_conv = [{"role": "User", "content": image_gen_prompt}, {"role": "Assistant", "content": ""}]
            sft_image_prompt = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=img_conv, sft_format=vl_chat_processor.sft_format, system_prompt=""
            )
            prompt_inputs = tokenizer(
                text=[sft_image_prompt], return_tensors="pt", padding=True, padding_side="right", add_special_tokens=True
            )
            image_prompt_ids = prompt_inputs["input_ids"].to(device)
            image_start_token_id = tokenizer.encode(vl_chat_processor.image_start_tag)[1]
            image_prompt_ids = torch.cat(
                [image_prompt_ids, image_prompt_ids.new_full((image_prompt_ids.size(0), 1), image_start_token_id)],
                dim=1
            ).repeat(1, 1)
            cond_inputs_embeds = model.language_model.get_input_embeddings()(image_prompt_ids)
            pad_input_embeds = model.language_model.get_input_embeddings()(
                image_prompt_ids.new_full((1, 1), vl_chat_processor.pad_id)
            )
            uncond_inputs_embeds = cond_inputs_embeds.clone()
            uncond_inputs_embeds[:, 1:-1] = pad_input_embeds
            image_prompt_embed = torch.repeat_interleave(cond_inputs_embeds, 2, dim=0)
            image_prompt_embed[1::2] = uncond_inputs_embeds

            optimizer_img.zero_grad()

            logits = model.gen_head(optimized_states_img)  # [update_len, 2, vocab]（按你原本的假设）
            logit_cond = logits[:, 0, :]
            logit_uncond = logits[:, 1, :]
            fused_logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(fused_logits / temperature, dim=-1) + 1e-8
            token_ids = torch.argmax(probs, dim=-1)
            log_pi = torch.log(probs[torch.arange(update_length_img), token_ids] + 1e-10)

            image_loss = - current_reward * log_pi.sum()
            print(f"-- Image Phase Loss[{i}]: {image_loss.item():.6f}")
            image_loss.backward()
            optimizer_img.step()

            # 用更新后的 optimized_states_img 采样若干 token，并与 image_prompt_embed 拼接生成完整图
            with torch.no_grad():
                logits = model.gen_head(optimized_states_img)
                logit_cond = logits[:, 0, :]
                logit_uncond = logits[:, 1, :]
                fused_logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
                probs = torch.softmax(fused_logits / temperature, dim=-1)

                sampled_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [update_len]
                optimized_token_embeds = model.prepare_gen_img_embeds(
                    sampled_token_ids.repeat_interleave(2).view(-1, 2).reshape(-1)
                ).reshape(update_length_img, 2, -1).permute(1, 0, 2)
                inputs_embeds_img = torch.cat([image_prompt_embed, optimized_token_embeds], dim=1)

                # 继续自回归生成剩余 token
                generated_tokens = torch.zeros((1, image_token_num), dtype=torch.int).to(device)
                generated_tokens[:, :update_length_img] = sampled_token_ids
                current_embeds = inputs_embeds_img
                img_outputs = None
                for j in range(update_length_img, image_token_num):
                    img_outputs = model.language_model.model(
                        inputs_embeds=current_embeds,
                        use_cache=True,
                        past_key_values=img_outputs.past_key_values if img_outputs is not None else None
                    )
                    last_hidden = img_outputs.last_hidden_state[:, -1, :]
                    lg = model.gen_head(last_hidden)
                    lc = lg[0::2, :]
                    lu = lg[1::2, :]
                    lg = lu + cfg_weight * (lc - lu)
                    nxt_tok = torch.multinomial(torch.softmax(lg / temperature, dim=-1), num_samples=1)
                    generated_tokens[:, j] = nxt_tok.squeeze(dim=-1)
                    nxt_ids = nxt_tok.repeat(1, 2).view(-1)
                    nxt_embeds = model.prepare_gen_img_embeds(nxt_ids).unsqueeze(1)
                    current_embeds = nxt_embeds

                decoded = model.gen_vision_model.decode_code(
                    generated_tokens.to(dtype=torch.int),
                    shape=[1, 8, img_size // patch_size, img_size // patch_size],
                )
                decoded = decoded.detach().to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
                decoded = np.clip((decoded + 1) / 2 * 255, 0, 255).astype(np.uint8)
                final_img = PIL.Image.fromarray(decoded[0])

            new_reward = reward_model.get_reward(final_img, data)
            print(f"-- Image Phase New Reward[{i}]: {new_reward:.6f}")
            reward_history.append(new_reward)
            current_reward = new_reward

            if save_base_path is not None:
                os.makedirs(save_base_path, exist_ok=True)
                img_path = os.path.join(save_base_path, f"optimized_image_img_{i}.png")
                final_img.save(img_path)
                with open(os.path.join(save_base_path, "both_image_trace.jsonl"), "a", encoding="utf-8") as f:
                    json.dump({
                        "phase": "image",
                        "step": i,
                        "image_gen_prompt": image_gen_prompt,
                        "reward": float(new_reward),
                        "loss": float(image_loss.item()),
                        "image_path": img_path
                    }, f, ensure_ascii=False)
                    f.write("\n")

            if current_reward > reward_threshold:
                print(f"Early Stop after Image Phase: reward {current_reward:.6f} > {reward_threshold}")
                return final_img, reward_history, (total_text + total_img), (image_token_num + len(generated_seq)), (update_length_text + update_length_img)

        # 两阶段都未达到阈值，返回最终结果
        return final_img, reward_history, (total_text + total_img), (image_token_num + len(generated_seq)), (update_length_text + update_length_img)
    
    elif optimize_mode == "image_text":
        print("-- Image→Text (sequential): image phase first, then text phase with fixed prefix from image...")

        # === Step 1: 直接使用已处理好得image prompt embedding ===
        image_prompt_embed = image_prompt_embed.to(device)  # [1, T]

        total_img = len(image_hidden_states_list)
        update_length_img = min(int(image_k * total_img), total_img)
        if update_length_img <= 0:
            print("Image Update Length Zero!!!")
            return image, reward_history, total_img, 0, 0

        optimized_states_img = torch.nn.Parameter(
            torch.stack([
                s.clone().detach().to(device).requires_grad_(True)
                for s in image_hidden_states_list[start_index : start_index + update_length_img]
            ])
        )
        optimizer_img = torch.optim.Adam([optimized_states_img], lr=lr)

        if current_reward > reward_threshold:
            print(f"Early stop before Image phase: reward {current_reward:.6f} > {reward_threshold}")
            return image, reward_history, total_img, 0, update_length_img

        final_img = image
        final_generated_tokens = None  # [1, image_token_num]
        img_outputs = None

        for i in range(max_image_steps):
            optimizer_img.zero_grad()

            logits = model.gen_head(optimized_states_img)          # [update_len, 2, vocab]
            lc = logits[:, 0, :]; lu = logits[:, 1, :]
            fused = lu + cfg_weight * (lc - lu)
            probs = torch.softmax(fused / temperature, dim=-1) + 1e-8
            token_ids = torch.argmax(probs, dim=-1)               # [update_len]
            log_pi = torch.log(probs[torch.arange(update_length_img), token_ids] + 1e-10)

            image_loss = - current_reward * log_pi.sum()
            print(f"-- Image Phase Loss[{i}]: {image_loss.item():.6f}")
            image_loss.backward()
            optimizer_img.step()

            with torch.no_grad():
                # 采样前缀 token
                logits = model.gen_head(optimized_states_img)
                lc = logits[:, 0, :]; lu = logits[:, 1, :]
                fused = lu + cfg_weight * (lc - lu)
                probs = torch.softmax(fused / temperature, dim=-1)
                sampled_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [update_len]

                # 拼入 prompt 后做自回归补全
                prefix_embeds = model.prepare_gen_img_embeds(
                    sampled_token_ids.repeat_interleave(2).view(-1, 2).reshape(-1)
                ).reshape(update_length_img, 2, -1).permute(1, 0, 2)                      # [2, update_len, H]
                current_embeds = torch.cat([image_prompt_embed, prefix_embeds], dim=1)     # [2, T0+update_len, H]

                generated_tokens = torch.zeros((1, image_token_num), dtype=torch.int).to(device)
                generated_tokens[:, :update_length_img] = sampled_token_ids
                img_outputs = None
                for j in range(update_length_img, image_token_num):
                    img_outputs = model.language_model.model(
                        inputs_embeds=current_embeds,
                        use_cache=True,
                        past_key_values=img_outputs.past_key_values if img_outputs is not None else None
                    )
                    last_hidden = img_outputs.last_hidden_state[:, -1, :]
                    lg = model.gen_head(last_hidden)
                    lc_ = lg[0::2, :]; lu_ = lg[1::2, :]
                    lg = lu_ + cfg_weight * (lc_ - lu_)
                    nxt_tok = torch.multinomial(torch.softmax(lg / temperature, dim=-1), num_samples=1)
                    generated_tokens[:, j] = nxt_tok.squeeze(dim=-1)
                    nxt_ids = nxt_tok.repeat(1, 2).view(-1)
                    nxt_embeds = model.prepare_gen_img_embeds(nxt_ids).unsqueeze(1)
                    current_embeds = nxt_embeds

                decoded = model.gen_vision_model.decode_code(
                    generated_tokens.to(dtype=torch.int),
                    shape=[1, 8, img_size // patch_size, img_size // patch_size]
                )
                decoded = decoded.detach().to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
                decoded = np.clip((decoded + 1) / 2 * 255, 0, 255).astype(np.uint8)
                final_img = PIL.Image.fromarray(decoded[0])
                final_generated_tokens = generated_tokens.clone()

            new_reward = reward_model.get_reward(final_img, data)
            print(f"-- Image Phase New Reward[{i}]: {new_reward:.6f}")
            reward_history.append(new_reward)
            current_reward = new_reward

            if save_base_path is not None:
                os.makedirs(save_base_path, exist_ok=True)
                img_path = os.path.join(save_base_path, f"optimized_image_img_{i}.png")
                final_img.save(img_path)
                with open(os.path.join(save_base_path, "image_text_image_trace.jsonl"), "a", encoding="utf-8") as f:
                    json.dump({
                        "phase": "image", "step": i, "image_gen_prompt": ori_image_prompt,
                        "reward": float(new_reward), "loss": float(image_loss.item()), "image_path": img_path
                    }, f, ensure_ascii=False); f.write("\n")

            if current_reward > reward_threshold:
                print(f"Early Stop after Image Phase: reward {current_reward:.6f} > {reward_threshold}")
                return final_img, reward_history, total_img, image_token_num, update_length_img

        image_hidden_states = optimized_states_img

        print("-- Switching to Text Phase with fixed image prefix...")

        total_text = len(text_hidden_states_list)
        update_length_text = min(int(text_k * total_text), total_text)
        if update_length_text <= 0:
            print("Text Update Length Zero!!!")
            return final_img, reward_history, (total_img + total_text), image_token_num, update_length_img

        inputs_txt = tokenizer([sft_prompt], return_tensors="pt").to(device)
        base_input_ids = inputs_txt.input_ids.clone()

        original_seq = []
        original_seq.extend(
            text_final_input_ids[0][len(base_input_ids[-1]) : len(base_input_ids[-1]) + start_index]
        )

        optimized_states_text = torch.nn.Parameter(
            torch.stack([
                s.clone().detach().requires_grad_(True)
                for s in text_hidden_states_list[start_index : min(start_index + update_length_text, len(text_hidden_states_list))]
            ])
        )
        optimizer_text = torch.optim.Adam([optimized_states_text], lr=lr)

        if current_reward > reward_threshold:
            print(f"Early stop before Text phase: reward {current_reward:.6f} > {reward_threshold}")
            return final_img, reward_history, (total_img + total_text), image_token_num, (update_length_img + update_length_text)

        input_ids = text_final_input_ids[:, : len(base_input_ids[-1]) + start_index]
        base_input_ids = input_ids.clone().to(device)

        last_img = final_img
        generated_seq = []

        for i in range(max_text_steps):
            optimizer_text.zero_grad()

            logits = model.language_model.lm_head(optimized_states_text)        # [update_len_text, vocab]
            probs = torch.softmax(logits, dim=-1) + 1e-8
            next_token_ids = torch.argmax(probs, dim=-1).squeeze(-1)
            log_pi = torch.log(probs[torch.arange(update_length_text), 0, next_token_ids] + 1e-10)

            text_loss = - current_reward * log_pi.sum()
            print(f"-- Text Phase Loss[{i}]: {text_loss.item():.6f}")
            text_loss.backward(retain_graph=True)
            if grad_clip:
                torch.nn.utils.clip_grad_norm_([optimized_states_text], grad_clip)
            optimizer_text.step()

            # 展开文本 → 组装新的 image prompt
            generated_seq = []
            generated_seq.extend(original_seq)
            with torch.no_grad():
                up = torch.softmax(model.language_model.lm_head(optimized_states_text), dim=-1) + 1e-8
                upd_ids = torch.argmax(up, dim=-1).squeeze(-1)
                generated_seq.extend(upd_ids.tolist())
                upd_input_ids = torch.cat([base_input_ids, upd_ids.unsqueeze(0)], dim=-1)

                steps = 0
                while True:
                    out = model.language_model.model(upd_input_ids, output_hidden_states=True)
                    last_h = out[0][:, -1]
                    nxt = torch.argmax(model.language_model.lm_head(last_h), dim=-1)
                    tok = tokenizer.decode(nxt.item())
                    generated_seq.append(nxt.item())
                    upd_input_ids = torch.cat([upd_input_ids, nxt.unsqueeze(0)], dim=-1)
                    steps += 1
                    if tok in stop_words or steps > max_text_tokens:
                        break

            new_generated_text = tokenizer.decode(generated_seq, skip_special_tokens=True)
            img_conv2 = [{"role": "User", "content": f"{data['prompt']}. {new_generated_text}"},
                        {"role": "Assistant", "content": ""}]
            sft_image_prompt2 = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=img_conv2, sft_format=vl_chat_processor.sft_format, system_prompt=""
            )
            prompt_inputs2 = tokenizer(
                text=[sft_image_prompt2], return_tensors="pt", padding=True, padding_side="right", add_special_tokens=True
            )
            image_prompt_ids2 = prompt_inputs2["input_ids"].to(device)
            image_start_token_id2 = tokenizer.encode(vl_chat_processor.image_start_tag)[1]
            image_prompt_ids2 = torch.cat(
                [image_prompt_ids2, image_prompt_ids2.new_full((image_prompt_ids2.size(0), 1), image_start_token_id2)],
                dim=1
            )
            cond2 = model.language_model.get_input_embeddings()(image_prompt_ids2)
            pad2 = model.language_model.get_input_embeddings()(image_prompt_ids2.new_full((1, 1), vl_chat_processor.pad_id))
            uncond2 = cond2.clone(); uncond2[:, 1:-1] = pad2
            image_prompt_embed2 = torch.repeat_interleave(cond2, 2, dim=0); image_prompt_embed2[1::2] = uncond2

            with torch.no_grad():
                # 采样前缀 token
                logits = model.gen_head(image_hidden_states)
                lc = logits[:, 0, :]; lu = logits[:, 1, :]
                fused = lu + cfg_weight * (lc - lu)
                probs = torch.softmax(fused / temperature, dim=-1)
                sampled_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [update_len]

                # 拼入 prompt 后做自回归补全
                prefix_embeds = model.prepare_gen_img_embeds(
                    sampled_token_ids.repeat_interleave(2).view(-1, 2).reshape(-1)
                ).reshape(update_length_img, 2, -1).permute(1, 0, 2)                      # [2, update_len, H]
                current_embeds = torch.cat([image_prompt_embed2, prefix_embeds], dim=1)     # [2, T0+update_len, H]

                generated_tokens = torch.zeros((1, image_token_num), dtype=torch.int).to(device)
                generated_tokens[:, :update_length_img] = sampled_token_ids
                img_outputs = None
                for j in range(update_length_img, image_token_num):
                    img_outputs = model.language_model.model(
                        inputs_embeds=current_embeds,
                        use_cache=True,
                        past_key_values=img_outputs.past_key_values if img_outputs is not None else None
                    )
                    last_hidden = img_outputs.last_hidden_state[:, -1, :]
                    lg = model.gen_head(last_hidden)
                    lc_ = lg[0::2, :]; lu_ = lg[1::2, :]
                    lg = lu_ + cfg_weight * (lc_ - lu_)
                    nxt_tok = torch.multinomial(torch.softmax(lg / temperature, dim=-1), num_samples=1)
                    generated_tokens[:, j] = nxt_tok.squeeze(-1)
                    nxt_ids = nxt_tok.repeat(1, 2).view(-1)
                    nxt_embeds = model.prepare_gen_img_embeds(nxt_ids).unsqueeze(1)
                    current_embeds = nxt_embeds

                decoded = model.gen_vision_model.decode_code(
                    generated_tokens.to(dtype=torch.int),
                    shape=[1, 8, img_size // patch_size, img_size // patch_size]
                )
                decoded = decoded.detach().to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
                decoded = np.clip((decoded + 1) / 2 * 255, 0, 255).astype(np.uint8)
                new_img2 = PIL.Image.fromarray(decoded[0])

            new_reward = reward_model.get_reward(new_img2, data)
            print(f"-- Text Phase (fixed prefix) New Reward[{i}]: {new_reward:.6f}")
            reward_history.append(new_reward)
            current_reward = new_reward
            last_img = new_img2

            if save_base_path is not None:
                os.makedirs(save_base_path, exist_ok=True)
                p2 = os.path.join(save_base_path, f"optimized_image_text_{i}.png")
                new_img2.save(p2)
                with open(os.path.join(save_base_path, "image_text_text_trace.jsonl"), "a", encoding="utf-8") as f:
                    json.dump({
                        "phase": "text", "step": i,
                        "generated_text": new_generated_text,
                        "image_gen_prompt": f"{data['prompt']}. {new_generated_text}",
                        "reward": float(new_reward),
                        "loss": float(text_loss.item()),
                        "image_path": p2
                    }, f, ensure_ascii=False); f.write("\n")

            if current_reward > reward_threshold:
                print(f"Early Stop after Text Phase: reward {current_reward:.6f} > {reward_threshold}")
                return new_img2, reward_history, (total_img + total_text), (image_token_num + len(generated_seq)), (update_length_img + update_length_text)

        return last_img, reward_history, (total_img + total_text), (image_token_num + len(generated_seq)), (update_length_img + update_length_text)

    # Branch: both
    elif optimize_mode == "text_image":  # "both" —— 串行：先 Text 阶段，达标早停；否则进入 Image 阶段
        print("-- Both (sequential): text phase first, then image phase if needed...")

        # ====== 文本阶段（仅优化文本 hidden states，最多 max_text_steps 步）======
        total_text = len(text_hidden_states_list)
        update_length_text = min(int(text_k * total_text), total_text)
        if update_length_text <= 0:
            print("Text Update Length Zero!!!")
            return None, reward_history, total_text, 0, 0

        total_img = len(image_hidden_states_list)
        update_length_img = min(int(image_k * total_img), total_img)
        if update_length_img <= 0:
            print("Image Update Length Zero!!!")
            return last_img, reward_history, total_text, len(generated_seq), update_length_text

        # 准备文本优化所需前缀
        inputs = tokenizer([sft_prompt], return_tensors="pt").to(device)
        base_input_ids = inputs.input_ids.clone()

        # 原始序列前缀（prompt 后到 start_index 的 token）
        original_seq = []
        original_seq.extend(
            text_final_input_ids[0][len(base_input_ids[-1]) : len(base_input_ids[-1]) + start_index]
        )

        optimized_states_text = torch.nn.Parameter(
            torch.stack(
                [
                    s.clone().detach().requires_grad_(True)
                    for s in text_hidden_states_list[
                        start_index : min(start_index + update_length_text, len(text_hidden_states_list))
                    ]
                ]
            )
        )
        optimizer_text = torch.optim.Adam([optimized_states_text], lr=lr)
        
        optimized_states_img = torch.nn.Parameter(
            torch.stack(
                [
                    s.clone().detach().to(device).requires_grad_(True)
                    for s in image_hidden_states_list[start_index : start_index + update_length_img]
                ]
            )
        )

        input_ids = text_final_input_ids[:, : len(base_input_ids[-1]) + start_index]
        base_input_ids = input_ids.clone().to(device)

        last_generated_text = None
        last_img = image  # 文本阶段每步会生成对应图片并评估 reward
        generated_seq = []

        # 若初始就已超过阈值，直接退出
        if current_reward > reward_threshold:
            print(f"Early stop before Text phase: reward {current_reward:.6f} > {reward_threshold}")
            return last_img, reward_history, total_text, 0, 0

        for i in range(max_text_steps):
            optimizer_text.zero_grad()

            # 计算文本阶段 loss（基于当前 optimized_states_text）
            logits = model.language_model.lm_head(optimized_states_text)             # [update_len, vocab]
            probs = torch.softmax(logits, dim=-1) + 1e-8
            next_token_ids = torch.argmax(probs, dim=-1).squeeze(-1)                # [update_len]
            # 取每个位置的 top-1 对应概率
            log_pi = torch.log(probs[torch.arange(update_length_text), 0, next_token_ids] + 1e-10)

            text_loss = - current_reward * log_pi.sum()
            print(f"-- Text Phase Loss[{i}]: {text_loss.item():.6f}")
            text_loss.backward(retain_graph=True)
            if grad_clip:
                torch.nn.utils.clip_grad_norm_([optimized_states_text], grad_clip)
            optimizer_text.step()

            # 用更新后的 states 展开文本，直到 stop word 或长度上限
            generated_seq = []
            generated_seq.extend(original_seq)
            with torch.no_grad():
                updated_probs = torch.softmax(model.language_model.lm_head(optimized_states_text), dim=-1)
                updated_token_ids = torch.argmax(updated_probs, dim=-1).squeeze(-1)
                generated_seq.extend(updated_token_ids.tolist())
                updated_input_ids = torch.cat([base_input_ids, updated_token_ids.unsqueeze(0)], dim=-1)

                steps = 0
                while True:
                    outputs = model.language_model.model(updated_input_ids, output_hidden_states=True)
                    last_hidden = outputs[0][:, -1]
                    nxt = torch.argmax(model.language_model.lm_head(last_hidden), dim=-1)
                    tok = tokenizer.decode(nxt.item())
                    generated_seq.append(nxt.item())
                    updated_input_ids = torch.cat([updated_input_ids, nxt.unsqueeze(0)], dim=-1)
                    steps += 1
                    if tok in stop_words or steps > max_text_tokens:
                        break

            new_generated_text = tokenizer.decode(generated_seq, skip_special_tokens=True)
            last_generated_text = new_generated_text
            print(f"New Optimized Text[{i}]: {new_generated_text}")

            # 用最新文本生成图像并评估 reward（image 不做优化，只重生成）
            image_gen_prompt = f"{data['prompt']}. {new_generated_text}"
            save_path = None
            if save_base_path is not None:
                os.makedirs(save_base_path, exist_ok=True)
                save_path = os.path.join(save_base_path, f"optimized_image_text_{i}.png")

            new_img = generate_image_from_prompt_and_states(model, vl_chat_processor, tokenizer, image_gen_prompt, optimized_states_img, update_length_img,save_path=save_path)  ##optimize the origin image rather than the optimized image
            
            last_img = new_img

            new_reward = reward_model.get_reward(new_img, data)
            print(f"-- Text Phase New Reward[{i}]: {new_reward:.6f}")
            reward_history.append(new_reward)
            current_reward = new_reward

            # 记录 trace
            if save_base_path is not None:
                with open(os.path.join(save_base_path, "both_text_trace.jsonl"), "a", encoding="utf-8") as f:
                    json.dump({
                        "phase": "text",
                        "step": i,
                        "generated_text": new_generated_text,
                        "image_gen_prompt": image_gen_prompt,
                        "reward": float(new_reward),
                        "loss": float(text_loss.item()),
                        "image_path": f"optimized_image_text_{i}.png" if save_path else None
                    }, f, ensure_ascii=False)
                    f.write("\n")

            # 早停：文本阶段达阈值直接退出
            if current_reward > reward_threshold:
                print(f"Early Stop after Text Phase: reward {current_reward:.6f} > {reward_threshold}")
                return new_img, reward_history, total_text, len(generated_seq), update_length_text

        # ====== 图像阶段（仅优化图像 hidden states，最多 max_image_steps 步）======
        print("-- Switching to Image Phase with the last optimized text...")
        # if last_generated_text is None:
        #     last_generated_text = ""

        image_gen_prompt = f"{data['prompt']}. {last_generated_text}"
        # 根据最新文本重建 image_prompt_embed（与原 both 分支一致）
        img_conv = [{"role": "User", "content": image_gen_prompt}, {"role": "Assistant", "content": ""}]
        sft_image_prompt = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=img_conv, sft_format=vl_chat_processor.sft_format, system_prompt=""
        )
        prompt_inputs = tokenizer(
            text=[sft_image_prompt], return_tensors="pt", padding=True, padding_side="right", add_special_tokens=True
        )
        image_prompt_ids = prompt_inputs["input_ids"].to(device)
        image_start_token_id = tokenizer.encode(vl_chat_processor.image_start_tag)[1]
        image_prompt_ids = torch.cat(
            [image_prompt_ids, image_prompt_ids.new_full((image_prompt_ids.size(0), 1), image_start_token_id)],
            dim=1
        ).repeat(1, 1)
        cond_inputs_embeds = model.language_model.get_input_embeddings()(image_prompt_ids)
        pad_input_embeds = model.language_model.get_input_embeddings()(
            image_prompt_ids.new_full((1, 1), vl_chat_processor.pad_id)
        )
        uncond_inputs_embeds = cond_inputs_embeds.clone()
        uncond_inputs_embeds[:, 1:-1] = pad_input_embeds
        image_prompt_embed = torch.repeat_interleave(cond_inputs_embeds, 2, dim=0)
        image_prompt_embed[1::2] = uncond_inputs_embeds

        # total_img = len(image_hidden_states_list)
        # update_length_img = min(int(image_k * total_img), total_img)
        # if update_length_img <= 0:
        #     print("Image Update Length Zero!!!")
        #     return last_img, reward_history, total_text, len(generated_seq), update_length_text

        # optimized_states_img = torch.nn.Parameter(
        #     torch.stack(
        #         [
        #             s.clone().detach().to(device).requires_grad_(True)
        #             for s in image_hidden_states_list[start_index : start_index + update_length_img]
        #         ]
        #     )
        # )
        optimizer_img = torch.optim.Adam([optimized_states_img], lr=lr)

        final_img = last_img
        img_outputs = None

        # 若已经达阈值，也直接退出（极端情况下文本阶段末尾已经满足）
        if current_reward > reward_threshold:
            print(f"Early stop before Image phase: reward {current_reward:.6f} > {reward_threshold}")
            return final_img, reward_history, (total_text + total_img), (image_token_num + len(generated_seq)), (update_length_text + update_length_img)

        for i in range(max_image_steps):
            optimizer_img.zero_grad()

            logits = model.gen_head(optimized_states_img)  # [update_len, 2, vocab]（按你原本的假设）
            logit_cond = logits[:, 0, :]
            logit_uncond = logits[:, 1, :]
            fused_logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(fused_logits / temperature, dim=-1) + 1e-8
            token_ids = torch.argmax(probs, dim=-1)
            log_pi = torch.log(probs[torch.arange(update_length_img), token_ids] + 1e-10)

            image_loss = - current_reward * log_pi.sum()
            print(f"-- Image Phase Loss[{i}]: {image_loss.item():.6f}")
            image_loss.backward()
            optimizer_img.step()

            # 用更新后的 optimized_states_img 采样若干 token，并与 image_prompt_embed 拼接生成完整图
            with torch.no_grad():
                logits = model.gen_head(optimized_states_img)
                logit_cond = logits[:, 0, :]
                logit_uncond = logits[:, 1, :]
                fused_logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
                probs = torch.softmax(fused_logits / temperature, dim=-1)

                sampled_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [update_len]
                optimized_token_embeds = model.prepare_gen_img_embeds(
                    sampled_token_ids.repeat_interleave(2).view(-1, 2).reshape(-1)
                ).reshape(update_length_img, 2, -1).permute(1, 0, 2)
                inputs_embeds_img = torch.cat([image_prompt_embed, optimized_token_embeds], dim=1)

                # 继续自回归生成剩余 token
                generated_tokens = torch.zeros((1, image_token_num), dtype=torch.int).to(device)
                generated_tokens[:, :update_length_img] = sampled_token_ids
                current_embeds = inputs_embeds_img
                img_outputs = None
                for j in range(update_length_img, image_token_num):
                    img_outputs = model.language_model.model(
                        inputs_embeds=current_embeds,
                        use_cache=True,
                        past_key_values=img_outputs.past_key_values if img_outputs is not None else None
                    )
                    last_hidden = img_outputs.last_hidden_state[:, -1, :]
                    lg = model.gen_head(last_hidden)
                    lc = lg[0::2, :]
                    lu = lg[1::2, :]
                    lg = lu + cfg_weight * (lc - lu)
                    nxt_tok = torch.multinomial(torch.softmax(lg / temperature, dim=-1), num_samples=1)
                    generated_tokens[:, j] = nxt_tok.squeeze(dim=-1)
                    nxt_ids = nxt_tok.repeat(1, 2).view(-1)
                    nxt_embeds = model.prepare_gen_img_embeds(nxt_ids).unsqueeze(1)
                    current_embeds = nxt_embeds

                decoded = model.gen_vision_model.decode_code(
                    generated_tokens.to(dtype=torch.int),
                    shape=[1, 8, img_size // patch_size, img_size // patch_size],
                )
                decoded = decoded.detach().to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
                decoded = np.clip((decoded + 1) / 2 * 255, 0, 255).astype(np.uint8)
                final_img = PIL.Image.fromarray(decoded[0])

            new_reward = reward_model.get_reward(final_img, data)
            print(f"-- Image Phase New Reward[{i}]: {new_reward:.6f}")
            reward_history.append(new_reward)
            current_reward = new_reward

            if save_base_path is not None:
                os.makedirs(save_base_path, exist_ok=True)
                img_path = os.path.join(save_base_path, f"optimized_image_img_{i}.png")
                final_img.save(img_path)
                with open(os.path.join(save_base_path, "both_image_trace.jsonl"), "a", encoding="utf-8") as f:
                    json.dump({
                        "phase": "image",
                        "step": i,
                        "image_gen_prompt": image_gen_prompt,
                        "reward": float(new_reward),
                        "loss": float(image_loss.item()),
                        "image_path": img_path
                    }, f, ensure_ascii=False)
                    f.write("\n")

            # 早停：图像阶段达阈值直接退出
            if current_reward > reward_threshold:
                print(f"Early Stop after Image Phase: reward {current_reward:.6f} > {reward_threshold}")
                return final_img, reward_history, (total_text + total_img), (image_token_num + len(generated_seq)), (update_length_text + update_length_img)

        # 两阶段都未达到阈值，返回最终结果
        return final_img, reward_history, (total_text + total_img), (image_token_num + len(generated_seq)), (update_length_text + update_length_img)


# model_path = "deepseek-ai/Janus-Pro-7B"
# vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)

# vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
#     model_path, trust_remote_code=True
# )
# vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
# input_text = "a photo of a purple suitcase and an orange pizza"
# data = {"tag": "color_attr", "include": [{"class": "suitcase", "count": 1, "color": "purple"}, {"class": "pizza", "count": 1, "color": "orange"}], "prompt": "a photo of a purple suitcase and an orange pizza"}

# reward_model = RewardModel(
#     model_path="/fs-computility/ai-shen/fanyuyu/latentseek/Multimodal-LatentSeek/src/rewards/<OBJECT_DETECTOR_FOLDER>",
#     object_names_path="/fs-computility/ai-shen/fanyuyu/latentseek/Multimodal-LatentSeek/src/rewards/object_names.txt",
#     options={"clip_model": "ViT-L-14"}
# )

# answer, text_hidden_states_list, text_final_input_ids, image_hidden_states_list, image_prompt_embed, generated_image_tokens = original_generation(input_text, vl_gpt, vl_chat_processor, optimize_mode="interleaved_text_image",device=torch.device("cuda"))
# new_img, reward_history, ori_length, generated_seq, update_length = optimized_generation(reward_model, answer, data, vl_gpt, vl_chat_processor, torch.device("cuda"), text_hidden_states_list, text_final_input_ids, image_hidden_states_list, image_prompt_embed, generated_image_tokens, optimize_mode="interleaved_text_image", save_base_path="/fs-computility/ai-shen/fanyuyu/latentseek/Multimodal-LatentSeek/src/geneval_results/test")