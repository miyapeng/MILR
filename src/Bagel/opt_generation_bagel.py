# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import List, Dict, Optional, Union, Any

from PIL import Image
import PIL.Image
import torch
from torch import nn

from data.data_utils import pil_img2rgb
from modeling.bagel.qwen2_navit import NaiveCache


VLM_THINK_SYSTEM_PROMPT = '''You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here'''

GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''


class opt_InterleaveInferencer:
    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        
    def init_gen_context(self): 
        gen_context = {
            'kv_lens': [0],
            'ropes': [0],
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }
        return gen_context

    @torch.no_grad()
    def update_context_text(self, text, gen_context):
        # used for interleave data, currently only support 1 data inference, 

        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            prompts=[text],
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )

        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)        
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=True, vit=True):
        # used for interleave data, currently only support 1 data inference, 

        assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes =  gen_context['ropes']

        if vae:
            ## update vae
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vae_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)
        
        if vit:
            ## update vit
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vit_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def gen_image(
        self, 
        image_shape, 
        gen_context, 
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,

        cfg_text_precontext=None, 
        cfg_img_precontext=None, 
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        
        num_timesteps=50, 
        timestep_shift=3.0
    ):
        # print(cfg_renorm_type)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            image_sizes=[image_shape], 
            new_token_ids=self.new_token_ids,
        ) 
        
        # text cfg
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
        )

        image = self.decode_image(unpacked_latent[0], image_shape)
        return image

        
    def decode_image(self, latent, image_shape):
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image

    @torch.no_grad()
    def gen_text(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0):
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        unpacked_latent, text_hideen_states = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            **generation_input,
        )
        output = self.tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
        return output,text_hideen_states
    
    @torch.no_grad()
    def opt_gen_text(self, gen_context, optimized_states, max_length: int = 500, do_sample: bool = False, temperature: float = 1.0):
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        packed_start_tokens = generation_input['packed_start_tokens']
        packed_query_position_ids = generation_input['packed_query_position_ids']
        packed_key_value_indexes = generation_input['packed_key_value_indexes']
        key_values_lens = generation_input['key_values_lens']
    
        step = 0
        generated_sequence = []
        curr_tokens = packed_start_tokens
        end_token_id=self.new_token_ids['eos_token_id']

        logits = self.model.language_model.lm_head(optimized_states)
        if do_sample:
            probs = nn.functional.softmax(logits / temperature, dim=-1)
            previous_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            previous_tokens = previous_tokens.to(curr_tokens.device)
        else:
            previous_tokens = torch.argmax(logits, dim=-1)
            previous_tokens = previous_tokens.to(curr_tokens.device)

        previous_tokens = previous_tokens.squeeze(1)

        curr_tokens = torch.cat([curr_tokens, previous_tokens])

        # 模拟逐步构建 KV 缓存
        for i in range(curr_tokens.shape[0]):
            prefix_token = curr_tokens[i:i+1]  
            generated_sequence.append(prefix_token)

            packed_text_embedding = self.model.language_model.model.embed_tokens(prefix_token)
            query_lens = torch.ones_like(prefix_token)
            packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
                0, len(key_values_lens), 
                device=key_values_lens.device, 
                dtype=key_values_lens.dtype
            )

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] += i
            packed_key_value_indexes = torch.cat(uppacked, dim=0)

            extra_inputs = {}
            extra_inputs = {"mode": "und"}

            output = self.model.language_model.forward_inference(
                packed_query_sequence=packed_text_embedding,
                query_lens=query_lens,
                packed_query_position_ids=packed_query_position_ids,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=True,
                is_causal=True,
                **extra_inputs,
            )
            past_key_values = output.past_key_values
            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] = torch.cat(
                    [uppacked[i], torch.tensor([uppacked[i][-1] + 1], device=uppacked[i].device)], dim=0
                )
            packed_key_value_indexes = torch.cat(uppacked, dim=0)
            key_values_lens = key_values_lens + 1
            packed_query_position_ids = packed_query_position_ids + 1

        curr_tokens = curr_tokens[-1:]  

        while step < max_length:
            generated_sequence.append(curr_tokens)
            packed_text_embedding = self.model.language_model.model.embed_tokens(curr_tokens)
            query_lens = torch.ones_like(curr_tokens)
            packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
                0, len(key_values_lens), 
                device=key_values_lens.device, 
                dtype=key_values_lens.dtype
            )

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] += i
            packed_key_value_indexes = torch.cat(uppacked, dim=0)

            extra_inputs = {}
            extra_inputs = {"mode": "und"}

            output = self.model.language_model.forward_inference(
                packed_query_sequence=packed_text_embedding,
                query_lens=query_lens,
                packed_query_position_ids=packed_query_position_ids,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=True,
                is_causal=True,
                **extra_inputs,
            )
            past_key_values = output.past_key_values
            packed_query_sequence = output.packed_query_sequence
            pred_logits = self.model.language_model.lm_head(packed_query_sequence)

            if do_sample:
                probs = nn.functional.softmax(pred_logits / temperature, dim=-1)
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(pred_logits, dim=-1)

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] = torch.cat(
                    [uppacked[i], torch.tensor([uppacked[i][-1] + 1], device=uppacked[i].device)], dim=0
                )
            packed_key_value_indexes = torch.cat(uppacked, dim=0)
            key_values_lens = key_values_lens + 1
            packed_query_position_ids = packed_query_position_ids + 1
            step += 1

            if end_token_id is not None and curr_tokens[0] == end_token_id: # only support batch=1
                break

        output_device = generated_sequence[0].device
        unpacked_latent = torch.stack([i.to(output_device) for i in generated_sequence], dim=0)

        output = self.tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
        return output
        
    #@torch.no_grad()
    def interleave_inference(
        self,
        reward_model,
        ori_img: PIL.Image,
        data,
        input_lists: List[Union[str, Image.Image]],
        text_hidden_states_list: Optional[torch.Tensor] = None,

        max_text_steps=10,
        lr=0.01,
        grad_clip=None,
        text_k=0.1,
        reward_threshold=-0.1,
        optimize_mode="text",

        think=False,
        understanding_output=False,

        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
    ) -> List[Union[str, Image.Image]]:

        reward_history = []
        # start from original image tokens
        initial_reward = reward_model.get_reward(ori_img, data)
        print(f"-- Initial Image Reward: {initial_reward}")
        reward_history.append(initial_reward)
        current_reward = initial_reward

        if optimize_mode == "text":
            #optimization parameters
            total = len(text_hidden_states_list)
            update_length = min(int(text_k * total), total)
            start_index = 0
            img = None

            if update_length <= 0:
                print("Update Length Zero!!!")
                return None, reward_history, total, 0, update_length

            #the hidden states for optimization
            optimized_states = torch.nn.Parameter(torch.stack(
                [s.clone().detach().requires_grad_(True)
                for s in text_hidden_states_list[start_index:min(start_index + update_length, len(text_hidden_states_list))]
                ])
            )
            optimizer = torch.optim.Adam([optimized_states], lr=lr)

            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                for i in range(max_text_steps):
                    if current_reward > reward_threshold:
                        break
                    
                    optimizer.zero_grad()
                    
                    logits = self.model.language_model.lm_head(optimized_states)
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

                    output_list = []
                    gen_context = self.init_gen_context()
                    cfg_text_context = deepcopy(gen_context)
                    cfg_img_context = deepcopy(gen_context)
                    
                    if think:
                        if understanding_output:
                            system_prompt = VLM_THINK_SYSTEM_PROMPT 
                        else:
                            system_prompt = GEN_THINK_SYSTEM_PROMPT
                        gen_context = self.update_context_text(system_prompt, gen_context)
                        cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)

                    for input_term in input_lists:
                        if isinstance(input_term, str):
                            cfg_text_context = deepcopy(gen_context)
                            gen_context = self.update_context_text(input_term, gen_context)
                            cfg_img_context = self.update_context_text(input_term, cfg_img_context)

                        elif isinstance(input_term, Image.Image):
                            input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                            gen_context = self.update_context_image(input_term, gen_context, vae=not understanding_output)

                            image_shapes = input_term.size[::-1]
                            cfg_text_context = deepcopy(gen_context)

                        else:
                            raise ValueError(f"Unsupported input type: {type(input_term)}")
                    
                    if understanding_output:
                        gen_text, _= self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                        output_list.append(gen_text)
                    else:
                        if think:
                            gen_text = self.opt_gen_text(gen_context, optimized_states, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                            print(f"-- Generated Text: {gen_text}")
                            gen_context = self.update_context_text(gen_text, gen_context)
                            output_list.append(gen_text)

                        img = self.gen_image(
                            image_shapes, 
                            gen_context, 
                            cfg_text_precontext=cfg_text_context, 
                            cfg_img_precontext=cfg_img_context,

                            cfg_text_scale=cfg_text_scale, 
                            cfg_img_scale=cfg_img_scale, 
                            cfg_interval=cfg_interval, 
                            timestep_shift=timestep_shift, 
                            num_timesteps=num_timesteps,
                            cfg_renorm_min=cfg_renorm_min,
                            cfg_renorm_type=cfg_renorm_type,
                        )

                        img.save(f"opt_image_{i}.png")
                        with torch.cuda.amp.autocast(enabled=False):
                            new_reward = reward_model.get_reward(img, data)
                            reward_history.append(new_reward)
                            current_reward = new_reward

                output_list.append(img)

        return output_list
    
    def __call__(
        self, 
        reward_model,
        ori_img,
        data,
        image: Optional[Image.Image] = None, 
        text: Optional[str] = None,
        text_hidden_states: list = None,
        max_text_steps=10,
        lr=0.01,
        grad_clip=None,
        text_k=0.1,
        reward_threshold=-0.1,
        optimize_mode="text",
        **kargs
    ) -> Dict[str, Any]:
        output_dict = {'image': None, 'text': None}

        if image is None and text is None:
            print('Please provide at least one input: either an image or text.')
            return output_dict

        input_list = []
        if image is not None:
            input_list.append(image)
        if text is not None:
            input_list.append(text)

        output_list = self.interleave_inference(reward_model=reward_model, 
                                                ori_img=ori_img, 
                                                data=data, 
                                                input_lists=input_list, 
                                                text_hidden_states_list=text_hidden_states, 
                                                max_text_steps=max_text_steps, 
                                                lr=lr, 
                                                grad_clip=grad_clip, 
                                                text_k=text_k, 
                                                reward_threshold=reward_threshold,
                                                optimize_mode=optimize_mode,
                                                **kargs)

        for i in output_list:
            if isinstance(i, Image.Image):
                output_dict['image'] = i
            elif isinstance(i, str):
                output_dict['text'] = i
        return output_dict
