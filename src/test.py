# input_text = "A red apple on a wooden table with a blue vase beside it."
# cot_prompt = (
#         'You are asked to generate an image based on this prompt: "{}"\n'
#         'Provide a brief, precise visualization of all elements in the prompt. Your description should:\n'
#         '1. Include every object mentioned in the prompt\n'
#         '2. Specify visual attributes (color, number, shape, texture) if specified in the prompt\n'
#         '3. Clarify relationships (e.g., spatial) between objects if specified in the prompt\n'
#         '4. Be concise (50 words or less)\n'
#         "5. Focus only on what's explicitly stated in the prompt\n"
#         '6. Do not elaborate beyond the attributes or relationships specified in the prompt\n'
#         'Do not miss objects. Output your visualization directly without explanation:'
#     )
# formatted_cot_prompt = cot_prompt.format(input_text)
# print(formatted_cot_prompt)

# from process import get_dataset
# dataset = get_dataset("prompts/Wise/cultural_common_sense.json",task_type="text", data_name="Wise")
# print(f"Example: {dataset[1]}")

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
# from copy import deepcopy
# from typing import (
#     Any,
#     AsyncIterable,
#     Callable,
#     Dict,
#     Generator,
#     List,
#     NamedTuple,
#     Optional,
#     Tuple,
#     Union,
# )
# import requests
# from io import BytesIO

# from PIL import Image
# import torch
# from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

# from Bagel.data.transforms import ImageTransform
# from Bagel.data.data_utils import pil_img2rgb, add_special_tokens
# from Bagel.modeling.bagel import (
#     BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
# )
# from Bagel.modeling.qwen2 import Qwen2Tokenizer
# from Bagel.modeling.bagel.qwen2_navit import NaiveCache
# from Bagel.modeling.autoencoder import load_ae
# from safetensors.torch import load_file

# model_path = "models/BAGEL-7B-MoT"  # Download from https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT

# # LLM config preparing
# llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
# llm_config.qk_norm = True
# llm_config.tie_word_embeddings = False
# llm_config.layer_module = "Qwen2MoTDecoderLayer"

# # ViT config preparing
# vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
# vit_config.rope = False
# vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

# # VAE loading
# vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

# # Bagel config preparing
# config = BagelConfig(
#     visual_gen=True,
#     visual_und=True,
#     llm_config=llm_config, 
#     vit_config=vit_config,
#     vae_config=vae_config,
#     vit_max_num_patch_per_side=70,
#     connector_act='gelu_pytorch_tanh',
#     latent_patch_size=2,
#     max_latent_size=64,
# )

# with init_empty_weights():
#     language_model = Qwen2ForCausalLM(llm_config)
#     vit_model      = SiglipVisionModel(vit_config)
#     model          = Bagel(language_model, vit_model, config)
#     model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

# # Tokenizer Preparing
# tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
# tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

# # Image Transform Preparing
# vae_transform = ImageTransform(1024, 512, 16)
# vit_transform = ImageTransform(980, 224, 14)

# from accelerate import load_checkpoint_and_dispatch
# import os

# # 强制单卡加载
# device_map = {"": "cuda:0"}
# print(device_map)

# model = load_checkpoint_and_dispatch(
#     model,
#     checkpoint=os.path.join(model_path, "ema.safetensors"),
#     device_map=device_map,            # ✅ 现在是 dict 类型
#     offload_buffers=True,
#     dtype=torch.bfloat16,
#     force_hooks=True,
#     offload_folder="/tmp/offload"
# )

# model = model.eval()
# print('Model loaded')

# from Bagel.inferencer import InterleaveInferencer

# inferencer = InterleaveInferencer(
#     model=model, 
#     vae_model=vae_model, 
#     tokenizer=tokenizer, 
#     vae_transform=vae_transform, 
#     vit_transform=vit_transform, 
#     new_token_ids=new_token_ids
# )

# import random
# import numpy as np

# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# inference_hyper=dict(
#     max_think_token_n=1000,
#     do_sample=False,
#     cfg_text_scale=4.0,
#     cfg_img_scale=1.0,
#     cfg_interval=[0.4, 1.0],
#     timestep_shift=3.0,
#     num_timesteps=50,
#     cfg_renorm_min=0,
#     cfg_renorm_type="global",
# )

# #prompt = "A female cosplayer portraying an ethereal fairy or elf, wearing a flowing dress made of delicate fabrics in soft, mystical colors like emerald green and silver. She has pointed ears, a gentle, enchanting expression, and her outfit is adorned with sparkling jewels and intricate patterns. The background is a magical forest with glowing plants, mystical creatures, and a serene atmosphere."
# prompt = "A photo of four sandwiches."
# print(prompt)
# output_dict = inferencer(text=prompt, think=True, **inference_hyper)

# # # 显示图像
# # output_dict['image'].show()  # ✅ 推荐做法

# # # 或保存图像
# output_dict['image'].save("generated_image.jpg")

# import torch

# state_dict = torch.load("/media/raid/workspace/miyapeng/Multimodal-LatentSeek/src/results/Janus-Pro-7B-geneval-text-text_k0.1-image_k0.01-lr0.01/logistics.pt", map_location="cpu")
# for key, value in state_dict.items():
#     print(f"{key}: {value}")

import torch
from transformers import AutoModel
from PIL import Image

# 模型路径
model_path = "Efficient-Large-Model/NVILA-Lite-2B-Verifier"

device = torch.device("cuda:0")

# 加载模型
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)

# 获取 yes/no 的 token ID（用于查看分数）
yes_id = model.tokenizer.encode("yes", add_special_tokens=False)[0]
no_id = model.tokenizer.encode("no", add_special_tokens=False)[0]

# 输入图像路径和 prompt
image_path = "/media/raid/workspace/miyapeng/Multimodal-LatentSeek/src/geneval_results/long_results/Janus-Pro-7B-geneval-geneval-both-text_k0.2-image_k0.02-steps30-lr0.03-reward_threshold-0.1/final_img/00000/samples/0000.png"
prompt_text = "a photo of a bench."

# 构造 prompt
prompt = f"""You are an AI assistant specializing in image analysis and ranking. Your task is to analyze and compare image based on how well they match the given prompt. The given prompt is:{prompt_text}. Please consider the prompt and the image to make a decision and response directly with 'yes' or 'no'."""

# 打开图片
image = Image.open(image_path)

# 模型推理
response, scores = model.generate_content([image, prompt])

# 输出判断结果与 logits
yes_score = scores[0][0, yes_id].item()
no_score = scores[0][0, no_id].item()
confidence = yes_score - no_score

print(f"Model response: {response}")
print(f"Yes score: {yes_score:.4f}, No score: {no_score:.4f}, Confidence: {confidence:.4f}")
