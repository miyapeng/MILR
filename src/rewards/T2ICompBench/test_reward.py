# ### Geneval Reward test script
# from reward import RewardModel
# from PIL import Image

# model = RewardModel(
#     model_path="<OBJECT_DETECTOR_FOLDER>",
#     object_names_path="object_names.txt",
#     options={"clip_model": "ViT-L-14"}
# )
# img = Image.open("/media/raid/workspace/miyapeng/Bagel/eval/gen/geneval/results/geneval_no_think/images/00530/samples/00000.png")
# solution = '{"tag": "color_attr", "include": [{"class": "suitcase", "count": 1, "color": "purple"}, {"class": "pizza", "count": 1, "color": "orange"}], "prompt": "a photo of a purple suitcase and an orange pizza"}'

# reward = model.get_reward(img, solution)
# print(f"Reward: {reward}")

# import os
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# from tqdm import tqdm
# import argparse
# import numpy as np
# import random
# import json
# from PIL import Image

# from janus.models import MultiModalityCausalLM, VLChatProcessor
# from rewards.self_reward import SelfRewardModel

# model_path = "deepseek-ai/Janus-Pro-7B"
# vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
# tokenizer = vl_chat_processor.tokenizer

# vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
#     model_path, trust_remote_code=True
# )
# vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# model = SelfRewardModel(
#     vl_gpt=vl_gpt,
#     vl_chat_processor=vl_chat_processor,
#     device="cuda"
# )

# img = Image.open("/media/raid/workspace/miyapeng/Multimodal-LatentSeek/src/geneval_results/test/optimized_image_1.png")
# solution = '{"tag": "color_attr", "include": [{"class": "suitcase", "count": 1, "color": "purple"}, {"class": "pizza", "count": 1, "color": "orange"}], "prompt": "a photo of a purple suitcase and an orange pizza"}'

# reward = model.get_reward(img, solution)
# print(f"Reward: {reward}")
# test_unified_reward.py

# ## Test UnifiedReward
# import json
# from PIL import Image
# from rewards.unified_reward import UnifiedReward  # 假设你把类保存到 unified_reward.py

# if __name__ == "__main__":
#     # 1. 实例化模型（替换成你自己的路径）
#     model = UnifiedReward(
#         model_path="CodeGoat24/UnifiedReward-qwen-7b",
#         device="cuda:0"
#     )

#     # 2. 读取测试图片
#     img = Image.open("/media/raid/workspace/miyapeng/Multimodal-LatentSeek/src/geneval_results/test/optimized_image_9.png")

#     # 3. 构造 solution JSON（与 get_reward 接口一致）
#     solution = '{"tag": "color_attr", "include": [{"class": "suitcase", "count": 1, "color": "purple"}, {"class": "pizza", "count": 1, "color": "orange"}], "prompt": "a photo of a purple suitcase and an orange pizza"}'

#     # 4. 调用 get_reward
#     reward = model.get_reward(img, solution)
#     print(f"Raw Reward (mapped): {reward}")  
#     # 这里 reward == 0 表示模型给了满分；< 0 表示越低分，越不匹配

#     # 5. 调用 judge_answer
#     is_correct = model.judge_answer(img, solution)
#     print(f"Is Correct? {is_correct}")

import json
from PIL import Image
from reward import CompBenchRewardModel
if __name__ == "__main__":
    # 1. 实例化模型（替换成你自己的路径）
    reward_model = CompBenchRewardModel(
        task_type='blip_vqa',
        device='cuda:0'
    )

    # 2. 读取测试图片
    img = Image.open("/media/raid/workspace/miyapeng/T2I-CompBench/Janus_test/results/color/samples/a_green_bench_and_a_blue_bowl_000000.png")

    # 3. 构造 solution JSON（与 get_reward 接口一致）
    solution = {'tag': 'color_val', 'prompt': 'a green bench and a blue bowl'}

    reward = reward_model.get_reward(img, solution)
    print(f"Raw Reward (mapped): {reward}")  
