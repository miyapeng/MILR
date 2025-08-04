import json
import torch
from transformers import AutoModel
from PIL import Image

class NVILAReward:
    """
    Reward model (binary) based on NVILA-like yes/no verifier.
    - get_reward(image, solution) -> 0 (yes) or -1 (no)
    - judge_answer(image, solution) -> bool (True for 0 / False for -1)
    """

    def __init__(self, model_path: str = "Efficient-Large-Model/NVILA-Lite-2B-Verifier",
                 device: str = "cuda:0"):
        """
        Args:
            model_path: HF repo or local path for the NVILA verifier.
            device: e.g., 'cuda:0' or 'cpu'. 建议单卡，避免 device_map="auto" 跨卡拼接错误。
        """
        self.device = device
        # 加载模型到单卡，避免跨卡 cat 报错
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True
        ).to(torch.device(device)).eval()

        # 取 yes/no 的 token id（用于 logits 回退决策）
        self.yes_id = self.model.tokenizer.encode("yes", add_special_tokens=False)[0]
        self.no_id  = self.model.tokenizer.encode("no",  add_special_tokens=False)[0]

    def _ensure_image(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            # 若需要可统一转 RGB
            if image.mode != "RGB":
                image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        return image

    def _ensure_prompt(self, solution):
        if isinstance(solution, str):
            meta = json.loads(solution)
        elif isinstance(solution, dict):
            meta = solution
        else:
            raise ValueError(f"Unsupported solution type: {type(solution)}")
        prompt = meta.get("prompt")
        if not prompt:
            raise ValueError("Missing 'prompt' (or 'question') in solution.")
        return prompt

    def _build_prompt(self, prompt_text: str) -> str:
        # 与你现有单图判断保持一致的指令风格
        return (
            f"You are an AI assistant specializing in image analysis and ranking. "
            f"Your task is to analyze the image based on how well it matches the given prompt. "
            f"The given prompt is: {prompt_text} "
            f"Please consider the prompt and the image, and respond directly with 'yes' or 'no'."
        )


    def _decide_label(self, response: str, scores):
        """
        将模型输出映射为 0 或 -1。
        优先用文本 'yes'/'no'；如不规范，则用 yes/no logits 比较回退。
        """
        if isinstance(response, str):
            resp = response.strip().lower()
            if resp == "yes":
                return 0
            if resp == "no":
                return -1

        # 回退：比较 logits
        try:
            yes_score = scores[0][0, self.yes_id].item()
            no_score  = scores[0][0, self.no_id].item()
            return 0 if yes_score >= no_score else -1
        except Exception:
            # 再次回退：保守判 -1
            return -1

    def get_reward(self, image, solution):
        """
        Returns:
            int: 0 (yes) or -1 (no). 返回 None 表示异常。
        """
        try:
            img = self._ensure_image(image)
            prompt_text = self._ensure_prompt(solution)
            prompt = self._build_prompt(prompt_text)

            # NVILA 接口：generate_content([image, prompt]) -> (response:str, scores:tensor)
            with torch.no_grad():
                response, scores = self.model.generate_content([img, prompt])

            label = self._decide_label(response, scores)
            return label  # 0 或 -1
        except Exception as e:
            print(f"[NVILAReward][ERR] {e}")
            return None

    def judge_answer(self, image, solution) -> bool:
        """
        True  -> yes (0)
        False -> no (-1) 或异常
        """
        label = self.get_reward(image, solution)
        return (label == 0)

# if __name__ == "__main__":
#     img = Image.open("/media/raid/workspace/miyapeng/Multimodal-LatentSeek/src/geneval_results/long_results/Janus-Pro-7B-geneval-geneval-both-text_k0.2-image_k0.02-steps30-lr0.03-reward_threshold-0.1/final_img/00003/samples/0000.png").convert("RGB")

#     # 2. 构造 prompt 数据字典
#     solution = {"tag": "single_object", "include": [{"class": "clock", "count": 1}], "prompt": "a photo of a clock"}

#     rm = NVILAReward(
#         model_path="Efficient-Large-Model/NVILA-Lite-2B-Verifier",
#         device="cuda:0",
#     )
#     score = rm.get_reward(img, solution)  # 0 或 -1
#     print("Reward:", score)
#     print("Pass? ", rm.judge_answer(img, solution))  # True/False
