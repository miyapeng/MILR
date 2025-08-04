# from PIL import Image
# import io
# import base64
# import re
# from typing import Dict, Any, Optional
# from openai import AzureOpenAI

# class GPT4oReward:
#     """
#     用 OpenAI(AzureOpenAI) 作为图文评估模型。
#     对单张图片与文本 prompt 的对齐程度打分，并进一步做二分类（0/ -1）。
#     """
#     def __init__(
#         self,
#         api_key: str,
#         model: str,
#         api_base: Optional[str] = None,
#         region: str = "eastus2",
#         threshold: float = -0.5  # 二分类阈值（基于 final_score ∈ [-1, 0]）
#     ):
#         # 允许外部传入 api_base；如未传则使用你的代理配置
#         API_BASE = api_base or "http://123.127.249.51/proxy"
#         ENDPOINT = f"{API_BASE}/{region}"

#         self.client = AzureOpenAI(
#             api_key=api_key,
#             api_version="2025-03-01-preview",
#             azure_endpoint=ENDPOINT,
#         )
#         self.model = model
#         self.threshold = threshold  # 保存阈值

#     def _encode_image(self, image: Image.Image) -> str:
#         with io.BytesIO() as buffer:
#             image.save(buffer, format="PNG")
#             png_bytes = buffer.getvalue()
#         return base64.b64encode(png_bytes).decode()

#     def _build_messages(
#         self,
#         prompt_data: Dict[str, Any],
#         img64: str
#     ) -> list:
#         return [
#             {
#                 "role": "system",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": "You are a professional Vincennes image quality audit expert, please evaluate the image quality strictly according to the protocol."
#                     }
#                 ]
#             },
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": f"""Please evaluate strictly and return ONLY the three scores as requested.

# # Text-to-Image Quality Evaluation Protocol

# ## System Instruction
# You are an AI quality auditor for text-to-image generation. Apply these rules with ABSOLUTE RUTHLESSNESS. Only images meeting the HIGHEST standards should receive top scores.

# **Input Parameters**  
# - PROMPT: [User's original prompt to]
# ---

# ## Scoring Criteria

# **Consistency (0-2):**  How accurately and completely the image reflects the PROMPT.
# * **0 (Rejected):**  Fails to capture key elements of the prompt, or contradicts the prompt.
# * **1 (Conditional):** Partially captures the prompt. Some elements are present, but not all, or not accurately.  Noticeable deviations from the prompt's intent.
# * **2 (Exemplary):**  Perfectly and completely aligns with the PROMPT.  Every single element and nuance of the prompt is flawlessly represented in the image. The image is an ideal, unambiguous visual realization of the given prompt.

# **Realism (0-2):**  How realistically the image is rendered.
# * **0 (Rejected):**  Physically implausible and clearly artificial. Breaks fundamental laws of physics or visual realism.
# * **1 (Conditional):** Contains minor inconsistencies or unrealistic elements.  While somewhat believable, noticeable flaws detract from realism.
# * **2 (Exemplary):**  Achieves photorealistic quality, indistinguishable from a real photograph.  Flawless adherence to physical laws, accurate material representation, and coherent spatial relationships. No visual cues betraying AI generation.

# **Aesthetic Quality (0-2):**  The overall artistic appeal and visual quality of the image.
# * **0 (Rejected):**  Poor aesthetic composition, visually unappealing, and lacks artistic merit.
# * **1 (Conditional):**  Demonstrates basic visual appeal, acceptable composition, and color harmony, but lacks distinction or artistic flair.
# * **2 (Exemplary):**  Possesses exceptional aesthetic quality, comparable to a masterpiece.  Strikingly beautiful, with perfect composition, a harmonious color palette, and a captivating artistic style. Demonstrates a high degree of artistic vision and execution.

# ---

# ## Output Format

# **Do not include any other text, explanations, or labels.** You must return only three lines of text, each containing a metric and the corresponding score, for example:

# **Example Output:**
# Consistency: 2
# Realism: 1
# Aesthetic Quality: 0

# ---

# **IMPORTANT Enforcement:**

# Be EXTREMELY strict in your evaluation. A score of '2' should be exceedingly rare and reserved only for images that truly excel and meet the highest possible standards in each metric. If there is any doubt, downgrade the score.

# For **Consistency**, a score of '2' requires complete and flawless adherence to every aspect of the prompt, leaving no room for misinterpretation or omission.

# For **Realism**, a score of '2' means the image is virtually indistinguishable from a real photograph in terms of detail, lighting, physics, and material properties.

# For **Aesthetic Quality**, a score of '2' demands exceptional artistic merit, not just pleasant visuals.

# --- 
# Here are the Prompt and EXPLANATION for this evaluation:
# PROMPT: "{prompt_data['prompt']}"
# Please strictly adhere to the scoring criteria and follow the template format when providing your results."""
#                     },
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": f"data:image/png;base64,{img64}"
#                         }
#                     }
#                 ]
#             }
#         ]

#     def _extract_scores(self, text: str) -> Dict[str, float]:
#         """
#         从模型文本输出中解析三项分数。
#         返回键：'consistency', 'realism', 'aesthetic_quality'
#         """
#         pat = r"\*{0,2}(Consistency|Realism|Aesthetic Quality)\*{0,2}\s*[::]?\s*(\d)"
#         matches = re.findall(pat, text, re.IGNORECASE)
#         out = {}
#         for k, v in matches:
#             out[k.lower().replace(" ", "_")] = float(v)
#         return out

#     def _calculate_wiscore(self, consistency, realism, aesthetic_quality):
#         """
#         WiScore = (0.7*consistency + 0.2*realism + 0.1*aesthetic) / 2
#         取值范围约在 [0, 1]；后续 final_score = WiScore - 1.0 ∈ [-1, 0]
#         """
#         return (0.7 * consistency + 0.2 * realism + 0.1 * aesthetic_quality) / 2

#     def get_reward(
#         self,
#         image: Image.Image,
#         prompt_data: Dict[str, Any]
#     ) -> Optional[float]:
#         """
#         评估单样本并返回一个连续分数 final_score ∈ [-1, 0]。
#         返回：
#             float: final_score
#             None: 出错时
#         """
#         try:
#             # 1) Encode image
#             img64 = self._encode_image(image)
#             # 2) Build messages and call OpenAI API
#             messages = self._build_messages(prompt_data, img64)
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=messages,
#                 temperature=0.0,
#                 max_tokens=2000
#             )
#             content = response.choices[0].message.content
#             print(f"gpt output\n{content}")

#             # 3) Extract scores
#             scores = self._extract_scores(content)
#             consistency = scores.get('consistency', 0.0)
#             realism = scores.get('realism', 0.0)
#             aesthetic_quality = scores.get('aesthetic_quality', 0.0)

#             # 4) Continuous score → [-1, 0]
#             wisescore = self._calculate_wiscore(consistency, realism, aesthetic_quality)
#             print(f"wisescore: {wisescore:.4f}")
#             final_score = wisescore - 1.0
#             print(f"final_score: {final_score:.4f}")
#             return final_score

#         except Exception as e:
#             pid = prompt_data.get('prompt_id', 'N/A')
#             print(f"[ERR] {pid}: {e}")
#             return None

#     def judge_answer(self, image: Image.Image, data: Dict[str, Any]) -> int:
#         """
#         二分类：返回 0（正确）或 -1（错误）。
#         规则：final_score >= self.threshold → 0，否则 -1。
#         出错或无法得到分数时，返回 -1。
#         """
#         # final_score = self.get_reward(image, data)
#         final_score = -0.4  # 模拟分数，实际使用时应调用 get_reward 方法
#         if final_score is None:
#             return -1
#         return 0 if final_score >= self.threshold else -1

from PIL import Image
import io
import base64
import re
from typing import Dict, Any, Optional
from openai import AzureOpenAI

class GPT4oReward:
    """
    使用 OpenAI(Azure) 对单张图进行二分类评估：
    基于 Consistency / Realism / Aesthetic Quality 的严格标准进行总体判断，
    但只输出一个最终结果：0（正确）或 -1（错误）。
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        api_base: Optional[str] = None,
        region: str = "eastus2",
    ):
        API_BASE = api_base or "http://123.127.249.51/proxy"
        ENDPOINT = f"{API_BASE}/{region}"

        self.client = AzureOpenAI(
            api_key=api_key,
            api_version="2025-03-01-preview",
            azure_endpoint=ENDPOINT,
        )
        self.model = model

    def _encode_image(self, image: Image.Image) -> str:
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            png_bytes = buffer.getvalue()
        return base64.b64encode(png_bytes).decode()

    def _build_messages(
        self,
        prompt_data: Dict[str, Any],
        img64: str
    ) -> list:
        """
        让模型只输出 0 或 -1（二分类），
        但评判标准仍基于 Consistency / Realism / Aesthetic Quality。
        """
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a rigorous image–text alignment judge. "
                            "Decide ONLY whether the image correctly satisfies the prompt."
                        )
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Evaluate the image against the PROMPT strictly from three angles:
1) Consistency (faithfulness to every required element in the prompt; no contradictions or omissions),
2) Realism (physical plausibility and visual coherence; avoid obvious artifacts),
3) Aesthetic Quality (composition, color harmony, clarity).

Decision rule (EXTREMELY STRICT):
- Output **0** if AND ONLY IF the image clearly and fully satisfies the prompt with strong realism and acceptable aesthetics. Minor doubts → downgrade to -1.
- Otherwise, output **-1**.

IMPORTANT:
- Your entire response MUST be a single token: `0` or `-1`.
- Do NOT add any words, punctuation, spaces, or explanations.

PROMPT: "{prompt_data['prompt']}"
"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img64}"
                        }
                    }
                ]
            }
        ]

    def _extract_label(self, text: str) -> Optional[int]:
        """
        只从模型返回文本中提取首个 0 或 -1。
        容错：允许出现杂质文本时，仍能抓到第一个合法标签。
        """
        m = re.search(r'(?<!\d)-(?:1)\b|\b0\b', text.strip())
        if m:
            token = m.group(0)
            if token == '0':
                return 0
            if token == '-1':
                return -1
        m2 = re.search(r'-1|0', text)
        if m2:
            return int(m2.group(0))
        return None

    def get_reward(
        self,
        image: Image.Image,
        prompt_data: Dict[str, Any]
    ) -> Optional[int]:
        """
        返回二分类结果：
        - 0  ：正确
        - -1 ：错误
        - None：出错或解析失败
        """
        try:
            img64 = self._encode_image(image)
            messages = self._build_messages(prompt_data, img64)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,   # 确定性输出
                max_tokens=4       # 只需 0 或 -1
            )
            content = response.choices[0].message.content or ""
            print(f"[LLM raw] {content!r}")

            label = self._extract_label(content)
            if label in (0, -1):
                return label
            else:
                print("[WARN] Could not parse a valid label (0 or -1).")
                return None

        except Exception as e:
            print(f"[ERR]: {e}")
            return None

    def judge_answer(self, image: Image.Image, data: Dict[str, Any]) -> int:
        """
        直接返回 0 或 -1；若出错或未解析到标签，则保守返回 -1。
        """
        label = self.get_reward(image, data)
        return label if label in (0, -1) else -1


# if __name__ == '__main__':
#     # 1. 加载图像
#     img = Image.open("/media/raid/workspace/miyapeng/Multimodal-LatentSeek/src/geneval_results/long_results/Janus-Pro-7B-geneval-geneval-both-text_k0.2-image_k0.02-steps30-lr0.03-reward_threshold-0.1/final_img/00503/samples/0000.png").convert("RGB")

#     # 2. 构造 prompt 数据字典
#     solution = {"tag": "color_attr", "include": [{"class": "cow", "count": 1, "color": "blue"}, {"class": "computer keyboard", "count": 1, "color": "black"}], "prompt": "a photo of a blue cow and a black computer keyboard"}

#     # 3. 初始化 GPT4oReward 实例
#     reward_model = GPT4oReward(
#         api_key="64cd78bc94b8b7d6f02ee4263c3ed709",  # 替换成你自己的 API key
#         model="gpt-4o-2024-11-20"       # 模型名
#     )

#     # 4. 执行评估
#     # result = reward_model.judge_answer(img, solution)
#     result = reward_model.get_reward(img, solution)

#     # 5. 输出结果
#     print(f"Evaluation result: {result}")  # 输出 0 或 -1
