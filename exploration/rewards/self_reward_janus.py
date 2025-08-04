import json
from PIL import Image
from janus.utils.io import load_pil_images

class SelfRewardModel:
    def __init__(self, vl_gpt, vl_chat_processor, device="cuda"):
        self.vl_gpt = vl_gpt
        self.vl_chat_processor = vl_chat_processor
        self.device = device
        self.tokenizer = vl_chat_processor.tokenizer

    def _build_check_prompt(self, init_prompt: str) -> str:
        return (
            f"I've generated this image based on the initial caption: \"{init_prompt}\".\n"
            "First, list all objects you see in the image along with their color, location and number."
            "Then compare them one-by-one with the caption requirements:\n"
            "- For each object in the caption, check its color matches and count matches.\n\n"
            "For the whole image, check if the spatial relationships match the caption.\n\n"
            "Please evaluate whether the image accurately reflects the caption in terms of:\n"
            "- object presence and count\n"
            "- spatial relationships (position)\n"
            "- colors and other fine-grained details\n\n"
            "If right, the score is 0, otherwise -1.\n"
            "Your response must follow this strict JSON format:\n"
            "{\n"
            '  "Thought": "<your detailed reasoning about the image vs. caption>",\n'
            '  "Score": 0 or -1,\n'
            '  "Reason": "<brief reason only if Score is -1; leave empty string if Score is 0>"\n'
            "}\n\n"
            "Examples:\n"
            '{\n'
            '  "Thought": "The image contains one dog as described, and all details match the caption.",\n'
            '  "Score": 0,\n'
            '  "Reason": ""\n'
            '}\n'
            '{\n'
            '  "Thought": "The image contains three dogs, which is more than the one dog described in the caption.",\n'
            '  "Score": -1,\n'
            '  "Reason": "The image contains three dogs instead of one dog."\n'
            '}'
        )
        # return (
        #     f"I've generated this image based on the caption: \"{init_prompt}\".\n\n"
        #     "First, list all objects you see in the image along with their color and quantity. "
        #     "Then compare them one-by-one with the caption requirements:\n"
        #     "- For each object in the caption, check its color matches and count matches.\n\n"
        #     "If all match exactly, respond with JSON:\n"
        #     "{\n"
        #     '  "Thought": "<your step-by-step reasoning>",\n'
        #     '  "Score": 0,\n'
        #     '  "Reason": ""\n'
        #     "}\n\n"
        #     "If there's any mismatch (color, quantity, missing objects), respond with:\n"
        #     "{\n"
        #     '  "Thought": "<your step-by-step reasoning>",\n'
        #     '  "Score": -1,\n'
        #     '  "Reason": "<short reason for mismatch>"\n'
        #     "}"
        # )


    def _build_messages(self, prompt: str, image: Image.Image):
        # 统一把图文消息组织成一个列表
        return [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{prompt}",
                "images": [image],
            },
            {
                "role": "<|Assistant|>",
                "content": ""
            }
        ]

    def get_reward(self, image, solution):
        """Use the model's own reasoning to decide reward.
        Returns 0 if Score==0, -1 if Score==-1.
        """
        try:
            # —— 解析 solution JSON ——
            if isinstance(solution, str):
                metadata = json.loads(solution)
            elif isinstance(solution, dict):
                metadata = solution
            else:
                raise ValueError(f"Unsupported solution format: {type(solution)}")

            question = metadata.get("prompt") or metadata.get("question")
            if not question:
                raise ValueError("Missing 'prompt' or 'question' in solution")

            # —— 构造 prompt & messages ——
            prompt = self._build_check_prompt(question)
            messages = self._build_messages(prompt, image)

            # —— 调用模型生成答案 ——
            pil_images = load_pil_images(messages)
            prepare_inputs = self.vl_chat_processor(
                conversations=messages,
                images=pil_images,
                force_batchify=True
            ).to(self.device)

            inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
            outputs = self.vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=128,
                do_sample=False,
                use_cache=True,
            )
            raw_answer = self.tokenizer.decode(
                outputs[0].cpu().tolist(),
                skip_special_tokens=True
            ).strip()
            
            # —— 解析 JSON 输出 ——
            try:
                result = json.loads(raw_answer)
            except json.JSONDecodeError as e:
                print(f"[Warn] Output is not valid JSON:\n{raw_answer}")
                return -1

            # —— 根据 Score 判断 reward ——
            score = result.get("Score", -1)
            reason = result.get("Reason", "")

            if score == 0:
                return 0
            elif score == -1:
                print(f"[Eval failed]: {reason}")
                return -1
            else:
                print(f"[Warn] Unknown Score value: {score}, defaulting to -1")
                return -1

        except Exception as e:
            print(f"[Error] SelfReward failed: {e}")
            return -1


    def judge_answer(self, image, solution) -> bool:
        reward_score = self.get_reward(image, solution)
        if reward_score == -1:
            return False
        elif reward_score == 0:
            return True
        
