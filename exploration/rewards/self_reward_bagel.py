import json
from PIL import Image
from janus.utils.io import load_pil_images

class SelfRewardModel:
    def __init__(self, inferencer, device="cuda"):
        self.inferencer = inferencer
        self.device = device
        self.inference_hyper=dict(
            max_think_token_n=1000,
            do_sample=False,
            # text_temperature=0.3,
        )

    def _build_check_prompt(self, init_prompt: str) -> str:
        # return (
        #     f"I've generated this image based on the initial caption: \"{init_prompt}\".\n"
        #     "First, list all objects you see in the image along with their color, location and number."
        #     "Then compare them one-by-one with the caption requirements:\n"
        #     "- For each object in the caption, check its color matches and count matches.\n\n"
        #     "For the whole image, check if the spatial relationships match the caption.\n\n"
        #     "Please evaluate whether the image accurately reflects the caption in terms of:\n"
        #     "- object presence and count\n"
        #     "- spatial relationships (position)\n"
        #     "- colors and other fine-grained details\n\n"
        #     "If right, the score is 0, otherwise -1.\n"
        #     "Your response must follow this strict JSON format:\n"
        #     "{\n"
        #     '  "Thought": "<your detailed reasoning about the image vs. caption>",\n'
        #     '  "Score": 0 or -1,\n'
        #     '  "Reason": "<brief reason only if Score is -1; leave empty string if Score is 0>"\n'
        #     "}\n\n"
        #     "Examples:\n"
        #     '{\n'
        #     '  "Thought": "The image contains one dog as described, and all details match the caption.",\n'
        #     '  "Score": 0,\n'
        #     '  "Reason": ""\n'
        #     '}\n'
        #     '{\n'
        #     '  "Thought": "The image contains three dogs, which is more than the one dog described in the caption.",\n'
        #     '  "Score": -1,\n'
        #     '  "Reason": "The image contains three dogs instead of one dog."\n'
        #     '}'
        # )
        return (
            f"I've generated this image based on the caption: \"{init_prompt}\".\n\n"
            "First, list all objects you see in the image along with their color and quantity. "
            "Then compare them one-by-one with the caption requirements:\n"
            "- For each object in the caption, check its color matches and count matches.\n\n"
            "If all match exactly, respond with JSON:\n"
            "{\n"
            '  "Thought": "<your step-by-step reasoning>",\n'
            '  "Score": 0,\n'
            '  "Reason": ""\n'
            "}\n\n"
            "If there's any mismatch (color, quantity, missing objects), respond with:\n"
            "{\n"
            '  "Thought": "<your step-by-step reasoning>",\n'
            '  "Score": -1,\n'
            '  "Reason": "<short reason for mismatch>"\n'
            "}"
        )

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

            raw_answer = self.inferencer(image=image,text=prompt,think=True,understanding_output=True,**self.inference_hyper)['text'].strip()
            
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
        
