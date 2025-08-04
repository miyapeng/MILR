import json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch

class UnifiedReward:
    """
    Reward model based on UnifiedReward framework (CodeGoat24/UnifiedReward-qwen-7b).
    Evaluates an image against a text prompt and outputs a numerical score.
    """
    def __init__(self, model_path: str, device: str = "cuda:0"):
        """
        Args:
            model_path: Hugging Face path or local path for the UnifiedReward model.
            device: device string, e.g., 'cuda:0' or 'cpu'.
        """
        self.device = device
        # Load the multimodal generation model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map={"": device}
        ).eval().to(device)
        # Load corresponding processor
        self.processor = AutoProcessor.from_pretrained(model_path)

    def get_reward(self, image, solution):
        """
        Generates a score for the image based on the provided caption.
        Args:
            image: PIL.Image.Image or file path to the image.
            solution: dict or JSON string containing at least a 'prompt' field.
        Returns:
            int: Final Score extracted from the model's output (1-5). Returns None if not found.
        """
        # Ensure image is a PIL.Image
        if isinstance(image, str):
            image = Image.open(image)
        # Parse solution to extract prompt
        if isinstance(solution, str):
            metadata = json.loads(solution)
        elif isinstance(solution, dict):
            metadata = solution
        else:
            raise ValueError(f"Unsupported solution format: {type(solution)}")
        prompt = metadata.get("prompt") or metadata.get("question")
        if prompt is None:
            raise ValueError("Missing 'prompt' or 'question' in solution metadata.")

        # Construct the chat-like messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            "You are given a text caption and a generated image based on that caption. "
                            "Your task is to evaluate this image based on two key criteria:\n"
                            "1. Alignment with the Caption: Assess how well this image aligns with the provided caption. "
                            "Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n"
                            "2. Overall Image Quality: Examine the visual quality of this image, including clarity, "
                            "detail preservation, color accuracy, and overall aesthetic appeal.\n"
                            "Extract key elements from the provided text caption, evaluate their presence in the generated image "
                            "using the format: 'element (type): value' (where value=0 means not generated, and value=1 means generated), "
                            "and assign a score from 1 to 5 after 'Final Score:'.\n"
                            "Your task is provided as follows:\n"
                            f"Text Caption: [{prompt}]"
                        )
                    },
                ],
            }
        ]

        # Apply the chat template and prepare vision inputs
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        # Generate model output
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        # Trim input tokens and decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        print(output_text)

        # Parse final score from the output
        final_score = None
        MAX_SCORE = 5.0
        for line in output_text.splitlines():
            if 'Final Score:' in line:
                try:
                    raw = float(line.split('Final Score:')[1].strip())
                    # raw ∈ [1.0..MAX_SCORE] → mapped ∈ [-(MAX_SCORE-1)..0]
                    final_score = raw - MAX_SCORE
                    break   # 拿到第一个数值后就行
                except ValueError:
                    continue
        return final_score

    ### need modification
    def judge_answer(self, image, solution) -> bool:
        reward_score = self.get_reward(image, solution)
        if reward_score!=0:
            return False
        else:
            return True