from PIL import Image
import io
import base64
import re
import openai
from typing import Dict, Any
from openai import AzureOpenAI

class WiseReward:
    """
    Image-text evaluation class based on OpenAI ChatCompletion API.
    Evaluates a single example (prompt_data + PIL.Image) and returns a dictionary of three scores.
    """
    def __init__(
        self,
        api_key: str,
        model: str,
        api_base: str = None
    ):
        """
        Initialize the evaluator.

        Args:
            api_key: OpenAI API key.
            model: Name of the model to use.
            api_base: Optional base URL for custom OpenAI deployments.
        """
        # REGION = "eastus"
        # REGION = "eastus2"
        REGION = "northcentralus"
        MODEL = model
        API_KEY = api_key
        # API_BASE = "http://123.127.249.51/proxy"
        API_BASE = "https://api.tonggpt.mybigai.ac.cn/proxy"
        ENDPOINT = f"{API_BASE}/{REGION}"

        self.client = AzureOpenAI(
            api_key=API_KEY,
            api_version="2025-03-01-preview",
            azure_endpoint=ENDPOINT,
        )
        # openai.api_key = api_key
        # if api_base:
        #     openai.api_base = api_base

        self.model = MODEL

    def _encode_image(self, image: Image.Image) -> str:
        """
        Encode a PIL.Image.Image to a base64 PNG string.

        Args:
            image: PIL Image to encode.

        Returns:
            Base64-encoded PNG string.
        """
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            png_bytes = buffer.getvalue()
        # At this point buffer is closed and the only copy lives in png_bytes
        return base64.b64encode(png_bytes).decode()

    def _build_messages(
        self,
        prompt_data: Dict[str, Any],
        img64: str
    ) -> list:
        return [

        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a professional Vincennes image quality audit expert, please evaluate the image quality strictly according to the protocol."
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""Please evaluate strictly and return ONLY the three scores as requested.

    # Text-to-Image Quality Evaluation Protocol

    ## System Instruction
    You are an AI quality auditor for text-to-image generation. Apply these rules with ABSOLUTE RUTHLESSNESS. Only images meeting the HIGHEST standards should receive top scores.

    **Input Parameters**  
    - PROMPT: [User's original prompt to]  
    - EXPLANATION: [Further explanation of the original prompt] 
    ---

    ## Scoring Criteria

    **Consistency (0-2):**  How accurately and completely the image reflects the PROMPT.
    * **0 (Rejected):**  Fails to capture key elements of the prompt, or contradicts the prompt.
    * **1 (Conditional):** Partially captures the prompt. Some elements are present, but not all, or not accurately.  Noticeable deviations from the prompt's intent.
    * **2 (Exemplary):**  Perfectly and completely aligns with the PROMPT.  Every single element and nuance of the prompt is flawlessly represented in the image. The image is an ideal, unambiguous visual realization of the given prompt.

    **Realism (0-2):**  How realistically the image is rendered.
    * **0 (Rejected):**  Physically implausible and clearly artificial. Breaks fundamental laws of physics or visual realism.
    * **1 (Conditional):** Contains minor inconsistencies or unrealistic elements.  While somewhat believable, noticeable flaws detract from realism.
    * **2 (Exemplary):**  Achieves photorealistic quality, indistinguishable from a real photograph.  Flawless adherence to physical laws, accurate material representation, and coherent spatial relationships. No visual cues betraying AI generation.

    **Aesthetic Quality (0-2):**  The overall artistic appeal and visual quality of the image.
    * **0 (Rejected):**  Poor aesthetic composition, visually unappealing, and lacks artistic merit.
    * **1 (Conditional):**  Demonstrates basic visual appeal, acceptable composition, and color harmony, but lacks distinction or artistic flair.
    * **2 (Exemplary):**  Possesses exceptional aesthetic quality, comparable to a masterpiece.  Strikingly beautiful, with perfect composition, a harmonious color palette, and a captivating artistic style. Demonstrates a high degree of artistic vision and execution.

    ---

    ## Output Format

    **Do not include any other text, explanations, or labels.** You must return only three lines of text, each containing a metric and the corresponding score, for example:

    **Example Output:**
    Consistency: 2
    Realism: 1
    Aesthetic Quality: 0

    ---

    **IMPORTANT Enforcement:**

    Be EXTREMELY strict in your evaluation. A score of '2' should be exceedingly rare and reserved only for images that truly excel and meet the highest possible standards in each metric. If there is any doubt, downgrade the score.

    For **Consistency**, a score of '2' requires complete and flawless adherence to every aspect of the prompt, leaving no room for misinterpretation or omission.

    For **Realism**, a score of '2' means the image is virtually indistinguishable from a real photograph in terms of detail, lighting, physics, and material properties.

    For **Aesthetic Quality**, a score of '2' demands exceptional artistic merit, not just pleasant visuals.

    --- 
    Here are the Prompt and EXPLANATION for this evaluation:
    PROMPT: "{prompt_data['prompt']}"
    EXPLANATION: "{prompt_data['explanation']}"
    Please strictly adhere to the scoring criteria and follow the template format when providing your results."""
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

    def _extract_scores(self, text: str) -> Dict[str, float]:
        """
        Extract the three scores from the model's text response.

        Args:
            text: The raw text returned by the model.

        Returns:
            A dict with keys 'consistency', 'realism', 'aesthetic_quality'.
        """
        pat = r"\*{0,2}(Consistency|Realism|Aesthetic Quality)\*{0,2}\s*[::]?\s*(\d)"
        matches = re.findall(pat, text, re.IGNORECASE)
        out = {}
        for k, v in matches:
            out[k.lower().replace(" ", "_")] = float(v)
        return out
    
    def _calculate_wiscore(self, consistency, realism, aesthetic_quality):
        """Calculates the WiScore based on given components."""
        return (0.7 * consistency + 0.2 * realism + 0.1 * aesthetic_quality) / 2

    def get_reward(
        self,
        image: Image.Image,
        prompt_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Evaluate a single example and return the scores.

        Args:
            prompt_data: Dictionary containing 'prompt_id', 'prompt', 'explanation', etc.
            image: A PIL.Image.Image object.

        Returns:
            A dict with keys 'consistency', 'realism', 'aesthetic_quality'.
        """
        try:
            # 1) Encode image
            img64 = self._encode_image(image)
            # 2) Build messages and call OpenAI API
            messages = self._build_messages(prompt_data, img64)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=2000
            )
            content = response.choices[0].message.content
            print(f"gpt output\n{content}")
            # 3) Extract and return scores
            scores = self._extract_scores(content)
            
            consistency = scores.get('consistency', 0.0)
            realism = scores.get('realism', 0.0)
            aesthetic_quality = scores.get('aesthetic_quality', 0.0)
            
            wisescore = self._calculate_wiscore(consistency, realism, aesthetic_quality)
            print(f"wisescore:{wisescore}")
            final_score = wisescore - 1.0
            return final_score

        except Exception as e:
            print(f"[ERR] {prompt_data['prompt_id']}: {e}")
            return None

    def judge_answer(self, image, data):
        #reward_score = self.get_reward(image, data)
        reward_score = -0.4
        if reward_score < -0.5:
            return False
        else:
            return True

# # Example usage
# if __name__ == '__main__':
#     # Example prompt_data
#     example = {
#         'prompt_id': 1,
#         'tag': 'Cultural knowledge',
#         'subcategory': 'Festival',
#         'prompt': 'Traditional food of the Mid-Autumn Festival',
#         'explanation': 'This refers to mooncakes, the round pastries filled with lotus seed paste or red bean paste'
#     }
#     # Load test image
#     test_image = Image.open('/media/raid/workspace/miyapeng/T2I-R1/src/t2i-r1/src/generated_samples_test7/img_3.jpg')  # Replace with your test image path
#     # Initialize evaluator
#     wr = WiseReward(api_key='64cd78bc94b8b7d6f02ee4263c3ed709', model='gpt-4o-2024-05-13')
#     # Perform evaluation
#     result_scores = wr.get_reward(example, test_image)
#     print('Evaluation results:', result_scores)
