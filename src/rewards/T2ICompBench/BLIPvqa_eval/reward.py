import os
import json
import tempfile
import shutil
import spacy
import torch
from typing import Dict, Any, Union
from PIL import Image
from BLIP.train_vqa_func import VQA_main


class BlipVQAEvaluator:
    def __init__(self, np_num: int = 8, device: str = "cuda:0"):
        self.np_num = np_num
        self.device = device
        self._nlp = spacy.load("en_core_web_sm")

    def _create_annotation(self, image_path: str, prompt: str, tag: str, np_index: int, out_dir: str):
        ann = [{
            "image": image_path,
            "question_id": 0,
            "question": prompt + "?" if prompt else "",
            "dataset": "color",
        }]
        annotation_dir = os.path.join(out_dir, f"annotation{np_index + 1}_blip")
        vqa_dir = os.path.join(annotation_dir, "VQA")
        os.makedirs(vqa_dir, exist_ok=True)

        with open(os.path.join(annotation_dir, "vqa_test.json"), "w", encoding="utf-8") as f:
            json.dump(ann, f)

        return annotation_dir, vqa_dir

    def eval(self, image_input: Union[str, Image.Image], metadata: Dict[str, Any]) -> float:
        prompt = metadata.get("prompt", "")
        tag = metadata.get("tag", "")
        if not prompt:
            raise ValueError("metadata must contain 'prompt' field")

        # Handle PIL.Image.Image input
        temp_img_file = None
        if isinstance(image_input, Image.Image):
            temp_img_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            image_input.save(temp_img_file.name)
            image_path = temp_img_file.name
        elif isinstance(image_input, str):
            image_path = image_input
        else:
            raise TypeError("image_input must be a str path or PIL.Image.Image object")

        # 1. Extract noun phrases
        doc = self._nlp(prompt)
        exclude = {'top', 'the side', 'the left', 'the right'}
        noun_phrases = [chunk.text for chunk in doc.noun_chunks if chunk.text not in exclude]
        np_index = min(self.np_num, len(noun_phrases))

        reward = torch.zeros((1, np_index)).to(self.device)

        # 2. For each noun phrase, run VQA and record score
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(np_index):
                print(f"start VQA{i + 1}/{np_index}!")
                ann_dir, vqa_dir = self._create_annotation(
                    image_path=image_path,
                    prompt=noun_phrases[i],
                    tag=tag,
                    np_index=i,
                    out_dir=tmpdir,
                )

                with open(os.path.join(ann_dir, "vqa_test.json"), "r") as f:
                    print(json.load(f))

                blip_root = os.path.dirname(__file__)            # .../BLIPvqa_eval
                cwd_before = os.getcwd()
                os.chdir(blip_root)                              # 切到 .../BLIPvqa_eval
                try:
                    VQA_main(ann_dir + "/", vqa_dir + "/")
                finally:
                    os.chdir(cwd_before)                         # 恢复原来的 cwd

                with open(os.path.join(vqa_dir, "result", "vqa_result.json"), "r") as file:
                    r = json.load(file)
                with open(os.path.join(ann_dir, "vqa_test.json"), "r") as file:
                    r_tmp = json.load(file)

                reward[0][i] = float(r[0]["answer"]) if r_tmp[0]["question"] != "" else 1.0
                print(f"end VQA{i + 1}/{np_index}!")

        if temp_img_file:
            os.remove(temp_img_file.name)

        # 3. Compute reward product
        reward_final = reward[:, 0]
        for i in range(1, np_index):
            reward_final *= reward[:, i]

        return float(f"{reward_final.item():.4f}")


# if __name__ == "__main__":
#     # Instantiate the evaluator
#     evaluator = BlipVQAEvaluator(np_num=8, device="cuda:0")

#     # Example 1: pass image path
#     image_path = "/media/raid/workspace/miyapeng/T2I-CompBench/examples/samples/a horse on the right of a car_000003.png"
#     metadata = {'tag': 'color', 'prompt': 'a horse on the right of a car'}
    
#     # score = evaluator.eval(image_path, metadata)
#     # print("Final BLIP-VQA Score (from path):", score)

#     # Example 2: pass PIL.Image.Image
#     img = Image.open(image_path)
#     score = evaluator.eval(img, metadata)
#     print("Final BLIP-VQA Score (from PIL):", score)
