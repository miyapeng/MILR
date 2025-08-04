# reward2.py
import warnings
warnings.filterwarnings("ignore")

import os
import json
import tempfile
import shutil
from typing import Union, Dict, Any
from argparse import Namespace

import numpy as np
from PIL import Image

from rewards.MixedReward.unified_reward import UnifiedReward
from rewards.MixedReward.reward_gdino    import GDino
from rewards.MixedReward.score_git       import ColorEvaluator
from rewards.MixedReward.reward_hps      import HPSv2
# from unified_reward import UnifiedReward
# from reward_gdino    import GDino
# from score_git       import ColorEvaluator
# from reward_hps      import HPSv2


class MixedReward:
    """
    Four-way fusion for single-image evaluation:
      • ColorEvaluator (GIT)      weight 0.35
      • GroundDINO (GDino)        weight 0.35
      • UnifiedReward (1..5→0..1) weight 0.20
      • HPSv2 (cosine)            weight 0.10
    """
    def __init__(
        self,
        git_ckpt_path: str,
        unified_model_path: str,
        gdino_ckpt_path: str,
        gdino_config_path: str,
        hps_ckpt_path: str,
        device: str = "cuda:0"
    ):
        # 1) GIT evaluator
        self.git_evaluator  = ColorEvaluator(git_ckpt_path)

        # 2) UnifiedReward
        self.unified_reward = UnifiedReward(unified_model_path, device=device)

        # 3) GroundDINO
        gdino_args = Namespace(
            gdino_ckpt_path   = gdino_ckpt_path,
            gdino_config_path = gdino_config_path
        )
        self.gdino = GDino(gdino_args)
        self.gdino.load_to_device(device)

        # 4) HPSv2
        hps_args = Namespace(hps_ckpt_path=hps_ckpt_path)
        self.hps = HPSv2(hps_args)
        self.hps.load_to_device(device)


    def get_reward(
        self,
        image: Union[str, Image.Image],
        solution: Union[str, Dict[str, Any]]
    ) -> float:
        # --- 1) normalize image to filesystem path ---
        temp_img = None
        if isinstance(image, Image.Image):
            fd, temp_img = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            image.save(temp_img)
            image_path = temp_img
        else:
            image_path = image

        # --- 2) parse solution metadata ---
        metadata = json.loads(solution) if isinstance(solution, str) else solution

        try:
            # --- GIT color score (0..1) ---
            pairs     = self.extract_color_objects(metadata)
            git_res   = self.git_evaluator.evaluate_color(image_path, pairs)
            git_score = git_res.get("avg_score", 0.0)
            print(f"git_score = {git_score:.4f}")

            # --- UnifiedReward raw score (1..5) → normalize to 0..1 ---
            raw_uni   = self.unified_reward.get_reward(image, metadata)
            uni_score = np.clip((raw_uni - 1.0) / 4.0, 0.0, 1.0)

            # --- GroundDINO score (0..1) ---
            gdino_score = self._evaluate_gdino(image, metadata)

            # --- HPSv2 cosine sim (−1..1 → [0..1]) ---
            img_for_hps = Image.open(image_path).convert("RGB")
            hps_cosine  = self.hps(
                [metadata["prompt"]],
                [img_for_hps]
            )[0]
            hps_score   = np.clip((hps_cosine + 1.0) / 2.0, 0.0, 1.0)
            print(f"hps_score = {hps_score}")

            # --- final weighted sum ---
            final = (
                0.35 * git_score   +
                0.35 * gdino_score +
                0.20 * uni_score   +
                0.10 * hps_score
            )
            return float(final-1.0)

        finally:
            if temp_img and os.path.exists(temp_img):
                os.remove(temp_img)


    def extract_color_objects(self, metadata: Dict[str, Any]):
        """
        从 metadata['include'] 构造 (color, class, count) 列表
        """
        return [
            (item.get("color", ""),
             item.get("class", ""),
             item.get("count", 1))
            for item in metadata.get("include", [])
        ]


    def _evaluate_gdino(self, image, metadata: Dict[str, Any]) -> float:
        """
        GroundDINO 单图评测：position>count>object
        """
        # normalize image
        temp = None
        if isinstance(image, Image.Image):
            fd, temp = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            image.save(temp)
            image_path = temp
        else:
            image_path = image

        include = metadata.get("include", [])
        nouns   = [it["class"] for it in include]
        text_prompt, token_spans = self.gdino.make_prompt(nouns)

        # decide task
        if any("position" in it for it in include):
            pos    = next(it for it in include if "position" in it)
            others = [o for o in include if o is not pos]
            obj1   = pos["class"]
            obj2   = others[0]["class"] if others else nouns[0]
            locality = pos["position"][0]
            task_type     = ["spatial"]
            spatial_info  = [{"obj1": obj1, "obj2": obj2, "locality": locality}]
            numeracy_info = None

        elif any("count" in it for it in include):
            task_type     = ["numeracy"]
            all_counts    = [
                {"obj_name": it["class"], "num": it["count"]}
                for it in include if "count" in it
            ]
            numeracy_info = [all_counts]
            spatial_info  = None

        else:
            task_type     = ["object"]
            numeracy_info = None
            spatial_info  = None

        imgs_batch    = [Image.open(image_path).convert("RGB")]
        prompts_batch = [text_prompt]
        det_prompts   = [{"text_prompt": text_prompt, "token_spans": token_spans}]
        nouns_list    = [nouns]

        scores = self.gdino(
            prompts       = prompts_batch,
            images        = imgs_batch,
            task_type     = task_type,
            nouns         = nouns_list,
            det_prompt    = det_prompts,
            numeracy_info = numeracy_info,
            spatial_info  = spatial_info
        )

        if temp and os.path.exists(temp):
            os.remove(temp)
        return float(scores[0])

    def judge_answer(self, image, data):
        reward_score = self.get_reward(image, data)
        if reward_score < -0.1:
            return False
        else:
            return True

# if __name__ == "__main__":
#     git_ckpt   = "./reward_weights/git-large-vqav2"
#     uni_model  = "CodeGoat24/UnifiedReward-qwen-7b"
#     gdino_ckpt = "./reward_weights/groundingdino_swint_ogc.pth"
#     gdino_cfg  = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
#     hps_ckpt   = "./reward_weights/HPS_v2_compressed.pt"
#     device     = "cuda:0"

#     mr = MixedReward(
#         git_ckpt_path      = git_ckpt,
#         unified_model_path = uni_model,
#         gdino_ckpt_path    = gdino_ckpt,
#         gdino_config_path  = gdino_cfg,
#         hps_ckpt_path      = hps_ckpt,
#         device             = device
#     )

#     img = Image.open("/media/raid/workspace/miyapeng/Multimodal-LatentSeek/src/geneval_results/both_results/Janus-Pro-7B-geneval-both-text_k0.1-image_k0.01-steps10-lr0.01/final_img/00200/samples/0000.png").convert("RGB")
#     solution = {"tag": "counting", "include": [{"class": "bus", "count": 3}], "exclude": [{"class": "bus", "count": 4}], "prompt": "a photo of three buses"}

#     score = mr.get_reward(img, solution)
#     print(f"MixedReward → final score = {score}")
