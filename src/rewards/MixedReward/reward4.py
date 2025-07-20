import warnings
# 忽略所有 warning
warnings.filterwarnings("ignore")

import os
import json
import tempfile
import shutil
from typing import Union, Dict, Any
from PIL import Image
import numpy as np
from argparse import Namespace

from rewards.MixedReward.reward_gdino    import GDino
from rewards.MixedReward.score_git       import ColorEvaluator


class MixedReward:
    """
    Three‐way fusion for single‐image & prompt:
      • GroundDINO (GDino)        weight 0.4
      • ColorEvaluator (GIT)      weight 0.4
    """

    def __init__(
        self,
        git_ckpt_path: str,
        gdino_ckpt_path: str,
        gdino_config_path: str,
        device: str = "cuda:0"
    ):
        # 1) GIT evaluator
        self.git_evaluator    = ColorEvaluator(git_ckpt_path)

        # 3) GDino
        gdino_args = Namespace(
            gdino_ckpt_path   = gdino_ckpt_path,
            gdino_config_path = gdino_config_path
        )
        self.gdino = GDino(gdino_args)
        self.gdino.load_to_device(device)


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
            # --- 3) GIT color score (0..1) ---
            pairs     = self.extract_color_objects(metadata)
            git_res   = self.git_evaluator.evaluate_color(image_path, pairs)
            git_score = git_res.get("avg_score", 0.0)
            print(f"git_score = {git_score}")

            # --- 5) GDino spatial/object score (0..1) ---
            gdino_score = self._evaluate_gdino(image, metadata)
            print(f"gdino_score = {gdino_score}")
            # --- 6) weighted sum ---
            if metadata.get("tag") in ("colors", "color_attr"):
                # numeracy task: use 0.5 weight for GIT
                final_score = (
                    0.5 * gdino_score +
                    0.5 * git_score
                )
            else:
                final_score = (
                    0.75 * gdino_score +
                    0.25 * git_score
                )
            return float(final_score-1.0)

        finally:
            if temp_img and os.path.exists(temp_img):
                os.remove(temp_img)


    def extract_color_objects(self, metadata: Dict[str, Any]):
        """
        From metadata['include'] build list of (color, class, count).
        """
        return [
            (item.get("color", ""),
             item.get("class", ""),
             item.get("count", 1))
            for item in metadata.get("include", [])
        ]


    def _evaluate_gdino(self, image: Union[str, Image.Image], metadata: Dict[str, Any]) -> float:
        """
        Wraps GroundDINO for one image:
         - spatial if any 'position'
         - numeracy if any 'count'
         - else object
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

        include      = metadata.get("include", [])
        nouns        = [itm["class"] for itm in include]
        text_prompt, token_spans = self.gdino.make_prompt(nouns)

        # pick task
        if any("position" in itm for itm in include):
            pos     = next(itm for itm in include if "position" in itm)
            others  = [o for o in include if o is not pos]
            obj1    = pos["class"]
            obj2    = others[0]["class"] if others else nouns[0]
            locality= pos["position"][0]
            task_type     = ["spatial"]
            spatial_info  = [{"obj1": obj1, "obj2": obj2, "locality": locality}]
            numeracy_info = None

        elif any("count" in itm for itm in include):
            task_type     = ["numeracy"]
            # collect all counts into a single list
            all_counts    = [
                {"obj_name": itm["class"], "num": itm["count"]}
                for itm in include if "count" in itm
            ]
            numeracy_info = [ all_counts ]
            spatial_info  = None

        else:
            task_type     = ["object"]
            numeracy_info = None
            spatial_info  = None

        # prepare batch-of-1
        imgs_batch    = [ Image.open(image_path).convert("RGB") ]
        prompts_batch = [ text_prompt ]
        det_prompts   = [{"text_prompt":text_prompt, "token_spans":token_spans}]
        nouns_list    = [ nouns ]

        scores = self.gdino(
            prompts       = prompts_batch,
            images        = imgs_batch,
            task_type     = task_type,
            nouns         = nouns_list,
            det_prompt    = det_prompts,
            numeracy_info = numeracy_info,
            spatial_info  = spatial_info
        )

        # cleanup
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
#     # setup
#     git_ckpt   = "./reward_weights/git-large-vqav2"
#     uni_model  = "CodeGoat24/UnifiedReward-qwen-7b"
#     gdino_ckpt = "./reward_weights/groundingdino_swint_ogc.pth"
#     gdino_cfg  = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
#     device     = "cuda:0"

#     mr = MixedReward(
#         git_ckpt_path      = git_ckpt,
#         unified_model_path = uni_model,
#         gdino_ckpt_path    = gdino_ckpt,
#         gdino_config_path  = gdino_cfg,
#         device             = device
#     )

#     img = Image.open("/media/raid/workspace/miyapeng/Multimodal-LatentSeek/src/geneval_results/both_results/Janus-Pro-7B-geneval-both-text_k0.1-image_k0.01-steps10-lr0.01/final_img/00457/samples/0000.png").convert("RGB")
#     solution = {"tag": "color_attr", "include": [{"class": "oven", "count": 1, "color": "pink"}, {"class": "motorcycle", "count": 1, "color": "green"}], "prompt": "a photo of a pink oven and a green motorcycle"}

#     score = mr.get_reward(img, solution)
#     print(f"Weighted MixedReward = {score}")
