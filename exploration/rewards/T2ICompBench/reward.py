import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import tempfile
from typing import Union, Dict, Any
from PIL import Image
from pathlib import Path

from .BLIPvqa_eval.reward import BlipVQAEvaluator
from .UniDet_eval.reward import SpatialRelationEvaluator
from .CLIPScore_eval.reward import ClipSimilarityEvaluator

class CompBenchRewardModel:
    """
    T2I-CompBench reward integration.
    """

    def __init__(
        self,
        task_type: str,
        device: str = "cuda:0",
        complex: bool = False,
    ):
        """
        Args:
            task_type: one of ['shape', 'color', 'texture', 'spatial', 'non_spatial', 'complex']
            device: device string, e.g. "cuda:0" or "cpu"
            complex: whether to use complex parsing for spatial / clipscore prompts
        """
        self.task_type = task_type.lower()
        self.device = device
        self.complex = complex

        if self.task_type in ["shape", "color", "texture"]:
            self.evaluator = BlipVQAEvaluator(device=self.device)
        elif self.task_type in ["spatial"]:
            self.evaluator = SpatialRelationEvaluator(device=self.device, complex=self.complex)
        elif self.task_type in ["non_spatial"]:
            self.evaluator = ClipSimilarityEvaluator(device=self.device, complex=self.complex)
        elif self.task_type == "complex":
            self.blip = BlipVQAEvaluator(device=self.device)
            self.spatial = SpatialRelationEvaluator(device=self.device, complex=self.complex)
            self.clip = ClipSimilarityEvaluator(device=self.device, complex=self.complex)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    def get_reward(
        self,
        image_input: Union[str, Image.Image],
        solution: Union[str, Dict[str, Any]]
    ) -> float:
        if isinstance(solution, str):
            metadata = json.loads(solution)
        else:
            metadata = solution

        if self.task_type == "complex":
            reward = self._eval_3_in_1(image_input, metadata) - 1.0
            return reward
        else:
            reward = self.evaluator.eval(image_input, metadata) - 1.0
            return reward

    def _eval_3_in_1(self, image_input: Union[str, Image.Image], metadata: Dict[str, Any]) -> float:
        # Call all 3 evaluators
        blip_score = self.blip.eval(image_input, metadata)
        spatial_score = self.spatial.eval(image_input, metadata)
        clip_score = self.clip.eval(image_input, metadata)
        
        current_dir = Path(__file__).resolve().parent  # 即 reward.py 所在目录
        target_file_1 = current_dir / "dataset" / "complex_val_spatial.txt"
        target_file_2 = current_dir / "dataset" / "complex_val_action.txt"

        with open(target_file_1, 'r') as f:
            spatial=f.readlines()
            spatial=[i.strip('\n').split('.')[0].lower() for i in spatial]
        with open(target_file_2, 'r') as f:
            action=f.readlines()
            action=[i.strip('\n').split('.')[0].lower() for i in action]

        # Combine
        if metadata['prompt'] in spatial:
            final_score = (blip_score + spatial_score)*0.5
        elif metadata['prompt'] in action:
            final_score = (blip_score + clip_score)*0.5
        else:
            final_score = (blip_score + spatial_score + clip_score) / 3

        return round(final_score, 4)
    
    def judge_answer(self, image, data) -> bool:
        reward_score = self.get_reward(image, data)
        if self.task_type in ["shape", "color", "texture"]:
            if reward_score < -0.2:
                return False
            else:
                return True
        elif self.task_type == "spatial":
            if reward_score < -0.1:
                return False
            else:
                return True
        elif self.task_type == "non_spatial":
            if reward_score < -0.5:
                return False
            else:
                return True
        elif self.task_type == "complex":
            if reward_score < -0.1:
                return False
            else:
                return True
        else:
            raise ValueError(f"Unknown task type for judge_answer: {self.task_type}")

