import os
import json
import tempfile
from typing import Union, Dict, Any
from PIL import Image

from BLIPvqa_eval.reward import BlipVQAEvaluator
from UniDet_eval.reward import SpatialRelationEvaluator
from CLIPScore_eval.reward import ClipSimilarityEvaluator


class CompBenchRewardModel:
    """
    T2I-CompBench reward integration.
    Supports unified interface for shape/color/texture/spatial/clipscore/3-in-1 evaluation.
    """

    def __init__(
        self,
        task_type: str,
        device: str = "cuda:0",
        complex: bool = False,
        reduction: str = "product"  # or "mean"
    ):
        """
        Args:
            task_type: one of ['shape', 'color', 'texture', 'spatial', 'non-spatial', '3_in_1']
            device: device string, e.g. "cuda:0" or "cpu"
            complex: whether to use complex parsing for spatial / clipscore prompts
            reduction: for '3_in_1', how to merge scores. One of ['product', 'mean']
        """
        self.task_type = task_type.lower()
        self.device = device
        self.complex = complex
        self.reduction = reduction

        if self.task_type in ["shape", "color", "texture"]:
            self.evaluator = BlipVQAEvaluator(device=self.device)
        elif self.task_type in ["spatial"]:
            self.evaluator = SpatialRelationEvaluator(device=self.device, complex=self.complex)
        elif self.task_type in ["non-spatial"]:
            self.evaluator = ClipSimilarityEvaluator(device=self.device, complex=self.complex)
        elif self.task_type == "3_in_1":
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

        if self.task_type == "3_in_1":
            return self._eval_3_in_1(image_input, metadata)
        else:
            return self.evaluator.eval(image_input, metadata)

    def _eval_3_in_1(self, image_input: Union[str, Image.Image], metadata: Dict[str, Any]) -> float:
        # Call all 3 evaluators
        blip_score = self.blip.eval(image_input, metadata)
        spatial_score = self.spatial.eval(image_input, metadata)
        clip_score = self.clip.eval(image_input, metadata)

        # Combine
        if self.reduction == "mean":
            final_score = (blip_score + spatial_score + clip_score) / 3
        elif self.reduction == "product":
            final_score = blip_score * spatial_score * clip_score
        else:
            raise ValueError("Unsupported reduction method. Choose from ['product', 'mean'].")

        return round(final_score, 4)
