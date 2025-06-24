import os
import re
import json
import torch
import warnings
import numpy as np
from PIL import Image, ImageOps
from termcolor import colored
import tempfile

import mmdet
from mmdet.apis import inference_detector, init_detector
import open_clip
from clip_benchmark.metrics import zeroshot_classification as zsc
zsc.tqdm = lambda it, *args, **kwargs: it

warnings.filterwarnings("ignore")

class RewardModel(object):
    def __init__(
        self,
        model_config=None,
        model_path="./",
        object_names_path="object_names.txt",
        options=None,
        device="cuda"
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        assert self.device == "cuda", "This evaluation requires CUDA."

        self.options = options or {}
        self.threshold = float(self.options.get('threshold', 0.3))
        self.counting_threshold = float(self.options.get('counting_threshold', 0.9))
        self.max_objects = int(self.options.get('max_objects', 16))
        self.nms_threshold = float(self.options.get('max_overlap', 1.0))
        self.position_threshold = float(self.options.get('position_threshold', 0.1))

        # Load object detector
        if model_config is None:
            model_config = os.path.join(
                os.path.dirname(mmdet.__file__),
                "../configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
            )
        detector_ckpt = os.path.join(model_path, self.options.get('model', "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco") + ".pth")
        self.detector = init_detector(model_config, detector_ckpt, device=self.device)

        # Load CLIP
        clip_arch = self.options.get('clip_model', "ViT-L-14")
        self.clip_model, _, self.clip_transform = open_clip.create_model_and_transforms(
            clip_arch, pretrained="openai", device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(clip_arch)

        # Load object names
        with open(object_names_path) as f:
            self.classnames = [line.strip() for line in f]

        self.COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
        self.COLOR_CLASSIFIERS = {}

    class ImageCrops(torch.utils.data.Dataset):
        def __init__(self, image: Image.Image, objects, transform, bgcolor="#999"):
            self.image = image.convert("RGB")
            self.transform = transform
            self.blank = self.image.copy() if bgcolor == "original" else Image.new("RGB", image.size, color=bgcolor)
            self.objects = objects

        def __len__(self):
            return len(self.objects)

        def __getitem__(self, idx):
            box, mask = self.objects[idx]
            image = Image.composite(self.image, self.blank, Image.fromarray(mask)) if mask is not None else self.image
            image = image.crop(box[:4])
            return (self.transform(image), 0)

    def color_classification(self, image, bboxes, classname):
        if classname not in self.COLOR_CLASSIFIERS:
            self.COLOR_CLASSIFIERS[classname] = zsc.zero_shot_classifier(
                self.clip_model, self.tokenizer, self.COLORS,
                [
                    f"a photo of a {{c}} {classname}",
                    f"a photo of a {{c}}-colored {classname}",
                    f"a photo of a {{c}} object"
                ],
                self.device
            )
        clf = self.COLOR_CLASSIFIERS[classname]
        dataloader = torch.utils.data.DataLoader(
            self.ImageCrops(image, bboxes, self.clip_transform),
            batch_size=16, num_workers=4
        )
        with torch.no_grad():
            pred, _ = zsc.run_classification(self.clip_model, clf, dataloader, self.device)
            return [self.COLORS[index.item()] for index in pred.argmax(1)]

    def compute_iou(self, box_a, box_b):
        def area(box): return max(box[2]-box[0]+1, 0) * max(box[3]-box[1]+1, 0)
        i_area = area([
            max(box_a[0], box_b[0]), max(box_a[1], box_b[1]),
            min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
        ])
        u_area = area(box_a) + area(box_b) - i_area
        return i_area / u_area if u_area else 0

    def relative_position(self, obj_a, obj_b):
        boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
        center_a, center_b = boxes.mean(axis=-2)
        dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
        offset = center_a - center_b
        revised_offset = np.maximum(np.abs(offset) - self.position_threshold * (dim_a + dim_b), 0) * np.sign(offset)
        if np.all(np.abs(revised_offset) < 1e-3):
            return set()
        dx, dy = revised_offset / np.linalg.norm(offset)
        relations = set()
        if dx < -0.5: relations.add("left of")
        if dx > 0.5: relations.add("right of")
        if dy < -0.5: relations.add("above")
        if dy > 0.5: relations.add("below")
        return relations

    def evaluate(self, image, objects, metadata):
        correct, reason, matched_groups = True, [], []
        for req in metadata.get('include', []):
            classname = req['class']
            found_objects = objects.get(classname, [])[:req['count']]
            matched = len(found_objects) >= req['count']
            if not matched:
                reason.append(f"expected {classname}>={req['count']}, found {len(found_objects)}")
                correct = False
            else:
                if 'color' in req:
                    colors = self.color_classification(image, found_objects, classname)
                    if colors.count(req['color']) < req['count']:
                        correct = matched = False
                        reason.append(f"expected {req['color']} {classname}, found {colors}")
                if 'position' in req and matched:
                    expected_rel, target_idx = req['position']
                    if matched_groups[target_idx] is None:
                        reason.append(f"no target for {classname} to be {expected_rel}")
                        correct = matched = False
                    else:
                        for obj in found_objects:
                            for tgt in matched_groups[target_idx]:
                                rels = self.relative_position(obj, tgt)
                                if expected_rel not in rels:
                                    reason.append(f"{classname} not {expected_rel} target, got {rels}")
                                    correct = matched = False
                                    break
                            if not matched: break
            matched_groups.append(found_objects if matched else None)

        for req in metadata.get('exclude', []):
            classname = req['class']
            if len(objects.get(classname, [])) >= req['count']:
                correct = False
                reason.append(f"expected {classname}<{req['count']}, found {len(objects[classname])}")
        return correct, reason

    def evaluate_image(self, image_path, metadata):
        result = inference_detector(self.detector, image_path)
        bbox = result[0] if isinstance(result, tuple) else result
        segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        image = ImageOps.exif_transpose(Image.open(image_path))
        detected = {}
        conf_thresh = self.counting_threshold if metadata['tag'] == "counting" else self.threshold

        for i, classname in enumerate(self.classnames):
            boxes = bbox[i]
            ordering = np.argsort(boxes[:, 4])[::-1]
            ordering = ordering[boxes[ordering, 4] > conf_thresh][:self.max_objects].tolist()
            detected[classname] = []
            while ordering:
                idx = ordering.pop(0)
                detected[classname].append((boxes[idx], None if segm is None else segm[i][idx]))
                ordering = [
                    j for j in ordering
                    if self.nms_threshold == 1 or self.compute_iou(boxes[idx], boxes[j]) < self.nms_threshold
                ]
            if not detected[classname]:
                del detected[classname]
        is_correct, reason = self.evaluate(image, detected, metadata)
        return 0 if is_correct else 1, reason

    def get_reward(self, image, solution):
        """
        Args:
            image: str, path to image
            solution: str, json string containing metadata
        Returns:
            int: reward (1 or 0)
        """
        tmp_path = None
        try:
            if isinstance(solution, dict):
                metadata = json.loads(json.dumps(solution))  # 安全转换：dict → str → dict（保持结构）
            elif isinstance(solution, str):
                metadata = json.loads(solution)              # 原始逻辑

            tmp_path = None
            if isinstance(image, Image.Image):
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    image.save(tmp.name)
                    tmp_path = tmp.name
                    image_path = tmp_path
            elif isinstance(image, str):
                image_path = image
            else:
                raise ValueError("Unsupported image format.")

            reward, reason = self.evaluate_image(image_path, metadata)
            print(colored(f"[Eval] Image: {os.path.basename(image_path)} => Reward: {reward}, Reason: {reason}", "red" if reward else "green"))
            return reward

        except Exception as e:
            print(colored(f"[Error] Evaluation failed: {e}", "red"))
            return 0

        finally:
            # 清理临时图片
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
