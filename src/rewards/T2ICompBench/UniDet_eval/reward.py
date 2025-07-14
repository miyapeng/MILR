import torch
import tempfile
import os
import json
from PIL import Image
import spacy
from typing import Dict, Any
from experts.model_bank import load_expert_model
from experts.obj_detection.generate_dataset import Dataset, collate_fn
from torch.utils.data import DataLoader
from accelerate import Accelerator
import numpy as np

def determine_position(locality, box1, box2, iou_threshold=0.1,distance_threshold=150):
    # Calculate centers of bounding boxes
    box1_center = ((box1['x_min'] + box1['x_max']) / 2, (box1['y_min'] + box1['y_max']) / 2)
    box2_center = ((box2['x_min'] + box2['x_max']) / 2, (box2['y_min'] + box2['y_max']) / 2)

    # Calculate horizontal and vertical distances
    x_distance = box2_center[0] - box1_center[0]
    y_distance = box2_center[1] - box1_center[1]

    # Calculate IoU
    x_overlap = max(0, min(box1['x_max'], box2['x_max']) - max(box1['x_min'], box2['x_min']))
    y_overlap = max(0, min(box1['y_max'], box2['y_max']) - max(box1['y_min'], box2['y_min']))
    intersection = x_overlap * y_overlap
    box1_area = (box1['x_max'] - box1['x_min']) * (box1['y_max'] - box1['y_min'])
    box2_area = (box2['x_max'] - box2['x_min']) * (box2['y_max'] - box2['y_min'])
    union = box1_area + box2_area - intersection
    iou = intersection / union

    # Determine position based on distances and IoU and give a soft score
    score=0
    if locality in ['next to', 'on side of', 'near']:
        if (abs(x_distance)< distance_threshold or abs(y_distance)< distance_threshold):
            score=1
        else:
            score=distance_threshold/max(abs(x_distance),abs(y_distance))
    elif locality == 'on the right of':
        if x_distance < 0:
            if abs(x_distance) > abs(y_distance) and iou < iou_threshold:
                score=1
            elif abs(x_distance) > abs(y_distance) and iou >= iou_threshold:
                score=iou_threshold/iou
        else:
            score=0
    elif locality == 'on the left of':
        if x_distance > 0:
            if abs(x_distance) > abs(y_distance) and iou < iou_threshold:
                score=1
            elif abs(x_distance) > abs(y_distance) and iou >= iou_threshold:
                score=iou_threshold/iou
        else:
            score=0
    elif locality =='on the bottom of':
        if y_distance < 0:
            if abs(y_distance) > abs(x_distance) and iou < iou_threshold:
                score=1
            elif abs(y_distance) > abs(x_distance) and iou >= iou_threshold:
                score=iou_threshold/iou
    elif locality =='on the top of':
        if y_distance > 0:
            if abs(y_distance) > abs(x_distance) and iou < iou_threshold:
                score=1
            elif abs(y_distance) > abs(x_distance) and iou >= iou_threshold:
                score=iou_threshold/iou
    else:
        score=0
    return score



def get_mask_labels(depth, instance_boxes, instance_id):
    obj_masks = []
    obj_ids = []
    obj_boundingbox = []
    for i in range(len(instance_boxes)):
        is_duplicate = False
        mask = torch.zeros_like(depth)
        x1, y1, x2, y2 = instance_boxes[i][0].item(), instance_boxes[i][1].item(), \
                         instance_boxes[i][2].item(), instance_boxes[i][3].item()
        mask[int(y1):int(y2), int(x1):int(x2)] = 1
        if not is_duplicate:
            obj_masks.append(mask)
            obj_ids.append(instance_id[i])
            obj_boundingbox.append([x1, y1, x2, y2])

    instance_labels = {}
    for i in range(len(obj_ids)):
        instance_labels[i] = obj_ids[i].item()
    return obj_boundingbox, instance_labels


class SpatialRelationEvaluator:
    def __init__(self, device: str = "cuda:0", complex: bool = False):
        self.device = device
        self.complex = complex

         # 记录当前 cwd
        cwd_before = os.getcwd()

        # 切到这个模块所在的文件夹，也就是 ....../T2ICompBench/UniDet_eval
        eval_root = os.path.dirname(__file__)
        os.chdir(eval_root)

        try:
            # 现在 load_expert_model 内部 merge_from_file("experts/obj_detection/configs/...") 
            # 就会在 eval_root/experts/... 下找到正确的 .yaml
            self.model, self.transform = load_expert_model(task='obj_detection', ckpt="RS200")
            self.label_map = torch.load("dataset/detection_features.pt")["labels"]
        finally:
            # 恢复原来的 cwd
            os.chdir(cwd_before)

        self.accelerator = Accelerator(mixed_precision="fp16")
        self.model = self.accelerator.prepare(self.model)

        self.nlp       = spacy.load("en_core_web_sm")

    def eval(self, image: Image.Image, metadata: Dict[str, Any]) -> float:
        """
        对单张 PIL.Image 和对应 metadata 进行空间关系评测，
        返回一个 [0..1] 区间内的分数（四舍五入到小数点后 4 位）。
        metadata 需包含 'prompt' 和 'tag'。
        """
        prompt = metadata.get("prompt", "")
        tag    = metadata.get("tag", "")
        if not prompt:
            raise ValueError("metadata 必须包含 'prompt' 字段")

        
        # 1) 把 PIL.Image 存成临时文件
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create "samples" subfolder inside the temp directory
            sample_dir = os.path.join(tmpdir, "samples")
            os.makedirs(sample_dir, exist_ok=True)

            # Save image inside the "samples" folder
            img_path = os.path.join(sample_dir, "dummy_000000.png")
            image.save(img_path)

            # 2) 构造单样本 Dataset + DataLoader
            ds = Dataset(tmpdir, self.transform)
            dl = torch.utils.data.DataLoader(
                ds, batch_size=1, shuffle=False,
                collate_fn=collate_fn
            )
            dl = self.accelerator.prepare(dl)

            # 3) 只取第一（唯一）个 batch
            for batch in dl:
                with torch.no_grad():
                    test_pred = self.model(batch)[0]  # dict for this one image

                # 4) 拿到所有 bbox/class/score
                instance_boxes = test_pred['instances'].get_fields()['pred_boxes'].tensor
                instance_id = test_pred['instances'].get_fields()['pred_classes']
                instance_score = test_pred['instances'].get_fields()['scores']
                depth   = batch[0]["image"][0]

                obj_bounding_box, obj_labels_dict = get_mask_labels(depth, instance_boxes, instance_id)
                
                vocab_spatial = ['on side of', 'next to', 'near', 'on the left of', 'on the right of', 'on the bottom of', 'on the top of','on top of'] #locality words

                locality = None
                for word in vocab_spatial:
                    if word in prompt:
                        locality = word
                        break

                obj = []  
                for i in range(len(obj_bounding_box)):
                    obj_name = self.label_map[obj_labels_dict[i]]  
                    obj.append(obj_name)

                if (self.complex):
                    # Define the sentence
                    sentence = prompt
                    # Process the sentence using spaCy
                    doc = self.nlp(sentence)
                    # Define the target prepositions
                    prepositions = ["on top of", "on bottom of", "on the left", "on the right",'next to','on side of','near']
                    # Extract objects before and after the prepositions
                    objects = []
                    for i in range(len(doc)):
                        if doc[i:i + 3].text in prepositions or doc[i:i + 2].text in prepositions or doc[i:i + 1].text in prepositions:
                            if doc[i:i + 3].text in prepositions:
                                k=3
                            elif doc[i:i + 2].text in prepositions:
                                k=2
                            elif doc[i:i + 1].text in prepositions:
                                k=1
                            preposition_phrase = doc[i:i + 3].text
                            for j in range(i - 1, -1, -1):
                                if doc[j].pos_ == 'NOUN':
                                    objects.append(doc[j].text)
                                    break
                                elif doc[j].pos_ == 'PROPN':
                                    objects.append(doc[j].text)
                                    break
                            flag=False
                            for j in range(i + k, len(doc)):
                                if doc[j].pos_ == 'NOUN':
                                    objects.append(doc[j].text)
                                    break
                                if(j==len(doc)-1):
                                    flag=True 
                            if flag:
                                for j in range(i + k, len(doc)):
                                    if (j+1<len(doc)) and doc[j].pos_ == 'PROPN' and doc[j+1].pos_ != 'PROPN':
                                        objects.append(doc[j].text)
                                        break
                    if (len(objects)==2):
                        obj1=objects[0]
                        obj2=objects[1]
                    else:
                        obj1=None
                        obj2=None
                else:
                    #for simple structure
                    doc = self.nlp(prompt)
                    obj1= [token.text for token in doc if token.pos_=='NOUN'][0]
                    obj2= [token.text for token in doc if token.pos_=='NOUN'][-1]

                person = ['girl','boy','man','woman']
                if obj1 in person:
                    obj1 = "person"
                if obj2 in person:
                    obj2 = "person"
                if obj1 in obj and obj2 in obj:
                    obj1_pos = obj.index(obj1)
                    obj2_pos = obj.index(obj2)
                    obj1_bb = obj_bounding_box[obj1_pos]
                    obj2_bb = obj_bounding_box[obj2_pos]
                    box1, box2={},{}

                    box1["x_min"] = obj1_bb[0]
                    box1["y_min"] = obj1_bb[1]
                    box1["x_max"] = obj1_bb[2]
                    box1["y_max"] = obj1_bb[3]
                    box2["x_min"] = obj2_bb[0]
                    box2["y_min"] = obj2_bb[1]
                    box2["x_max"] = obj2_bb[2]
                    box2["y_max"] = obj2_bb[3]


                    score = 0.25 * instance_score[obj1_pos].item() + 0.25 * instance_score[obj2_pos].item()  # score = avg across two objects score
                    score += determine_position(locality, box1, box2) / 2
                elif obj1 in obj:
                    obj1_pos = obj.index(obj1)  
                    score = 0.25 * instance_score[obj1_pos].item()
                elif obj2 in obj:
                    obj2_pos = obj.index(obj2)
                    score = 0.25 * instance_score[obj2_pos].item()
                else:
                    score = 0
                if (score<0.5):
                    score=0

        return round(score, 4)

# if __name__ == "__main__":
#     # Example usage
#     evaluator = SpatialRelationEvaluator(device="cuda:0", complex=False)
#     # Load image
#     img = Image.open("/media/raid/workspace/miyapeng/T2I-CompBench/examples/samples/a blue bench and a green cake_000002.png").convert("RGB")
#     metadata = {"prompt": "a blue bench and a green cake", "tag": "spatial"}
#     score = evaluator.eval(img, metadata)
#     print(f"Spatial Relation Score: {score}")