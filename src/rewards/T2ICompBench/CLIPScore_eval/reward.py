import torch
import clip
from PIL import Image
from typing import Dict, Any, Union
import spacy


class ClipSimilarityEvaluator:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu", complex: bool = False):
        self.device = device
        self.complex = complex
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.nlp = spacy.load("en_core_web_sm")

    def eval(self, image: Image.Image, metadata: Dict[str, Any]) -> float:
        """
        评估单张图像和 prompt 的 CLIP 相似度，返回 [0, 1] 区间内的分数。
        metadata 需包含 "prompt" 字段。
        """
        prompt = metadata.get("prompt", "")
        if not prompt:
            raise ValueError("metadata 必须包含 'prompt' 字段")

        # 预处理图像
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        # 文本处理
        if self.complex:
            doc = self.nlp(prompt)
            prompt =' '.join([token.text for token in doc if token.pos_ != 'ADJ'])

        text_tensor = clip.tokenize(prompt).to(self.device)

        # 相似度计算
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tensor)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).squeeze().item()
        return round(similarity, 4)

if __name__ == "__main__":
    from PIL import Image

    evaluator = ClipSimilarityEvaluator(complex=False)
    img = Image.open("/media/raid/workspace/miyapeng/T2I-CompBench/examples/samples/a green bench and a blue bowl_000000.png").convert("RGB")
    metadata = {"prompt": "a green bench and a blue bowl", "tag": "non-spatial"}

    score = evaluator.eval(img, metadata)
    print(f"CLIP 相似度得分: {score}")
