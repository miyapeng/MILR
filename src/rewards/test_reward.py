from reward import RewardModel
from PIL import Image

model = RewardModel(
    model_path="<OBJECT_DETECTOR_FOLDER>",
    object_names_path="object_names.txt",
    options={"clip_model": "ViT-L-14"}
)
img = Image.open("/media/raid/workspace/miyapeng/Bagel/eval/gen/geneval/results/geneval_no_think/images/00530/samples/00000.png")
solution = '{"tag": "color_attr", "include": [{"class": "suitcase", "count": 1, "color": "purple"}, {"class": "pizza", "count": 1, "color": "orange"}], "prompt": "a photo of a purple suitcase and an orange pizza"}'

reward = model.get_reward(img, solution)
print(f"Reward: {reward}")
