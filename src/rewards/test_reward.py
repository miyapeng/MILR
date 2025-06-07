from reward import RewardModel

model = RewardModel(
    model_path="<OBJECT_DETECTOR_FOLDER>",
    object_names_path="object_names.txt",
    options={"clip_model": "ViT-L-14"}
)

reward = model.get_reward(
    "/media/raid/workspace/miyapeng/Bagel/eval/gen/geneval/results/geneval_no_think/images/00530/samples/00000.png",
    '{"tag": "color_attr", "include": [{"class": "suitcase", "count": 1, "color": "purple"}, {"class": "pizza", "count": 1, "color": "orange"}], "prompt": "a photo of a purple suitcase and an orange pizza"}'
)
