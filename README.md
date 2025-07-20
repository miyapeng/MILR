
## Installation

```bash
conda create -n latentseek python=3.10
conda activate latentseek
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

#install Geneval configs
#You may meet package counters, it doesn't matter
pip install -U openmim
mim install mmengine mmcv-full==1.7.2
cd src/geneval
./evaluation/download_models.sh "<OBJECT_DETECTOR_FOLDER>/"
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; git checkout 2.x
pip install -v -e .

cd ../rewards
./evaluation/download_models.sh "<OBJECT_DETECTOR_FOLDER>/"
```
If you meet error with flash_attn==2.7.2.post1, you can refer to the `https://github.com/Dao-AILab/flash-attention/releases` to download.

## Usage
We support different kinds of reward types. And we test on three benchmarks: **Geneval**, **T2I-CompBench**, **Wise**

### Geneval

```bash
cd src
bash scripts/geneval_both.sh
```

The bash file

```bash
#!/bin/bash
PATH_TO_DATA="prompts/geneval/evaluation_metadata.jsonl"
PATH_TO_MODEL="deepseek-ai/Janus-Pro-7B"
output_dir="./geneval_results/long_results" #self create the dir
optimize_mode="both"  # or "image"
reward_model_type="geneval"
text_k=0.1 
image_k=0.01 
lr=0.01
max_text_steps=30
max_image_steps=30
max_both_steps=30

# === set log file name ===
if [ "$optimize_mode" = "text" ]; then
    LOG_FILE="$output_dir/${optimize_mode}_tk${text_k}_lr${lr}_ts${max_text_steps}.txt"
elif [ "$optimize_mode" = "image" ]; then
    LOG_FILE="$output_dir/${optimize_mode}_ik${image_k}_lr${lr}_is${max_image_steps}.txt"
else
    LOG_FILE="$output_dir/${optimize_mode}_tk${text_k}_ik${image_k}_lr${lr}_bs${max_both_steps}.txt"
fi

# === train script ===
CUDA_VISIBLE_DEVICES=1 python main_janus.py \
    --dataset "$PATH_TO_DATA" \
    --model_name_or_path "$PATH_TO_MODEL" \
    --output_dir "$output_dir" \
    --optimize_mode "$optimize_mode" \
    --reward_model_type "$reward_model_type" \
    --lr "$lr" \
    --text_k "$text_k" \
    --image_k "$image_k" \
    --max_text_steps "$max_text_steps" \
    --max_image_steps "$max_image_steps" \
    --max_both_steps "$max_both_steps" \
    --device "cuda" \
    > "$LOG_FILE" 2>&1 &
```
- `optimize_mode`: The mode of optimization, you can choose from `both`, `image` or `text`.
- `reward_model_type`: the reward model used for optimize, you can check in the main_janus.py file
- `text_k`: the ratio of text tokens for optimization
- `image_k`: the ratio of image tokens for optimization
- `lr`: the learning rate
- `max_text_steps`: the steps of text optimization
- `max_image_steps`: the steps of image optimization
- `max_both_steps`: the steps of both optimization

### T2I-CompBench
Coming soon....

## Files for Modification

* Main logic file: [main](./src/main.py)
* Opt generation file (LatentSeek core): [opt](./src/opt_generation.py)
* CoT generation file (original generation): [ori](./src/ori_generation.py)
* Data: [data](./src/data.py)
* Reward Model: [reward](./src/rewards/reward.py)
* Self-Reward Prompts: [self-reward prompts](./src/prompts/vera_prompts.py)
* CoT Prompts: [CoT prompts](./src/prompts/solver_prompts.py)


