
## Installation

```bash
conda create -n latentseek python=3.10
conda activate latentseek
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

#install Geneval configs
#You may meet package counters, it doesn't matter`
pip install -U openmim
mim install mmengine mmcv-full==1.7.2
cd src/rewards
./evaluation/download_models.sh "<OBJECT_DETECTOR_FOLDER>/"
```

## Usage
We support different kinds of reward types.

### Geneval

```bash
cd src
bash scripts/example.sh
```

The example.sh file

```bash
#!/bin/bash
## the prompt dataset
PATH_TO_DATA="prompts/geneval/evaluation_metadata.jsonl" 
## the model name or path
PATH_TO_MODEL="deepseek-ai/Janus-Pro-7B"
## the output dir
output_dir="./geneval_results/step_scaling_results"
## the optimize mode, you can choose from "both","text","image", both contain text and image optimization
optimize_mode="both"  # or "image","text"
# the reward model type, you can choose from "geneval", "self_reward"
# "unified_reward", "mixed_reward"
reward_model_type="geneval"
# the ratio of text token
text_k=0.1
# the ratio of image token 
image_k=0.01 
## the learning rate
lr=0.01
## the text steps if you choose text mode
max_text_steps=30
## the image step if you choose image mode
max_image_steps=30
## the both step if you choose both mode
max_both_steps=30

# === 设置日志文件名 ===
if [ "$optimize_mode" = "text" ]; then
    LOG_FILE="$output_dir/${optimize_mode}_tk${text_k}_lr${lr}_ts${max_text_steps}.txt"
elif [ "$optimize_mode" = "image" ]; then
    LOG_FILE="$output_dir/${optimize_mode}_ik${image_k}_lr${lr}_is${max_image_steps}.txt"
else
    LOG_FILE="$output_dir/${optimize_mode}_tk${text_k}_ik${image_k}_lr${lr}_bs${max_both_steps}.txt"
fi

# === 启动训练脚本 ===
CUDA_VISIBLE_DEVICES=0 python main_janus.py \
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


