
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

```bash
cd src
cd scripts
vim example.sh # modify this according to your need
cd ..
sh scripts/example.sh
```

The example.sh file

```bash
PATH_TO_DATA= # path to the dataset (the path str should contain either "AIME_2024", "gsm8k", "MATH-500")
PATH_TO_MODEL= # path to the model 
rho= # the value of rho, which is the hyperparameter for the fractional update
lr= # the learning rate
solver_prompt_idx= # the index of the solver prompt to use (0 for "boxex", 1 for "json")

python main.py \
    --dataset $PATH_TO_DATA \
    --model_name_or_path $PATH_TO_MODEL \
    --output_dir ./output \
    --k $rho \
    --lr $lr \
    --solver_prompt_idx $solver_prompt_idx \
    --device "cuda" \
```

## Files for Modification

* Main logic file: [main](./src/main.py)
* Opt generation file (LatentSeek core): [opt](./src/opt_generation.py)
* CoT generation file (original generation): [ori](./src/ori_generation.py)
* Data: [data](./src/data.py)
* Reward Model: [reward](./src/rewards/reward.py)
* Self-Reward Prompts: [self-reward prompts](./src/prompts/vera_prompts.py)
* CoT Prompts: [CoT prompts](./src/prompts/solver_prompts.py)


