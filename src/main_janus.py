import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from data import get_dataset
from tqdm import tqdm
from rewards.reward import RewardModel
from ori_generation_janus import original_generation
from opt_generation_janus import optimized_generation

import argparse
import numpy as np
import random
import json
from PIL import Image

from janus.models import MultiModalityCausalLM, VLChatProcessor

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument("--dataset", type=str, default="prompts/geneval/evaluation_metadata.jsonl", help="Dataset to evaluate")
    parser.add_argument("--model_name_or_path", type=str, help="Path to the model")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--optimize_mode", type=str, default="text", help="The mode of optimization, must be one of: 'text', 'image', 'both'")
    parser.add_argument("--start_data_idx", type=int, default=0, help="Start index of the data to evaluate")
    parser.add_argument("--end_data_idx", type=int, default=1319, help="End index of the data to evaluate")

    # seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")

    # optimization args
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--grad_clip", type=float, default=None, help="Gradient clipping threshold")
    parser.add_argument("--text_k", type=float, default=0.1, help="Ratio of update length to the total length of hidden states")
    parser.add_argument("--image_k", type=float, default=0.01, help="Ratio of update length to the total length of hidden states")
    parser.add_argument("--max_text_steps", type=int, default=10, help="Number of text optimization iterations")
    parser.add_argument("--max_image_steps", type=int, default=10, help="Number of image optimization iterations")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Number of generated tokens")
    parser.add_argument("--device", type=str, default=None)

    # reward model
    parser.add_argument("--reward_threshold", type=float, default=-0.1, help="Threshold for reward to stop optimization")

    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    return parser.parse_args()


def set_seed(seed):
    '''
    Set random seed for reproducibility

    Args:
        seed: random seed
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

def save_image_and_metadata(image: Image.Image, example: dict, base_path: str, index: int):
    folder_name = str(index).zfill(5)
    sample_folder = os.path.join(base_path, folder_name, "samples")
    os.makedirs(sample_folder, exist_ok=True)

    # 保存图片
    img_path = os.path.join(sample_folder, "0000.png")
    image.save(img_path)

    # 保存metadata.jsonl
    metadata_path = os.path.join(base_path, folder_name, "metadata.jsonl")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

def judge_answer(output: str, reward_model, data):
    '''
    Judge whether the output is correct

    Args:
        output: model output
        reward_model: reward model to judge the output
        data: data to judge the output

    Returns:
        bool: whether the output is correct
    '''
        
    reward_score = reward_model.get_reward(output,data)
    if reward_score == -1:
        return False
    elif reward_score == 0:
        return True

# evaluate function 
def main(args):
    '''
    Evaluate model

    Args:
        dataset: dataset to evaluate
        sample_num: number of samples to evaluate

    Returns:
        original_accuracy: original generation accuracy
        optimized_accuracy: optimized generation accuracy
    '''
    
    if args.seed:
        set_seed(args.seed)
    
    # set device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # load model and tokenizer
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model_name_or_path)

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    # load reward model
    reward_model = RewardModel(
            model_path="rewards/<OBJECT_DETECTOR_FOLDER>",
            object_names_path="rewards/object_names.txt",
            options={"clip_model": "ViT-L-14"}
        )

    # load dataset
    dataset = get_dataset(args.dataset)
    print(f"Example: {dataset[0]}")

    original_correct = 0
    optimized_correct = 0
    total = 0
    update_count = 0
    original_length = 0
    optimized_length = 0
    fitten_length = 0
    model_name = args.model_name_or_path.split("/")[-1]
    data_name = "geneval"

    if args.optimize_mode == "text":
        output_dir = f"{args.output_dir}/{model_name}-{data_name}-{args.optimize_mode}-text_k{args.text_k}-steps{args.max_text_steps}-lr{args.lr}"
    else:
        output_dir = f"{args.output_dir}/{model_name}-{data_name}-{args.optimize_mode}-image_k{args.image_k}-steps{args.max_image_steps}-lr{args.lr}"

    start_data_idx = max(0, args.start_data_idx)
    end_data_idx = min(args.end_data_idx, len(dataset))
  
    if args.resume:
        print(f"Resume from {output_dir}")
        # load logistics
        logistics = torch.load(f"{output_dir}/logistics.pt")
        start_data_idx = logistics["start_idx"]
        original_correct = logistics["original_correct"]
        optimized_correct = logistics["optimized_correct"]
        total = logistics["total"]
        update_count = logistics["update_count"]
        original_length = logistics["original_length"]
        optimized_length = logistics["optimized_length"]
        fitten_length = logistics["fitten_length"]

    
    print(f"Start to evaluate {data_name} from {start_data_idx} to {end_data_idx}...")

    data_idx_list = range(start_data_idx, end_data_idx)
    for i in tqdm(data_idx_list):
        example = dataset[i]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        prompt = example["prompt"]
        task_tag = example["tag"]

        print(f"Task_tag: {task_tag}")
        print(f"prompt: {prompt}")
        if prompt is None:
            continue

        img, text_hidden_states_list, text_final_input_ids, image_hidden_states_list, image_prompt_embed, generated_image_tokens = original_generation(
                input_text=prompt,
                model=vl_gpt,
                vl_chat_processor=vl_chat_processor,
                device=device)
        # save original image and metadata
        if img is not None:
            save_image_and_metadata(img, example, os.path.join(output_dir, "ori_img"), i)

        torch.cuda.empty_cache()
        new_img, reward_history, ori_total_length, generated_seq, update_length = optimized_generation(
                reward_model=reward_model,
                image=img,
                data=example,
                model=vl_gpt,
                vl_chat_processor = vl_chat_processor,
                device=device,
                text_hidden_states_list=text_hidden_states_list,
                text_final_input_ids=text_final_input_ids,
                image_hidden_states_list=image_hidden_states_list,
                image_prompt_embed=image_prompt_embed,
                generated_image_tokens=generated_image_tokens,
                start_index= start_data_idx,
                max_text_steps=args.max_text_steps,
                max_image_steps=args.max_image_steps,
                lr=args.lr,
                grad_clip=args.grad_clip,
                text_k=args.text_k,
                image_k=args.image_k,
                reward_threshold=args.reward_threshold,
                max_text_tokens=args.max_new_tokens,
                optimize_mode = args.optimize_mode,
                save_base_path = os.path.join(output_dir, "opt_history", str(i).zfill(4))
        )
        if new_img is not None:
            save_image_and_metadata(new_img, example, os.path.join(output_dir, "opt_img"), i)
        
        final_img = new_img if new_img is not None else img
        if final_img is not None:
            save_image_and_metadata(final_img, example, os.path.join(output_dir, "final_img"), i)
            
        update_count += (len(reward_history) - 1)   
        
        # extract answer from model response
        original_length += ori_total_length
        optimized_length += generated_seq
        fitten_length += (generated_seq - update_length) if len(reward_history) > 1 else 0

        # judge answer
        if img is not None:
            original_correct_add = judge_answer(img, reward_model,data=example)
        else:
            original_correct_add = False

        if new_img is not None:
            optimized_correct_add = judge_answer(new_img, reward_model, data=example)
        else:
            optimized_correct_add = False

        original_correct += original_correct_add
        optimized_correct += optimized_correct_add

        total += 1
        
        # save logistics
        # save original correct, optimized correct, total, update count
        torch.save({
            "original_correct": original_correct,
            "optimized_correct": optimized_correct,
            "total": total,
            "start_idx": i+1,
            "update_count": update_count,
            "original_length": original_length,
            "optimized_length": optimized_length,
            "fitten_length": fitten_length
        }, f"{output_dir}/logistics.pt")

    print(f"Original accuracy: {original_correct / total:.4f}")
    print(f"Optimized accuracy: {optimized_correct / total:.4f}")
    print(f"Average update length: {update_count / total:.4f}")
    print(f"Average original length: {original_length / total:.4f}")
    print(f"Average optimized length: {optimized_length / total:.4f}")
    print(f"Average fitten length: {fitten_length / total:.4f}")       


if __name__ == "__main__":
    args = parse_args()
    for arg in vars(args):
        print(f"-- {arg}: {getattr(args, arg)}")
    main(args)


