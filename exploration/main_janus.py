import os
from transformers import AutoModelForCausalLM
import torch
from process import get_dataset,save_image_and_metadata,set_seed
from tqdm import tqdm

from ori_generation_janus import original_generation
from opt_generation_janus import optimized_generation

import argparse
from janus.models import MultiModalityCausalLM, VLChatProcessor

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument("--dataset", type=str, default="prompts/geneval/evaluation_metadata.jsonl", help="Dataset to evaluate")
    parser.add_argument("--model_name_or_path", type=str, help="Path to the model")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--data_name", type=str, default="geneval", choices=["geneval", "T2I-CompBench","Wise"], help="Type of dataset to evaluate")
    parser.add_argument("--optimize_mode", type=str, default="text", help="The mode of optimization, must be one of: 'text', 'image', 'both'")
    parser.add_argument("--reward_model_type", type=str, default="geneval", choices=["geneval", "self_reward", "unified_reward","mixed_reward","T2I-CompBench","wise_reward"], help="Which reward model to use.")
    parser.add_argument("--start_data_idx", type=int, default=0, help="Start index of the data to evaluate")
    parser.add_argument("--end_data_idx", type=int, default=1319, help="End index of the data to evaluate")
    parser.add_argument("--task_type", type=str, default="color", help="Type of task for T2I-CompBench")

    # seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")

    # optimization args
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--grad_clip", type=float, default=None, help="Gradient clipping threshold")
    parser.add_argument("--text_k", type=float, default=0.1, help="Ratio of update length to the total length of hidden states")
    parser.add_argument("--image_k", type=float, default=0.01, help="Ratio of update length to the total length of hidden states")
    parser.add_argument("--max_text_steps", type=int, default=10, help="Number of text optimization iterations")
    parser.add_argument("--max_image_steps", type=int, default=10, help="Number of image optimization iterations")
    parser.add_argument("--max_both_steps", type=int, default=10, help="Number of both(text and image) optimization iterations")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Number of generated tokens")
    parser.add_argument("--device", type=str, default=None)

    # reward model
    parser.add_argument("--reward_threshold", type=float, default=-0.1, help="Threshold for reward to stop optimization")

    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    return parser.parse_args()


# evaluate function 
def main(args):
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

    if args.reward_model_type == "geneval":
        from rewards.reward import RewardModel
        reward_model = RewardModel(
            model_path="rewards/<OBJECT_DETECTOR_FOLDER>",
            object_names_path="rewards/object_names.txt",
            options={"clip_model": "ViT-L-14"}
        )
    elif args.reward_model_type == "self_reward":
        from rewards.self_reward_janus import SelfRewardModel
        reward_model = SelfRewardModel(vl_gpt=vl_gpt, vl_chat_processor=vl_chat_processor, device=device)
    elif args.reward_model_type == "T2I-CompBench":
        from rewards.T2ICompBench.reward import CompBenchRewardModel
        reward_model = CompBenchRewardModel(task_type=args.task_type, device=device)
    elif args.reward_model_type == "unified_reward":
        from rewards.unified_reward import UnifiedReward
        reward_model = UnifiedReward(
            model_path='CodeGoat24/UnifiedReward-qwen-7b',
            device=device
        )
    elif args.reward_model_type == "mixed_reward":
        from rewards.MixedReward.reward3 import MixedReward
        reward_model = MixedReward(
            git_ckpt_path="./rewards/MixedReward/reward_weights/git-large-vqav2",
            unified_model_path="CodeGoat24/UnifiedReward-qwen-7b",
            gdino_ckpt_path="./rewards/MixedReward/reward_weights/groundingdino_swint_ogc.pth",
            gdino_config_path="./rewards/MixedReward/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            device=device
        )
    elif args.reward_model_type == "wise_reward":
        from rewards.wise_reward import WiseReward
        reward_model = WiseReward(
            #api_key='64cd78bc94b8b7d6f02ee4263c3ed709', 
            model='gpt-4o-2024-05-13',
        )

    # load dataset
    dataset = get_dataset(args.dataset,args.task_type,args.data_name)
    print(f"Example: {dataset[0]}")

    original_correct = 0
    optimized_correct = 0
    total = 0
    update_count = 0
    original_length = 0
    optimized_length = 0
    fitten_length = 0
    model_name = args.model_name_or_path.split("/")[-1]
    data_name = args.data_name

    if args.optimize_mode == "text":
        output_dir = f"{args.output_dir}/{model_name}-{data_name}-{args.reward_model_type}-{args.optimize_mode}-text_k{args.text_k}-steps{args.max_text_steps}-lr{args.lr}-reward_threshold{args.reward_threshold}"
    elif args.optimize_mode == "image":
        output_dir = f"{args.output_dir}/{model_name}-{data_name}-{args.reward_model_type}-{args.optimize_mode}-image_k{args.image_k}-steps{args.max_image_steps}-lr{args.lr}-reward_threshold{args.reward_threshold}"
    else:
        output_dir = f"{args.output_dir}/{model_name}-{data_name}-{args.reward_model_type}-{args.optimize_mode}-text_k{args.text_k}-image_k{args.image_k}-steps{args.max_both_steps}-lr{args.lr}-reward_threshold{args.reward_threshold}"

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

        img, text_hidden_states_list, text_final_input_ids, image_hidden_states_list, image_prompt_embed, ori_image_prompt = original_generation(
                input_text=prompt,
                model=vl_gpt,
                vl_chat_processor=vl_chat_processor,
                optimize_mode = args.optimize_mode,
                device=device)
        # save original image and metadata
        if img is not None:
            save_image_and_metadata(img, example, os.path.join(output_dir, "ori_img"), i, data_name)
        # print(f"ori_image_prompt:{ori_image_prompt}")
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
                ori_image_prompt=ori_image_prompt,
                max_text_steps=args.max_text_steps,
                max_image_steps=args.max_image_steps,
                max_both_steps=args.max_both_steps,
                lr=args.lr,
                grad_clip=args.grad_clip,
                text_k=args.text_k,
                image_k=args.image_k,
                reward_threshold=args.reward_threshold,
                max_text_tokens=args.max_new_tokens,
                optimize_mode = args.optimize_mode,
                save_base_path = os.path.join(output_dir, "opt_history", str(i).zfill(4)),
        )
        if new_img is not None:
            save_image_and_metadata(new_img, example, os.path.join(output_dir, "opt_img"), i,data_name)
        
        final_img = new_img if new_img is not None else img
        if final_img is not None:
            save_image_and_metadata(final_img, example, os.path.join(output_dir, "final_img"), i, data_name)
            
        update_count += (len(reward_history) - 1)   
        
        # extract answer from model response
        original_length += ori_total_length
        optimized_length += generated_seq
        fitten_length += (generated_seq - update_length) if len(reward_history) > 1 else 0

        # judge answer
        if img is not None:
            original_correct_add = reward_model.judge_answer(img,example)
        else:
            original_correct_add = False
        
        if new_img is not None:
            optimized_correct_add = reward_model.judge_answer(new_img,example)
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


