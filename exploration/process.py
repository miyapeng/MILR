import json
import os
from PIL import Image
import torch
import numpy as np
import random

def get_dataset(data_path: str,task_type: str, data_name: str):
    """
    Load the dataset.

    - JSONL: Parse each non-empty line into a dictionary and return as a list.
      The file is expected to contain keys like "tag", "prompt", and optionally "include".
    - TXT: Treat each non-empty line as a prompt.
      The file's base name (without extension) is used as the "tag" for all entries.

    Args:
        data_path (str): Path to a .jsonl or .txt file.

    Returns:
        list[dict]: A list of dictionaries. Each dictionary contains at least "tag" and "prompt".
    """
    data_path = data_path.strip()

    if data_name == "geneval":
        data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Error decoding JSONL line: {line}\n{e}")
        return data

    elif data_name == "T2I-CompBench":
        data = []
        file_name = os.path.splitext(os.path.basename(data_path))[0]
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append({"tag": task_type, "prompt": line})
        return data
    elif data_name == "Wise":
        data = []
        
        with open(data_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        
        if content.startswith("["):
            try:
                items = json.loads(content)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error decoding JSON array in {data_path}:\n{e}")
        else:
            
            items = []
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Error decoding JSONL line: {line}\n{e}")

        
        for item in items:
            data.append({
                "tag": item.get("Category", ""),
                "subcategory": item.get("Subcategory", ""),
                "explanation": item.get("Explanation", ""),
                "prompt": item.get("Prompt", ""),
                "prompt_id": item.get("prompt_id")
            })
        return data

    else:
        raise ValueError(f"Unsupported dataset format: {data_path}")


def save_image_and_metadata(image: Image.Image, example: dict, base_path: str, index: int, data_name: str):
    if data_name == "geneval":
        folder_name = str(index).zfill(5)
        sample_folder = os.path.join(base_path, folder_name, "samples")
        os.makedirs(sample_folder, exist_ok=True)

        # Save image
        img_path = os.path.join(sample_folder, "0000.png")
        image.save(img_path)

        # Write metadata.jsonl
        metadata_path = os.path.join(base_path, folder_name, "metadata.jsonl")
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    elif data_name == "T2I-CompBench":
        # Place all T2I-CompBench samples into base_path/samples
        sample_folder = os.path.join(base_path, "samples")
        os.makedirs(sample_folder, exist_ok=True)

        # Create a safe filename prefix from the prompt
        prompt = example.get("prompt", "")
        safe_prompt = prompt.rstrip('.')

        # Use index as a 6-digit sequence number with leading zeros
        filename = f"{safe_prompt}_{index:06}.png"
        img_path = os.path.join(sample_folder, filename)
        image.save(img_path)

        # Append metadata.jsonl in base_path
        metadata_log = os.path.join(base_path, "metadata.jsonl")
        # Include the image filename in the record
        record = {**example, "image": filename}
        with open(metadata_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    elif data_name == "Wise":
        # Store images in base_path/samples
        sample_folder = os.path.join(base_path, "samples")
        os.makedirs(sample_folder, exist_ok=True)

        # Use index+1 for filenames: 1.png, 2.png, ...
        index = example.get("prompt_id", "")
        filename = f"{index}.png"
        img_path = os.path.join(sample_folder, filename)
        image.save(img_path)

        # Write metadata to wise_metadata.jsonl (append mode)
        meta_path = os.path.join(base_path, "wise_metadata.jsonl")
        record = {"filename": filename, **example}
        with open(meta_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    else:
        raise ValueError(f"Unsupported data_name: {data_name}")

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

# if __name__ == '__main__':
#     #dataset = get_dataset("prompts/geneval/evaluation_metadata.jsonl")
#     dataset = get_dataset("/media/raid/workspace/miyapeng/Multimodal-LatentSeek/src/prompts/geneval/evaluation_metadata.jsonl")
#     print(f"Example: {dataset[0]}")


# """
# Data api
# """
# from datasets import load_dataset, load_from_disk
# from prompts import get_dataset

# def get_dataset(data_name_or_path, tokenizer, prompt_idx):
#     """
#     Args:
#         data_name_or_path: dataset name or path
#         tokenizer: tokenizer
#         prompt_idx: which query prompt to use
#     Returns:
#         dataset: dataset
#     """

#     ### Load dataset ### 
#     # TODO:
#     if "xxx" in data_name_or_path:
        
#     else:
#         raise ValueError(f"Unsupported dataset: {data_name_or_path}")

#     # preprocess dataset
#     # Format the input for the model in the key {"formatted": }
#     def preprocess_function(examples):
#         '''
#         Preprocess dataset

#         Args:
#             examples: dataset examples

#         Returns:
#             formatted: formatted dataset
#         '''
#         formatted = []
#         questions = examples[question_col]
#         for q in questions:
#             # TODO
#             if "xxx" in data_name_or_path:
#                 messages = get_prompt(q, prompt_idx)
#             else:
#                 raise ValueError(f"Unsupported dataset: {data_name_or_path}")

#             formatted.append(tokenizer.apply_chat_template(
#                 messages, tokenize=False, add_generation_prompt=True
#             ))
#         return {"formatted": formatted, "question": questions, "answer": examples[answer_col]}

#     dataset = dataset.map(preprocess_function, batched=True, load_from_cache_file=False)
#     return dataset

