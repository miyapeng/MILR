import json
import os

def get_dataset(data_path: str):
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

    if data_path.endswith(".jsonl"):
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

    elif data_path.endswith(".txt"):
        data = []
        file_name = os.path.splitext(os.path.basename(data_path))[0]
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append({"tag": file_name, "prompt": line})
        return data

    else:
        raise ValueError(f"Unsupported dataset format: {data_path}")


if __name__ == '__main__':
    #dataset = get_dataset("prompts/geneval/evaluation_metadata.jsonl")
    dataset = get_dataset("/media/raid/workspace/miyapeng/Multimodal-LatentSeek/src/prompts/geneval/evaluation_metadata.jsonl")
    print(f"Example: {dataset[0]}")


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

