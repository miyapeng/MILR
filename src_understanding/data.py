import json

def get_dataset(data_path):
    """
    仅加载 JSONL 文件，每行一个 dict，保持原始结构不变。
    """
    if data_path.endswith('.jsonl'):
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 直接用 Python json 解析，允许不同字段类型混用
                data.append(json.loads(line))
        return data

    # 如果需要支持其他格式，再在这里拓展
    raise ValueError(f"Unsupported dataset format: {data_path}")

if __name__ == '__main__':
    from data import get_dataset

    dataset = get_dataset("prompts/geneval/evaluation_metadata.jsonl")
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

