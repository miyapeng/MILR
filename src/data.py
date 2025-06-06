"""
Data api
"""
from datasets import load_dataset, load_from_disk
from prompts import get_dataset

def get_dataset(data_name_or_path, tokenizer, prompt_idx):
    """
    Args:
        data_name_or_path: dataset name or path
        tokenizer: tokenizer
        prompt_idx: which query prompt to use
    Returns:
        dataset: dataset
    """

    ### Load dataset ### 
    # TODO:
    if "xxxx" in data_name_or_path:

    else:
        raise ValueError(f"Unsupported dataset: {data_name_or_path}")

    # preprocess dataset
    # Format the input for the model in the key {"formatted": }
    def preprocess_function(examples):
        '''
        Preprocess dataset

        Args:
            examples: dataset examples

        Returns:
            formatted: formatted dataset
        '''
        formatted = []
        questions = examples[question_col]
        for q in questions:
            # TODO
            if "xxx" in data_name_or_path:
                messages = get_prompt(q, prompt_idx)
            else:
                raise ValueError(f"Unsupported dataset: {data_name_or_path}")

            formatted.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))
        return {"formatted": formatted, "question": questions, "answer": examples[answer_col]}

    dataset = dataset.map(preprocess_function, batched=True, load_from_cache_file=False)
    return dataset

