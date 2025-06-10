# input_text = "A red apple on a wooden table with a blue vase beside it."
# cot_prompt = (
#         'You are asked to generate an image based on this prompt: "{}"\n'
#         'Provide a brief, precise visualization of all elements in the prompt. Your description should:\n'
#         '1. Include every object mentioned in the prompt\n'
#         '2. Specify visual attributes (color, number, shape, texture) if specified in the prompt\n'
#         '3. Clarify relationships (e.g., spatial) between objects if specified in the prompt\n'
#         '4. Be concise (50 words or less)\n'
#         "5. Focus only on what's explicitly stated in the prompt\n"
#         '6. Do not elaborate beyond the attributes or relationships specified in the prompt\n'
#         'Do not miss objects. Output your visualization directly without explanation:'
#     )
# formatted_cot_prompt = cot_prompt.format(input_text)
# print(formatted_cot_prompt)

from data import get_dataset
dataset = get_dataset("prompts/geneval/evaluation_metadata.jsonl")
print(f"Example: {dataset[0]}")