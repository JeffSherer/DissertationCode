import os
from llava.instruct import instruct_generate, instruct_postprocess

# Set paths
input_path = 'data/instruct/llava_med_instruct_fig_captions.json'
output_path_gen = 'data/instruct/llava_med_instruct_60k_inline_mentions_gen.jsonl'
output_path_post = 'data/instruct/llava_med_instruct_60k_inline_mentions_post.json'

# Ensure the paths exist
assert os.path.exists(input_path), f"Input path does not exist: {input_path}"

# Step 1: Generate visual instruct tuning conversations
instruct_generate.main([
    '--input_path', input_path,
    '--output_path', output_path_gen,
    '--max-size', '60000',
    '--use_inline_mentions', 'True'
])

# Step 2: Post-process generated conversations
instruct_postprocess.main([
    '--input_path', output_path_gen,
    '--output_path', output_path_post
])

print("Script executed successfully!")

