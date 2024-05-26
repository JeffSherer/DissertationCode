import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Paths to the model directory and tokenizer directory
model_path = "./llava-med-7b"  # Ensure this path is correct
tokenizer_path = "./tokenizer_dir"  # Ensure this path is correct

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

# Test the tokenizer and model with a simple input
input_text = "Hello, how are you?"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Get the model output
with torch.no_grad():
    outputs = model.generate(input_ids, max_length=50)

# Decode the generated tokens
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Model loaded and test input processed successfully.")
print("Input text:", input_text)
print("Generated text:", output_text)



