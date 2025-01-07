from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import torch

save_directory = "./meta-llama"

# Argument parser for prompt
parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="Kim jest recydywista", help="Input prompt for the model")
args = parser.parse_args()

# Load model and tokenizer
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    save_directory,
    torch_dtype=torch.bfloat16,  # Use float16 or float32 if bfloat16 isn't supported
    device_map="auto",  # Automatically map the model to GPU/CPU
    max_memory={
        "cpu": "12GiB",        # Allocate up to 12 GB of CPU RAM
        "0": "13GiB"      # Leave ~1.7 GB of headroom on GPU
    }
)
print("Loaded model", model.hf_device_map)

tokenizer = AutoTokenizer.from_pretrained(save_directory)

# Prepare input
input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(model.device)

# Generate token by token
print("Generating text token by token...\n")
generated_ids = input_ids
output_text = args.prompt

for _ in range(512):  # Limit to 512 tokens
    outputs = model(input_ids=generated_ids)
    
    # Get the next token
    next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
    next_token = tokenizer.decode(next_token_id, skip_special_tokens=True)

    # Append to the generated sequence
    generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=-1)
    output_text += next_token

    # Display the token as it's generated
    print(next_token, end="", flush=True)  # Print without newline for seamless display

    # Stop if EOS token is generated
    if tokenizer.eos_token_id == next_token_id.item():
        break

# Final output
print("\n\nFull Generated Text:\n", output_text)
