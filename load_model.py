from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type = str, default = "meta-llama/Meta-Llama-3.1-8B-Instruct")
args = parser.parse_args()

save_dir = "./meta-llama"

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    args.model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.save_pretrained(save_dir)
print("Saved ", args.model_id)

tokenizer = AutoTokenizer.from_pretrained(args.model_id)
tokenizer.save_pretrained(save_dir)
print("Saved tokenizer")

