import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/gpt-neo-1.3B")
    parser.add_argument(
        "--prompt",
        type=str,
        default="The movie about how AI will take over the world was great because",
    )
    args = parser.parse_args()
    print(f"Loading model from: {args.model}")
    print(f"Prompt: {args.prompt}")
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"ROCm/CUDA available: {n_gpus} GCD(s)")
        for i in range(n_gpus):
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GCD {i}: {torch.cuda.get_device_name(i)} ({total:.1f} GB)")
    else:
        print("No GPU available, exiting")
        exit(1)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        offload_state_dict=False,
        attn_implementation="eager",
    )
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(args.prompt, return_tensors="pt").to("cuda:0")
        outputs = model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=80,
            num_return_sequences=4,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print("Sample generated reviews:")
    for i, txt in enumerate(decoded_outputs):
        print("#######################")
        print(f"{i + 1}: {txt}")
