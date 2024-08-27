import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="EleutherAI/gpt-neo-1.3B",
        help="The pre-trained model from Hugging Face to use as basis: "
        "https://huggingface.co/models"
    )
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device {device}")
    if device.type == 'cuda':
        print(f"Device name is {torch.cuda.get_device_name(device)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)

    with torch.no_grad():
        prompt = "The movie 'How to run ML on LUMI - A documentation' was great because"
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        outputs = model.generate(**inputs, do_sample=True, max_length=80, num_return_sequences=4)
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        print('Sample generated reviews:')
        for i, txt in enumerate(decoded_outputs):
            print("#######################")
            print(f"{i+1}: {txt}")

    # for device_id in range(torch.cuda.device_count()):
    #     print(f"- GPU {device_id} max memory allocated: "
    #           f"{torch.cuda.max_memory_allocated(device_id)/1024/1024:.2f}MB")
