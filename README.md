# Fine-tuning LLMs on supercomputers

Example scripts showing how to fine-tune LLMs on CSC's supercomputers.

The script `finetuning.py` runs fine-tuning with the IMDb movie
reviews dataset on a given Hugging Face model, by default it uses
[EleutherAI/gpt-neo-1.3B](https://huggingface.co/EleutherAI/gpt-neo-1.3B)
which fits comfortably into the GPU memory of a V100. You can select
another model with the `--model` argument.

The launch scripts are:

- `run-finetuning-puhti-gpu1.sh` - fine-tuning on Puhti with 1 GPU
- `run-finetuning-puhti-gpu4.sh` - fine-tuning on Puhti with one full node (4 GPUs)
- `run-finetuning-puhti-gpu8.sh` - fine-tuning on Puhti with two full nodes (8 GPUs in total)
- `run-finetuning-puhti-gpu4-accelerate.sh` - fine-tuning on Puhti with one full node using [Accelerate](https://huggingface.co/docs/transformers/accelerate)
- `run-finetuning-puhti-gpu8-accelerate.sh` - fine-tuning on Puhti with two full nodes using Accelerate

You can use [PEFT (Parameter-Efficient
Fine-Tuning)](https://huggingface.co/docs/peft/index) which adaptively
trains a smaller number of parameters, thus decreasing the GPU memory
requirements for training a lot. PEFT can be enabled with the `--peft`
argument.

The [Accelerate](https://huggingface.co/docs/transformers/accelerate)
library supports more advanced modes of distributed training such as
[FSDP](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)
which enables using models which are too large for a single GPU's
memory.

## Run examples

Run on 1 GPU with specified model and using PEFT:

```bash
sbatch run-finetuning-puhti-gpu1.sh --model=EleutherAI/gpt-neo-1.3B --peft
```

Run on 4 GPUs (note that batch_size has to be a multiple of the number of GPUs):
```bash
sbatch run-finetuning-puhti-gpu4.sh --model=EleutherAI/gpt-neo-1.3B --b 4
```

Run on 8 GPUs (over two nodes) with Accelerate and FSDP (note: with the accelerate launch script we need to specify which config file to use):

```bash
sbatch run-finetuning-puhti-gpu8-accelerate.sh accelerate_config_fsdp.yaml --model=microsoft/Phi-3.5-mini-instruct --b 8
```


## Inference

There's also a example of inference (generating text with the model)
in `inference-demo.py` with corresponding launch script
`run-inference-puhti.sh`.
