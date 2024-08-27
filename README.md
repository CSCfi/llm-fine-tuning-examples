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

You can use [PEFT (Parameter-Efficient
Fine-Tuning)](https://huggingface.co/docs/peft/index) which adaptively
trains a smaller number of parameters, thus decreasing the GPU memory
requirements for training a lot. PEFT can be enabled with the `--peft`
argument.

## Run examples

Run on 1 GPU with specified model and using PEFT:

```bash
sbatch run-finetuning-puhti-gpu1.sh --model=EleutherAI/gpt-neo-1.3B --peft
```

Run on 4 GPUs (note that batch_size has to be a multiple of the number of GPUs):
```bash
sbatch run-finetuning-puhti-gpu4.sh --model=EleutherAI/gpt-neo-1.3B --b 4
```

## Distributed strategy

Note that currently the multi-gpu version uses distributed data
parallel (DDP), which puts a full copy of the model on each GPU and
splits up the training batch. This will speed up training (compared to
using a single GPU), but not save any GPU memory. 

Examples using FSDP, which should be able to handle models too big for
a single GPU, will be added later.

## Inference

There's also a example of inference (generating text with the model)
in `inference-demo.py` with corresponding launch script
`run-inference-puhti.sh`.
