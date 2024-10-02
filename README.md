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

There are also versions for Mahti, similarly named: just replace
`puhti` with `mahti`. The scripts for LUMI are named slightly
different due to the larger number of GPUs (or actually GCDs, due to
the dual chip cards).

- `run-finetuning-lumi-gpu1.sh` - fine-tuning on LUMI with 1 GPU
- `run-finetuning-lumi-gpu8.sh` - fine-tuning on LUMI with one full node (8 GCDs)
- `run-finetuning-lumi-gpu16.sh` - fine-tuning on LUMI with two full nodes (16 GCDs in total)
- `run-finetuning-lumi-gpu8-accelerate.sh` - fine-tuning on LUMI with one full node using [Accelerate](https://huggingface.co/docs/transformers/accelerate)
- `run-finetuning-lumi-gpu16-accelerate.sh` - fine-tuning on LUMI with two full nodes using Accelerate

**Note:** the scripts are for the most part made to be run in the
`gputest` or `dev-g` partition with a 15 minute time-limit. You
naturally need to change to the proper partition for longer jobs for
your real runs. Also change the `--account` parameter to your own
project code.

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

Finally, by installing the `bitsandbytes` library, you can also try
4-bit quantization with the `--4bit` argument, decreasing even further
the memory requirements.

## Run examples

Run on 1 GPU with specified model and using PEFT:

```bash
sbatch run-finetuning-puhti-gpu1.sh --model=EleutherAI/gpt-neo-1.3B --peft
```

Run on 4 GPUs (note that batch_size has to be a multiple of the number of GPUs):
```bash
sbatch run-finetuning-puhti-gpu4.sh --model=EleutherAI/gpt-neo-1.3B --b 4
```

Run on 8 GPUs (over two nodes) with Accelerate and FSDP (note: with
the accelerate launch script we need to specify which config file to
use):

```bash
sbatch run-finetuning-puhti-gpu8-accelerate.sh accelerate_config_fsdp.yaml \
       --model=microsoft/Phi-3.5-mini-instruct --b 8
```

**Note:** for a new model it might work best if your first run is with
a single GPU to get the model downloaded to the cache. Downloading
with multiple processes doesn't yet work well in the current script.

Fine-tune Llama-3.1-8B on Mahti with just 2 GPUs using Accelerate,
FSDP and PEFT:

```bash
sbatch run-finetuning-mahti-gpu2-accelerate.sh accelerate_config_fsdp.yaml \
       --model=meta-llama/Meta-Llama-3.1-8B --b 4 --peft
```

Note that the `Meta-LLama-3.1-8B` model is a "Gated model" on Hugging
Face, it requires that you log in and ask for access to the
model. Once you have recieved access you can generate an [Access Token
in Hugging Face](https://huggingface.co/settings/tokens). Just click
"Create new token" and select Token type: "Read".

On the supercomputer you can then install the access token like this:

```bash
export HF_HOME=/scratch/YOUR_PROJECT/${USER}/hf-cache
module load pytorch/2.4
huggingface-cli login
```

In the above command you need to replace `YOUR_PROJECT` with the
project you use for your runs. The important thing is just that you
use the same Hugging Face cache path as in the scripts. The
`huggingface-cli login` command will ask for the access token you
created. Just reply `n` to the question about git credentials (unless
you know you use that feature).

Fine-tune Llama-3.1-8B with 4bit quantization and PEFT on Puhti:

```bash
sbatch run-finetuning-puhti-gpu1.sh --model=meta-llama/Meta-Llama-3.1-8B \
       --b 4 --peft --4bit
```

Another example for fine-tuning the Poro model on LUMI with just 2
GPUs (thanks to PEFT/LoRA and 4bit quantization):

```bash
sbatch run-finetuning-lumi-gpu2.sh --model=LumiOpen/Poro-34B --b 8 --4bit --peft
```


## Inference

There's also a example of inference (generating text with the model)
in `inference-demo.py` with corresponding launch script
`run-inference-puhti.sh`.

For example to run inference with a checkpoint of a model you have
fine-tuned previously, you would run something like:

```bash
sbatch run-inference-puhti.sh --model=/PATH/TO/CHECKPOINT \
       --prompt="The movie was great because"
```

Naturally, you need to replace `/PATH/TO/CHECKPOINT` with the real
path to the checkpoint you wish to use. The path where checkpoints are
stored will usually be printed at the end of the job, but you need to
check yourself what is the specific checkpoint you wish to use. With
the above training scripts it will usually be something like
`/scratch/YOUR_PROJECT/${USER}/hf-data/MODEL_NAME/checkpoint-NNN/`
where `NNN` is the number of training steps when the checkpoint was
created.
