#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=0:15:00
#SBATCH --gres=gpu:v100:1

module purge
module load pytorch/2.4

# This will store all the Hugging Face cache such as downloaded models
# and datasets in the project's scratch folder
export HF_HOME=/scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf-cache

if [ -d "$LOCAL_SCRATCH" ]; then        # use local scratch if we have reserved NVME
    HF_HOME=${LOCAL_SCRATCH}/hf-cache   # note: this will cause model
                                        # to be redownloaded every
                                        # time
fi

mkdir -p $HF_HOME
#cp ~/.cache/huggingface/token ${HF_HOME}/

# Path to where the trained model and logging data will go
OUTPUT_DIR=/scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf-data
mkdir -p $OUTPUT_DIR

# Disable internal parallelism of huggingface's tokenizer since we
# want to retain direct control of parallelism options.
export TOKENIZERS_PARALLELISM=false

set -xv  # print the command so that we can verify setting arguments
         # correctly from the logs

srun torchrun --standalone \
     --nnodes=1 \
     --nproc-per-node=1 \
     finetuning.py $* \
     --output-path $OUTPUT_DIR \
     --num-workers 10
