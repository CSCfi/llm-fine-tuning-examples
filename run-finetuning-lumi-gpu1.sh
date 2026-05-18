#!/bin/bash
#SBATCH --account=project_xxxxxxxxx
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=0:15:00
#SBATCH --gpus-per-node=1

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif


# This will store all the Hugging Face cache such as downloaded models
# and datasets in the project's scratch folder
export HF_HOME=/scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf-cache
mkdir -p $HF_HOME

# Path to where the trained model and logging data will go
OUTPUT_DIR=/scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf-data
mkdir -p $OUTPUT_DIR

# Disable internal parallelism of huggingface's tokenizer since we
# want to retain direct control of parallelism options.
export TOKENIZERS_PARALLELISM=false

# Disable Weights & Biases logging
export WANDB_DISABLED=true

set -xv  # print the command so that we can verify setting arguments correctly from the logs

srun singularity exec "$SIF" torchrun --standalone \
     --nnodes=1 \
     --nproc_per_node=1 \
     finetuning.py $* \
     --output-path $OUTPUT_DIR \
     --num-workers $SLURM_CPUS_PER_TASK
