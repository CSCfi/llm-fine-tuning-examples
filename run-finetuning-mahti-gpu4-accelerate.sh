#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=0:15:00
#SBATCH --gres=gpu:a100:4

module purge
module load pytorch/2.4

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

ACCELERATE_CONFIG=$1  # first argument must be accelerate config to use
if [ ! -f "$ACCELERATE_CONFIG" ]; then
    echo "ERROR: first argument must be the accelerate config to use"
    exit 1
fi
shift  # remove first argument from argument list


set -xv  # print the command so that we can verify setting arguments
         # correctly from the logs

srun accelerate launch \
     --config_file=$ACCELERATE_CONFIG \
     --num_processes=4 \
     --num_machines=1 \
     --machine_rank=0 \
     finetuning.py $* \
     --output-path $OUTPUT_DIR \
     --num-workers 10
