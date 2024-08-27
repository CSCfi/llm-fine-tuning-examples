#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=0:15:00
#SBATCH --gres=gpu:v100:1,nvme:100

module purge
module load pytorch/2.4

# Some environment variables to set up cache directories
# if [ ! -z "$LOCAL_SCRATCH" ]; then
#     SCRATCH="$LOCAL_SCRATCH"
# else
#     SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}/${USER}"
# fi

# export TORCH_HOME=$SCRATCH/torch-cache
# export HF_HOME=$SCRATCH/hf-home
# mkdir -p $TORCH_HOME $HF_HOME

# Disable internal parallelism of huggingface's tokenizer since we
# want to retain direct control of parallelism options.
export TOKENIZERS_PARALLELISM=false

set -xv # print the command so that we can verify setting arguments correctly from the logs

srun python3 inference-demo.py $*
