#!/bin/bash
#------------------------------------------------------
# Example SLURM job script with SBATCH requesting GPUs
#------------------------------------------------------
#SBATCH -J space-train      # Job name
#SBATCH -o slurm_output/out.txt       # Name of stdout output file(%j expands to jobId)
#SBATCH -e slurm_output/err.txt       # Name of stderr output file(%j expands to jobId)
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -c 32               # Cores per task requested
#SBATCH -t 20:00:00         # Run time (hh:mm:ss) - 30 min
#SBATCH --account bsc70
#SBATCH --qos=acc_bsccs

CURRENT_DIR=$(pwd)
echo "Current directory: '$CURRENT_DIR'"

COMMIT_HASH=$(basename "$CURRENT_DIR")

# Update the commit_tag in config.yaml using sed
sed -i "s/^commit_tag: .*/commit_tag: \"$COMMIT_HASH\"/" configs/train.yaml

module load singularity
singularity exec -B /gpfs/scratch/bsc70/hpai/storage/projects \
                 --nv /gpfs/scratch/bsc70/hpai/storage/projects/space/singularity/space_v33.sif \
   bash -c 'export HYDRA_FULL_ERROR=1 && export PROJECT_ROOT='$CURRENT_DIR' && python src/train.py'
