#!/bin/bash
#------------------------------------------------------
# Example SLURM job script with SBATCH requesting GPUs
#------------------------------------------------------
#SBATCH -J eval-mn5      # Job name
#SBATCH -o slurm_output/out.txt       # Name of stdout output file(%j expands to jobId)
#SBATCH -e slurm_output/err.txt       # Name of stderr output file(%j expands to jobId)
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -c 32               # Cores per task requested
#SBATCH -t 01:10:00         # Run time (hh:mm:ss) - 30 min
#SBATCH --account bsc70
#SBATCH --qos=acc_bsccs

CURRENT_DIR=$(pwd)
echo "Current directory: '$CURRENT_DIR'"

# Check if we're in a Git repository
if git rev-parse --is-inside-work-tree 2>/dev/null; then
    COMMIT_HASH=$(git rev-parse --short HEAD)
    echo "Commit hash --: '$COMMIT_HASH'"
else
    echo "Not inside a Git repository. Cannot get commit hash."
    exit 1
fi

# Update the commit_tag in config.yaml using sed
sed -i "s/^commit_tag: .*/commit_tag: \"$COMMIT_HASH\"/" configs/train.yaml

module load singularity
singularity exec -B /gpfs/projects/bsc70 \
                 --nv /gpfs/projects/bsc70/space/singularity/space_v3.sif \
   bash -c 'export PROJECT_ROOT='$CURRENT_DIR' && python src/train.py'
