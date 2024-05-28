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
#SBATCH -t 00:30:00         # Run time (hh:mm:ss) - 30 min
#SBATCH --account bsc70
#SBATCH --qos=acc_bsccs

CURRENT_DIR=$(pwd)
echo "Current directory: '$CURRENT_DIR'"

module load singularity
singularity exec -B /gpfs/projects/bsc70 \
                 --nv /gpfs/scratch/bsc70/hpai/storage/projects/heka/singularity/space_v1.sif \
   bash -c 'export PROJECT_ROOT='$CURRENT_DIR' && python src/eval.py'
