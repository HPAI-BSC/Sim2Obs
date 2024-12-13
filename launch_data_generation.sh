#!/bin/bash
#------------------------------------------------------
# Example SLURM job script with SBATCH
#------------------------------------------------------
#SBATCH -J data-generation      # Job name
#SBATCH -o slurm_output/out.txt       # Name of stdout output file(%j expands to jobId)
#SBATCH -e slurm_output/err.txt       # Name of stderr output file(%j expands to jobId)
#SBATCH -n 1
#SBATCH -c 32               # Cores per task requested
#SBATCH -t 12:00:00         # Run time (hh:mm:ss)
#SBATCH --account bsc70
#SBATCH --qos=acc_bsccs

CURRENT_DIR=$(pwd)
echo "Current directory: '$CURRENT_DIR'"

COMMIT_HASH=$(basename "$CURRENT_DIR")

module load singularity
singularity exec -B /gpfs/projects/bsc70 \
                 --nv /gpfs/projects/bsc70/space/singularity/space_data_generation.sif \
   bash -c 'export PROJECT_ROOT='$CURRENT_DIR' && export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT && cd $PROJECT_ROOT && python src/generate_dataset.py'
