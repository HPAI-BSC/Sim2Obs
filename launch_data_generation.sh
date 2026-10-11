#!/bin/bash
#------------------------------------------------------
# Example SLURM job script with SBATCH
#------------------------------------------------------
#SBATCH -J data-generation      # Job name
#SBATCH -o slurm_output/out.txt       # Name of stdout output file(%j expands to jobId)
#SBATCH -e slurm_output/err.txt       # Name of stderr output file(%j expands to jobId)
#SBATCH -n 1
#SBATCH -c 64               # Cores per task requested — MN5 has no --mem flag; memory is
                             # auto-set per core (GPP: 2G/core), so 64 cores -> 128G total.
                             # Obelisk halos can have tens of millions of star particles.

#SBATCH -t 02:00:00         # Run time (hh:mm:ss)
#SBATCH --account bsc70
#SBATCH --qos=gp_debug

CURRENT_DIR=$(pwd)
echo "Current directory: '$CURRENT_DIR'"

COMMIT_HASH=$(basename "$CURRENT_DIR")

module load singularity
singularity exec -B /gpfs/scratch/bsc70/hpai/storage/projects \
                 --nv /gpfs/scratch/bsc70/hpai/storage/projects/space/singularity/space_data_generation_v2.sif \
   bash -c 'export PROJECT_ROOT='$CURRENT_DIR' && export COMMIT_HASH='$COMMIT_HASH' && export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT && cd $PROJECT_ROOT && python src/generate_dataset.py'
