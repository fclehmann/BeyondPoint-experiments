#!/bin/bash

#SBATCH --job-name="isml-notebook"
#SBATCH --output="/beegfs/ws/0/clehm-inferential_statistics_ml/.dev/slurm-%j.out"
#SBATCH --error="/beegfs/ws/0/clehm-inferential_statistics_ml/.dev/slurm-%j.out"
#SBATCH --account="p_scads"
#SBATCH --time=8:00:00
#SBATCH --partition=romeo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5000

module load modenv/hiera GCCcore/10.3.0 Python/3.9.5 GCC/10.3.0

source /beegfs/ws/0/clehm-inferential_statistics_ml/.dev/.yahor/statistics-in-machine-learning/aggregated_use_cases_analysis/py/.venv/bin/activate

export XDG_RUNTIME_DIR=""

cd /beegfs/ws/0/clehm-inferential_statistics_ml/.dev/.yahor/statistics-in-machine-learning/aggregated_use_cases_analysis && jupyter notebook --no-browser --port=8888