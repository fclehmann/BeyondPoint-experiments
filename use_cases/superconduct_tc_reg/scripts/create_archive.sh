#!/bin/bash

#SBATCH --job-name="isml-archive-stc"
#SBATCH --output="/data/horse/ws/s4610340-isml/dev/logs/slurm-archive-stc-%j.out"
#SBATCH --error="/data/horse/ws/s4610340-isml/dev/logs/slurm-archive-src-%j.out"
#SBATCH --account="p_scads"
#SBATCH --time=144:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=2000
#SBATCH --nodes=1

module load release/23.04 GCCcore/10.3.0 Python/3.9.5 GCC/10.3.0

source /data/horse/ws/s4610340-isml/dev/statistics-in-machine-learning/tools/experiment_runner/.venv/bin/activate

python -m experiment_runner backup \
 -o /data/horse/ws/s4610340-isml/archive \
 -e "/data/horse/ws/s4610340-isml/superconduct_tc_reg/experiments/*/*.toml" -rp "*.json" -c "gz" \
 --lazy \
 --resolve_path_from_ec \
 -w 6