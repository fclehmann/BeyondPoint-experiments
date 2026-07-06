#!/bin/bash

#SBATCH --job-name="isml-cifar10-process-opt"
#SBATCH --output="/data/horse/ws/s4610340-isml/dev/logs/slurm-%j.out"
#SBATCH --error="/data/horse/ws/s4610340-isml/dev/logs/slurm-%j.out"
#SBATCH --account="p_scads"
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5000

module load release/23.04 GCCcore/10.3.0 Python/3.9.5 GCC/10.3.0 CUDA/11.8.0 imkl/2022.2.1

source /data/horse/ws/s4610340-isml/dev/statistics-in-machine-learning/use_cases/cifar10_clf/.venv/bin/activate

python -m ipykernel install --user --name cifar10_clf-interactive

PATH_WS=/data/horse/ws/s4610340-isml
NB_NAME=3-cifar10_clf-opt-process+report.ipynb
PATH_NB=$PATH_WS/dev/statistics-in-machine-learning/use_cases/cifar10_clf/notebooks/$NB_NAME

# 3
papermill \
  $PATH_NB \
  $PATH_WS/cifar10_clf/experiments/003_cifar10_clf_opt_split_bootstrap/$NB_NAME \
  -p PATH_EXPERIMENT $PATH_WS/cifar10_clf/experiments/003_cifar10_clf_opt_split_bootstrap \
  -p SEED_ATTR train_val_split_algorithm_seed \
  --kernel cifar10_clf-interactive

# 4
papermill \
  $PATH_NB \
  $PATH_WS/cifar10_clf/experiments/004_cifar10_clf_opt_augment/$NB_NAME \
  -p PATH_EXPERIMENT $PATH_WS/cifar10_clf/experiments/004_cifar10_clf_opt_augment \
  -p SEED_ATTR augmentation_seed \
  --kernel cifar10_clf-interactive

# 5
papermill \
  $PATH_NB \
  $PATH_WS/cifar10_clf/experiments/005_cifar10_clf_opt_dropout/$NB_NAME \
  -p PATH_EXPERIMENT $PATH_WS/cifar10_clf/experiments/005_cifar10_clf_opt_dropout \
  -p SEED_ATTR dropout_seed \
  --kernel cifar10_clf-interactive