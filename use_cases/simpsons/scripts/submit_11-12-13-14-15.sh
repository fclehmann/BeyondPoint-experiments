#!/bin/bash

#SBATCH --job-name="isml-submit-11-12-13-14-15"
#SBATCH --output="/beegfs/ws/0/s4610340-isml/.dev/isml-submit-11-12-13-14-15-%j.out"
#SBATCH --error="/beegfs/ws/0/s4610340-isml/.dev/isml-submit-11-12-13-14-15-%j.out"
#SBATCH --account="p_scads"
#SBATCH --time=1:00:00
#SBATCH --partition=alpha
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3000

module load modenv/hiera GCCcore/10.3.0 Python/3.9.5 GCC/10.3.0  CUDA/11.3.1  OpenMPI/4.1.1

source /beegfs/ws/0/s4610340-isml/.venvs/venv-exp_runner-latest/bin/activate

experiment-runner run \
  /beegfs/ws/0/s4610340-isml/.dev/.yahor/statistics-in-machine-learning/use_cases/simpsons/experiments/11-simpsons-hopt-tpe-AdamW-seeds.toml  \
  --clean_workdir \
  --submit_jobs

experiment-runner run \
  /beegfs/ws/0/s4610340-isml/.dev/.yahor/statistics-in-machine-learning/use_cases/simpsons/experiments/12-simpsons-hopt-tpe-Adamax-seeds.toml  \
  --clean_workdir \
  --submit_jobs

experiment-runner run \
  /beegfs/ws/0/s4610340-isml/.dev/.yahor/statistics-in-machine-learning/use_cases/simpsons/experiments/13-simpsons-hopt-tpe-Adam-seeds.toml  \
  --clean_workdir \
  --submit_jobs

experiment-runner run \
  /beegfs/ws/0/s4610340-isml/.dev/.yahor/statistics-in-machine-learning/use_cases/simpsons/experiments/14-simpsons-hopt-tpe-Adagrad-seeds.toml  \
  --clean_workdir \
  --submit_jobs

experiment-runner run \
  /beegfs/ws/0/s4610340-isml/.dev/.yahor/statistics-in-machine-learning/use_cases/simpsons/experiments/15-simpsons-hopt-tpe-RMSprop-seeds.toml  \
  --clean_workdir \
  --submit_jobs