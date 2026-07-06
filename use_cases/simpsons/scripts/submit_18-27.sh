#!/bin/bash

module load GCCcore/10.3.0 Python/3.9.5 GCC/10.3.0 CUDA/11.8.0 imkl/2022.2.1

source /beegfs/ws/0/s4610340-isml/.dev/.yahor/statistics-in-machine-learning/tools/experiment_runner/.venv/bin/activate

USER=s4610340 HOST=$(hostname) python -m experiment_runner run \
  /beegfs/ws/0/s4610340-isml/.dev/.yahor/statistics-in-machine-learning/use_cases/simpsons/experiments/18-simpsons-opt-tl-vgg16-augment-AdamW.toml \
  --clean_workdir \
  --submit_jobs

USER=s4610340 HOST=$(hostname) python -m experiment_runner run \
    /beegfs/ws/0/s4610340-isml/.dev/.yahor/statistics-in-machine-learning/use_cases/simpsons/experiments/19-simpsons-opt-tl-vgg16-split-AdamW.toml \
    --clean_workdir \
    --submit_jobs

USER=s4610340 HOST=$(hostname) python -m experiment_runner run \
  /beegfs/ws/0/s4610340-isml/.dev/.yahor/statistics-in-machine-learning/use_cases/simpsons/experiments/20-simpsons-opt-split-augment-AdamW.toml \
  --clean_workdir \
  --submit_jobs


USER=s4610340 HOST=$(hostname) python -m experiment_runner run \
  /beegfs/ws/0/s4610340-isml/.dev/.yahor/statistics-in-machine-learning/use_cases/simpsons/experiments/21-simpsons-opt-weights-AdamW.toml \
  --clean_workdir \
  --submit_jobs

USER=s4610340 HOST=$(hostname) python -m experiment_runner run \
  /beegfs/ws/0/s4610340-isml/.dev/.yahor/statistics-in-machine-learning/use_cases/simpsons/experiments/22-simpsons-opt-split-cv-AdamW.toml \
  --clean_workdir \
  --submit_jobs

USER=s4610340 HOST=$(hostname) python -m experiment_runner run \
  /beegfs/ws/0/s4610340-isml/.dev/.yahor/statistics-in-machine-learning/use_cases/simpsons/experiments/23-simpsons-opt-split-bootstrap-AdamW.toml \
  --clean_workdir \
  --submit_jobs

USER=s4610340 HOST=$(hostname) python -m experiment_runner run \
  /beegfs/ws/0/s4610340-isml/.dev/.yahor/statistics-in-machine-learning/use_cases/simpsons/experiments/24-simpsons-opt-tl-vgg16-weights-AdamW.toml \
  --clean_workdir \
  --submit_jobs

USER=s4610340 HOST=$(hostname) python -m experiment_runner run \
  /beegfs/ws/0/s4610340-isml/.dev/.yahor/statistics-in-machine-learning/use_cases/simpsons/experiments/25-simpsons-opt-tl-vgg16-split-cv-AdamW.toml \
  --clean_workdir \
  --submit_jobs

USER=s4610340 HOST=$(hostname) python -m experiment_runner run \
  /beegfs/ws/0/s4610340-isml/.dev/.yahor/statistics-in-machine-learning/use_cases/simpsons/experiments/26-simpsons-opt-dropout-AdamW.toml \
  --clean_workdir \
  --submit_jobs

USER=s4610340 HOST=$(hostname) python -m experiment_runner run \
  /beegfs/ws/0/s4610340-isml/.dev/.yahor/statistics-in-machine-learning/use_cases/simpsons/experiments/27-simpsons-opt-tl-vgg16-dropout-AdamW.toml \
  --clean_workdir \
  --submit_jobs

