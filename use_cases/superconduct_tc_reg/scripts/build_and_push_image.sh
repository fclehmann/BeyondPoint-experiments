#!/bin/bash

DOCKER_BUILDKIT=1 docker build --build-arg CUDA_VERSION=11.8.0 --build-arg USE_CASE=superconduct_tc_reg --secret id=gitlab,src=$HOME/.config/pypoetry/auth.toml -f ./docker-files/Dockerfile_use-case -t gcr.hrz.tu-chemnitz.de/clehm--tu-dresden.de/statistics-in-machine-learning/superconduct_tc_reg:$(git rev-parse --short HEAD) .
docker push gcr.hrz.tu-chemnitz.de/clehm--tu-dresden.de/statistics-in-machine-learning/superconduct_tc_reg:$(git rev-parse --short HEAD)

echo gcr.hrz.tu-chemnitz.de/clehm--tu-dresden.de/statistics-in-machine-learning/superconduct_tc_reg:$(git rev-parse --short HEAD)