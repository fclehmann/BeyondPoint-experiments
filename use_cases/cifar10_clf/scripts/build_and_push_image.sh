#!/bin/bash

DOCKER_BUILDKIT=1 docker build --build-arg CUDA_VERSION=11.8.0 --build-arg USE_CASE=cifar10_clf --secret id=gitlab,src=$HOME/.config/pypoetry/auth.toml -f ./docker-files/Dockerfile_use-case -t gcr.hrz.tu-chemnitz.de/clehm--tu-dresden.de/statistics-in-machine-learning/cifar10_clf:$(git rev-parse --short HEAD) .
docker push gcr.hrz.tu-chemnitz.de/clehm--tu-dresden.de/statistics-in-machine-learning/cifar10_clf:$(git rev-parse --short HEAD)
