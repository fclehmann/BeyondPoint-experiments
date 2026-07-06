#!/bin/bash

DOCKER_BUILDKIT=1 docker build --build-arg USE_CASE=simpsons --secret id=gitlab,src=$HOME/.config/pypoetry/auth.toml -f ./docker-files/Dockerfile_use-case -t gcr.hrz.tu-chemnitz.de/clehm--tu-dresden.de/statistics-in-machine-learning/simpsons:$(git rev-parse --short HEAD) .
docker push gcr.hrz.tu-chemnitz.de/clehm--tu-dresden.de/statistics-in-machine-learning/simpsons:$(git rev-parse --short HEAD)
