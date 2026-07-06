.PHONY: build_image

GIT_TAG := $(shell git rev-parse --short HEAD)
IMAGE_TAG := gcr.hrz.tu-chemnitz.de/clehm--tu-dresden.de/statistics-in-machine-learning/superconduct_tc_reg:$(GIT_TAG)

build_image:
	cd ../../; \
	DOCKER_BUILDKIT=1 docker build --build-arg CUDA_VERSION=11.8.0 --build-arg USE_CASE=superconduct_tc_reg -f ./docker-files/Dockerfile_use-case -t $(IMAGE_TAG) .

push_image:
	docker push $(IMAGE_TAG)
	echo "Image pushed."
	echo $(IMAGE_TAG)
