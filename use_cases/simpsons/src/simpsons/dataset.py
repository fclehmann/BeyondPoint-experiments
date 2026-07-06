import functools
import os
import typing

import numpy as np
import pandas as pd
import torch

import isml_train.dataset
import torchvision.datasets
import torchvision.datasets.folder

import multiprocessing.pool

from isml_train.dataset import _KT


class SimpsonsDataset(torchvision.datasets.ImageFolder, isml_train.dataset.IndexableClassificationDataset[int]):

    @property
    def index(self) -> _KT:
        return list(range(len(self)))

    @property
    def class_indexes(self) -> typing.Iterable[int]:
        return self.targets

    @typing.overload
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        ...

    @typing.overload
    def __getitem__(self, index: slice) -> list[tuple[torch.Tensor, int]]:
        ...

    @typing.overload
    def __getitem__(self, index: list[int]) -> list[tuple[torch.Tensor, int]]:
        ...

    def __getitem__(self, index: typing.Union[int, slice, list]):
        if isinstance(index, int) or np.issubdtype(index, np.integer):
            return super().__getitem__(index)
        elif isinstance(index, slice):
            all_indices = list(range(len(self.samples)))
            return [super().__getitem__(idx) for idx in all_indices[index]]
        elif isinstance(index, list):
            return [super(SimpsonsDataset, self).__getitem__(idx) for idx in index]
        else:
            raise NotImplementedError(f"Index {index} is not supported.")


class SimpsonsDatasetInMemory(SimpsonsDataset):
    _processed_samples: list[tuple[torch.Tensor, int]]

    def __init__(self, processed_samples: list[tuple[torch.Tensor, int]], *args, **kwargs):
        super(SimpsonsDatasetInMemory, self).__init__(*args, **kwargs)
        self._processed_samples = processed_samples

    def __getitem__(self, index: typing.Union[int, slice, list]):
        if isinstance(index, int) or isinstance(index, slice) or isinstance(index, np.int64):
            return self._processed_samples[index]
        elif isinstance(index, list):
            return [self._processed_samples[idx] for idx in index]
        else:
            raise NotImplementedError(f"Index {index} is not supported. Type: {type(index)}")


class SimpsonsDatasetInMemoryRuntimeTransforms(SimpsonsDatasetInMemory):
    _runtime_transform: typing.Callable

    def __init__(self, runtime_transform: typing.Callable, *args, **kwargs):
        super(SimpsonsDatasetInMemoryRuntimeTransforms, self).__init__(*args, **kwargs)
        self._runtime_transforms = runtime_transform

    def __getitem__(self, index: typing.Union[int, slice, list]):
        item = super().__getitem__(index)

        if isinstance(item, list):
            return [(self._runtime_transform(tensor), label) for tensor, label in item]

        elif isinstance(item, tuple):
            return self._runtime_transform(item[0]), item[1]


def __read_and_transform(path: str, read: typing.Callable, transform: typing.Callable):
    return transform(read(path))


def preload_dataset(dataset: SimpsonsDataset) -> SimpsonsDatasetInMemory:
    paths, labels = zip(*dataset.samples)

    with multiprocessing.pool.Pool(processes=len(os.sched_getaffinity(0))) as p:
        images_loaded = p.map(
            functools.partial(
                __read_and_transform,
                transform=dataset.transform,
                read=torchvision.datasets.folder.default_loader
            ),
            paths
        )
    samples_loaded = list(zip(images_loaded, labels))
    return SimpsonsDatasetInMemory(
        processed_samples=samples_loaded,
        root=dataset.root,
        transform=dataset.transform,
        target_transform=dataset.target_transform
    )
