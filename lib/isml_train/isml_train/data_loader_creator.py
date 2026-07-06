import abc
import logging
import os
import typing

import torch.utils.data
import numpy as np
import random

_logger = logging.getLogger(__name__)


def seed_worker(worker_id):

    initial_seed = torch.initial_seed()

    _logger.debug(f"Torch initial seed of worker {worker_id}: {initial_seed}")

    worker_seed = torch.initial_seed() % 2**32

    _logger.info(f"Setting non-torch seeds of worker {worker_id} to: {worker_seed}")
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class SizedDataset(torch.utils.data.Dataset):
    def __len__(self):
        raise NotImplementedError()


class DataLoaderCreator:
    # https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    # https://pytorch.org/docs/stable/data.html#randomness-in-multi-process-data-loading

    _data_loader_kwargs: dict[str, typing.Any]
    _augmentation_seed: typing.Optional[int]

    def __init__(
        self,
        augmentation_seed: typing.Optional[int] = None,
        data_shuffle_seed: typing.Optional[int] = None,
    ):
        self._augmentation_seed = augmentation_seed
        self._data_shuffle_seed = data_shuffle_seed

    def __call__(
        self, 
        dataset: SizedDataset,
        shuffle: bool, 
        **data_loader_kwargs,
    ) -> torch.utils.data.DataLoader:
        """
        1. Creates torch random numbers generator
        2. Sets seed of this generator to:
            a. augmentation seed if it is set
            b. initial seed of current default torch generator (dependent config.seed if it was set) + worker number if
               augmentation seed is not set
        3. The seeds of the workers get generated using this generator.
        4. numpy and python random seeds in each worker are calculated and set in worker_init_fn based on the torch seed
           of the worker

        :param dataset:
        :param index:
        :return:
        """

        g_augmentation = torch.Generator()

        if self._augmentation_seed is not None:
            g_augmentation.manual_seed(self._augmentation_seed)
            _logger.info(f"Augmentation seed set to: {self._augmentation_seed}")
        else:
            seed = torch.initial_seed()
            g_augmentation.manual_seed(seed)
            _logger.warning(
                f"No augmentation seed set! Augmentation seed set to current torch seed: {seed}"
            )

        g_shuffle = torch.Generator()
        if self._data_shuffle_seed is not None:
            g_shuffle.manual_seed(self._data_shuffle_seed)
            _logger.info(f"Data shuffle seed set to: {self._data_shuffle_seed}")
        else:
            seed = torch.initial_seed()
            g_shuffle.manual_seed(seed)
            _logger.warning(
                f"No data shuffle seed set! Data shuffle seed set to current torch seed: {seed}"
            )

        if shuffle:
            base_sampler = torch.utils.data.RandomSampler(
                data_source=dataset,
                replacement=False,
                generator=g_shuffle
            )
        else:
            base_sampler = torch.utils.data.SequentialSampler(
                data_source=dataset,
            )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=base_sampler,
            worker_init_fn=seed_worker,
            generator=g_augmentation,
            **data_loader_kwargs,
        )
