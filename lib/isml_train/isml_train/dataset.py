import abc
import typing

import torch.utils.data
import pandas as pd

_KT = typing.TypeVar("_KT")


class IndexableClassificationDataset(torch.utils.data.Dataset, abc.ABC, typing.Generic[_KT]):

    @property
    @abc.abstractmethod
    def index(self) -> list[_KT]:
        """
        Must return index corresponding to (current) torch.utils.data.Dataset index
        :return:
        """
        ...

    @property
    @abc.abstractmethod
    def class_indexes(self) -> list[int]:
        """
        Must return class index values corresponding to index.
        :return:
        """
