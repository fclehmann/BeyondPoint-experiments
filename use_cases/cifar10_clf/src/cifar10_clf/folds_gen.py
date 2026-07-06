from abc import ABC, abstractmethod
import logging
from re import split
from typing import Any, Iterator, Optional, Sequence

import sklearn.model_selection
import sklearn.utils
import numpy.random

_logger = logging.getLogger(__name__)

class FoldsGeneratorAlgorithm(ABC):

    def __init__(self, split_algorithm_seed: Optional[int] = None) -> None:
        self._split_algorithm_seed = split_algorithm_seed

    @abstractmethod
    def __call__(
        self,
        index: Sequence[int],
        targets: Optional[Sequence[int]] = None,
    ) -> Iterator[tuple[Sequence[int], Sequence[int]]]: ...

    @property
    @abstractmethod
    def n_folds(self) -> int: ...


class TrainValBootstapSamplesGeneratorSingleFold(FoldsGeneratorAlgorithm):
    def __init__(
        self,
        n_samples_train: int,
        n_samples_val: int,
        with_replacement: bool,
        stratified: bool,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._n_samples_train = n_samples_train
        self._n_samples_val = n_samples_val
        self._with_replacement = with_replacement
        self._stratified = stratified

    @property
    def n_folds(self):
        return 1

    def __call__(
        self, elements: Sequence[int], targets: Optional[Sequence[int]] = None
    ) -> Iterator[tuple[Sequence[int], Sequence[int]]]:
        if self._stratified:
            assert targets is not None

        random_state = numpy.random.RandomState(seed=self._split_algorithm_seed)

        index = list(range(len(elements)))

        idx_val: list[int] = sklearn.utils.resample(
            index,
            replace=self._with_replacement,
            n_samples=self._n_samples_val,
            random_state=random_state,
            stratify=targets,
        )  # type: ignore

        idx_val_set = set(idx_val)
        idx_non_val = [i for i in index if i not in idx_val_set]

        targets_non_val = None
        if self._stratified:
            targets_non_val = [targets[idx] for idx in idx_non_val]  # type: ignore

        idx_train: list[int] = sklearn.utils.resample(
            idx_non_val,
            replace=self._with_replacement,
            n_samples=self._n_samples_train,
            random_state=random_state,
            stratify=targets_non_val,
        )  # type: ignore

        yield [elements[i] for i in idx_train], [elements[i] for i in idx_val]


class CVFoldsGeneratorAlgorithm(FoldsGeneratorAlgorithm):
    def __init__(self, n_folds: int, stratified: bool, **kwargs):
        super().__init__(**kwargs)
        self._n_folds = n_folds
        self._stratified = stratified

    def __call__(
        self, index: Sequence[int], targets: Optional[Sequence[int]] = None
    ) -> Iterator[tuple[Sequence[int], Sequence[int]]]:
        split_cls = sklearn.model_selection.KFold
        stratify_targets = None

        if self._stratified:
            assert targets is not None
            stratify_targets = targets
            split_cls = sklearn.model_selection.StratifiedKFold

        kf = split_cls(
            n_splits=self._n_folds,
            random_state=self._split_algorithm_seed,
            shuffle=True,
        )

        for fold_i, (train_indices, val_indices) in enumerate(
            kf.split(index, y=stratify_targets)
        ):
            yield train_indices, val_indices
