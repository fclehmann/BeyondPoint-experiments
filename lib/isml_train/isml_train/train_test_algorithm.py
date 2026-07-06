import abc
import logging
import typing

import numpy as np
import pandas as pd
import sklearn.model_selection
import torch.utils.data

import isml_train.structures
import isml_train.trainer
import isml_train.utils
import isml_train.dataset
import isml_train.data_loader_creator

TR = typing.TypeVar("TR", isml_train.structures.TrainResultRecord, isml_train.structures.TrainResultRecordFold)

_logger = logging.getLogger(__name__)


class AbstractTrainTestAlgo(abc.ABC, typing.Generic[TR]):
    """
    Defines abstract class / interface for train-test-split algorithm

    TrainTestAlgo <=> train - test - split - algorithm
    """

    trainer: isml_train.trainer.ModelTrainProtocol
    train_data_loader_creator: isml_train.data_loader_creator.DataLoaderCreator
    test_data_loader_creator: isml_train.data_loader_creator.DataLoaderCreator
    train_test_split_algorithm_seed: typing.Optional[int]

    @classmethod
    @abc.abstractmethod
    def algo_name(cls) -> str:
        """
        Must return name of train-test-split algorithm
        :return: algorithm name
        """
        ...

    def __init__(
            self,
            trainer: isml_train.trainer.ModelTrainProtocol,
            train_data_loader_creator: isml_train.data_loader_creator.DataLoaderCreator,
            test_data_loader_creator: isml_train.data_loader_creator.DataLoaderCreator,
            **kwargs
    ):
        self.trainer = trainer
        self.train_data_loader_creator = train_data_loader_creator
        self.test_data_loader_creator = test_data_loader_creator
        self.train_test_split_algorithm_seed = kwargs.get("train_test_split_algorithm_seed")

    @abc.abstractmethod
    def __call__(
            self,
            dataset_train: torch.utils.data.Subset,
            dataset_test: torch.utils.data.Subset,
            targets: typing.Sequence[int]
    ) -> isml_train.structures.TrainResults[TR]:
        """
        Must train the coin model using 'self.trainer' and concrete train-test-split algorithm.
        :param dataframe: coin dataset
        :return: metrics returned by trainer.
        """
        ...


class BootstrapTrainTestSplit(AbstractTrainTestAlgo[isml_train.structures.TrainResultRecord]):
    """
    Represents train-test split using bootstrapping.

    TODO: stratified
    """

    @classmethod
    def algo_name(cls):
        return "bootstrap"

    @classmethod
    def calculate_result_target_metric(cls, train_results: isml_train.structures.TrainResults[
        isml_train.structures.TrainResultRecord
    ]) -> float:
        """
        Gets accuracy of last epoch
        :param train_results:
        :return:
        """
        return train_results[0]["metrics"][-1]["loss"]

    def __call__(
            self,
            dataset_train: torch.utils.data.Subset,
            dataset_test: torch.utils.data.Subset,
            targets: typing.Sequence[int]
    ) -> isml_train.structures.TrainResults[
        isml_train.structures.TrainResultRecord
    ]:
        """
        Splits dataframe on train and test using coins.utils.train_test_split_bootstrap
        with with_replace=True and n_bootstrap_samples = length of dataframe.
        This way ~37% of rows are not sampled and go to test, other rows go to train (with replacement).
        Trains and validates the model using this split.

        :param dataset_train: dataset with train transformations applied
        :param dataset_test: dataset with test transformations applied
        :return: metrics returned by trainer
        """
        index = dataset_train.indices
        df_index = pd.DataFrame(data=index, index=index, columns=["idx"])

        df_index_train, df_index_test = isml_train.utils.train_test_split_bootstrap(
            df=df_index,
            n_bootstrap_samples=len(df_index),
            with_replace=True,
            random_state=self.train_test_split_algorithm_seed
        )

        subset_train = torch.utils.data.Subset(dataset_train.dataset, df_index_train.index)
        subset_val = torch.utils.data.Subset(dataset_test.dataset, df_index_test.index)

        train_metrics = self.trainer.train(
            train_data_loader=self.train_data_loader_creator(subset_train),
            test_data_loader=self.test_data_loader_creator(subset_val)
        )

        return [{
            "train_indices": list(subset_train.indices),
            "test_indices": list(subset_val.indices),
            "metrics": train_metrics
        }]


class SimpleTrainTestSplit(AbstractTrainTestAlgo[isml_train.structures.TrainResultRecord]):
    """
    Represents simple split on train and test dataframe.
    """

    @classmethod
    def algo_name(cls):
        return "split"

    def __init__(
            self,
            val_size: float,
            stratified: bool,
            trainer: isml_train.trainer.ModelTrainProtocol,
            **kwargs):
        """
        :param test_size: test size between 0 and 1
        :param trainer: model trainer
        :param kwargs:
        """
        super().__init__(trainer, **kwargs)
        self.__val_size = val_size
        self.__stratified = stratified

    def __call__(
            self,
            dataset_train: torch.utils.data.Subset,
            dataset_test: torch.utils.data.Subset,
            targets: typing.Sequence[int]
    ) -> isml_train.structures.TrainResults[isml_train.structures.TrainResultRecord]:
        """
        Splits dataframe using sklearn.model_selection.train_test_split
        Trains and validates the model using this split.
        :param dataset_train: dataset with train transformations applied
        :param dataset_test: dataset with test transformations applied
        :return: metrics returned by trainer
        """

        stratify = None
        if self.__stratified:
            stratify = targets

        train_indices, val_indices = sklearn.model_selection.train_test_split(
            dataset_train.indices,
            test_size=self.__val_size,
            stratify=stratify,
            shuffle=True,
            random_state=self.train_test_split_algorithm_seed
        )

        subset_train = torch.utils.data.Subset(dataset_train.dataset, train_indices)
        subset_val = torch.utils.data.Subset(dataset_test.dataset, val_indices)

        train_metrics = self.trainer.train(
            train_data_loader=self.train_data_loader_creator(subset_train),
            test_data_loader=self.test_data_loader_creator(subset_val)
        )

        return [{
            "train_indices": list(train_indices),
            "test_indices": list(val_indices),
            "metrics": train_metrics
        }]


class CVTrainTestSplit(AbstractTrainTestAlgo[isml_train.structures.TrainResultRecordFold]):
    """
    Represents train-test split using K-Fold cross validation
    """

    @classmethod
    def algo_name(cls):
        return "cross-validation"

    def __init__(
            self,
            n_folds: int,
            stratified: bool,
            reset_trainer: typing.Callable[[], isml_train.trainer.ModelTrainProtocol],
            trainer: isml_train.trainer.ModelTrainProtocol,
            **kwargs
    ):
        """
        :param n_folds: n folds
        :param trainer: model trainer
        :param reset_trainer: Reset trainer
        :param kwargs:
        """
        super().__init__(trainer, **kwargs)
        self.__n_folds = n_folds
        self.__stratified = stratified
        self.__reset_trainer = reset_trainer

    def __call__(
            self,
            dataset_train: torch.utils.data.Subset,
            dataset_test: torch.utils.data.Subset,
            targets: typing.Sequence[int]
    ) -> isml_train.structures.TrainResults[isml_train.structures.TrainResultRecordFold]:
        """
        Splits dataframe using sklearn.model_selection.KFold without shuffling.
        Trains and validates the model using this split.
        :param dataset_train: dataset with train transformations applied
        :param dataset_test: dataset with test transformations applied
        :return: metrics returned by trainer
        """
        train_metrics = []

        split_cls = sklearn.model_selection.KFold
        stratify_targets = None
        if self.__stratified:
            stratify_targets = targets
            split_cls = sklearn.model_selection.StratifiedKFold

        # TODO: KFold does shuffle once or before each split?
        kf = split_cls(
            n_splits=self.__n_folds,
            random_state=self.train_test_split_algorithm_seed,
            shuffle=True
        )
        train_indices: np.ndarray
        val_indices: np.ndarray
        for fold_i, (train_indices, val_indices) in enumerate(kf.split(dataset_train.indices, y=stratify_targets)):
            _logger.info(f"Processing fold {fold_i}")
            if fold_i != 0:
                _logger.info(f"Reinitializing model weights ...")
                self.trainer = self.__reset_trainer()
            _logger.info(
                f'number of elements train: {len(train_indices)}, val: {len(val_indices)}')

            subset_train = torch.utils.data.Subset(dataset_train.dataset, train_indices)
            subset_val = torch.utils.data.Subset(dataset_test.dataset, val_indices)

            acc_rates = self.trainer.train(
                train_data_loader=self.train_data_loader_creator(subset_train),
                test_data_loader=self.test_data_loader_creator(subset_val)
            )

            train_metrics.append({
                "fold": fold_i,
                "train_indices": train_indices.astype(int).tolist(),
                "test_indices": val_indices.astype(int).tolist(),
                "metrics": acc_rates
            })
        return train_metrics
