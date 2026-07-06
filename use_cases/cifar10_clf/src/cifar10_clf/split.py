import abc
import typing


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
            val_data_loader_creator: isml_train.data_loader_creator.DataLoaderCreator,
            **kwargs
    ):
        self.trainer = trainer
        self.train_data_loader_creator = train_data_loader_creator
        self.test_data_loader_creator = test_data_loader_creator
        self.train_test_split_algorithm_seed = kwargs.get("train_test_split_algorithm_seed")
