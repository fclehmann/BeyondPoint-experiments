import abc
import typing

import numpy as np

from isml_train.structures import TrainResults, TrainResultRecordFold, TrainResultRecord

TR = typing.TypeVar("TR", TrainResultRecord, TrainResultRecordFold)


class AbstractTargetMetric(abc.ABC):
    subset: typing.Literal["train", "val"]

    def __init__(self, subset: typing.Literal["train", "val"]):
        self.subset = subset

    @classmethod
    @abc.abstractmethod
    def metric_name(cls) -> str:
        ...

    @staticmethod
    def get_record_type(train_results: TrainResults[TR]) -> typing.Type[TR]:
        if len(train_results) > 1 and "fold" in train_results[0]:
            return TrainResultRecordFold
        else:
            return TrainResultRecord

    @typing.overload
    def __call__(self, train_results: TrainResults[TrainResultRecord]):
        ...

    @typing.overload
    def __call__(self, train_results: TrainResults[TrainResultRecordFold]):
        ...

    @abc.abstractmethod
    def __call__(
            self,
            train_results: typing.Union[TrainResults[TrainResultRecord], TrainResults[TrainResultRecordFold]]
    ) -> float:
        """
        Must calculate target metric result value (for omniopt)
        :param train_results: train results
        :return: target metric result value (for omniopt)
        """
        ...


class ErrorRateTargetMetric(AbstractTargetMetric):

    @classmethod
    def metric_name(cls) -> str:
        return "error_rate"

    @typing.overload
    def __call__(self, train_results: TrainResults[TrainResultRecord]):
        ...

    @typing.overload
    def __call__(self, train_results: TrainResults[TrainResultRecordFold]):
        ...

    def __call__(
            self,
            train_results: typing.Union[TrainResults[TrainResultRecord], TrainResults[TrainResultRecordFold]]
    ) -> float:
        if self.get_record_type(train_results) == TrainResultRecordFold:
            return np.mean([1 - train_result["metrics"][-1][self.subset]["acc"] for train_result in train_results])
        elif self.get_record_type(train_results) == TrainResultRecord:
            return 1 - train_results[0]["metrics"][-1][self.subset]["acc"]


class LossTargetMetric(AbstractTargetMetric):

    @classmethod
    def metric_name(cls) -> str:
        return "loss"

    @typing.overload
    def __call__(self, train_results: TrainResults[TrainResultRecord]):
        ...

    @typing.overload
    def __call__(self, train_results: TrainResults[TrainResultRecordFold]):
        ...

    def __call__(
            self,
            train_results: typing.Union[TrainResults[TrainResultRecord], TrainResults[TrainResultRecordFold]]
    ) -> float:
        if self.get_record_type(train_results) == TrainResultRecordFold:
            return np.mean([train_result["metrics"][-1][self.subset]["loss"] for train_result in train_results])
        elif self.get_record_type(train_results) == TrainResultRecord:
            return train_results[0]["metrics"][-1][self.subset]["loss"]
