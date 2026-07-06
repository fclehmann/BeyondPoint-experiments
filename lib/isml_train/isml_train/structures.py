import dataclasses
import typing
import torch

_O = typing.TypeVar("_O", bound=torch.optim.Optimizer)
_N = typing.TypeVar("_N", bound=torch.nn.Module)


class EvalResult(typing.TypedDict):
    loss: float
    acc: float
    confusion_matrix: list[int]


class TrainMetricRecord(typing.TypedDict):
    epoch: int
    train: EvalResult
    val: EvalResult


TrainMetrics = list[TrainMetricRecord]


class TrainResultRecord(typing.TypedDict):
    train_indices: list[int]
    test_indices: list[int]
    metrics: TrainMetrics


class TrainResultRecordFold(typing.TypedDict):
    fold: int
    train_indices: list[int]
    test_indices: list[int]
    metrics: TrainMetrics


TR = typing.TypeVar("TR", TrainResultRecord, TrainResultRecordFold)

TrainResults = list[TR]


@dataclasses.dataclass
class ModelSetup(typing.Generic[_N, _O]):
    """
    Defines data class to save "model setup": model, optimizer, loss function.
    """
    model: _N
    optimizer: _O
    criterion: typing.Callable
