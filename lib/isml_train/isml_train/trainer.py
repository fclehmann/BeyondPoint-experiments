import abc
import typing
import torch.nn
import torch.utils.data

import isml_train.structures
from isml_train.structures import TrainMetrics, TrainMetricRecord


class ModelTrainProtocol(typing.Protocol):
    """
    Defines protocol which must be fulfilled by object which trains coin model.
    """

    model: torch.nn.Module

    def train(
            self,
            train_data_loader: torch.utils.data.DataLoader,
            test_data_loader: torch.utils.data.DataLoader, ) -> TrainMetrics: ...


_O = typing.TypeVar("_O", bound=torch.optim.Optimizer)
_N = typing.TypeVar("_N", bound=torch.nn.Module)


class AbstractClassifierModelTrainer(typing.Generic[_O, _N], abc.ABC):
    _nn_model: _N
    _optimizer: _O
    _loss_f: torch.nn.Module
    _class_to_indexes: typing.Dict[int, int]
    _n_epochs: int
    _device: torch.device
    _show_step: int
    _lr_scheduler: typing.Optional[torch.optim.lr_scheduler._LRScheduler] = None

    def __init__(
            self,
            base_nn_model: _N,
            optimizer: _O,
            loss_function: torch.nn.Module,
            class_to_indexes: typing.Dict[int, int],
            n_epochs: int,
            device: torch.device,
            show_step: int,
            lr_scheduler: typing.Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        self._nn_model = base_nn_model
        self._optimizer = optimizer
        self._loss_f = loss_function
        self._class_to_indexes = class_to_indexes
        self._n_epochs = n_epochs
        self._device = device
        self._show_step = show_step
        self._lr_scheduler = lr_scheduler

    @abc.abstractmethod
    def train_batch(self, batch: tuple[torch.Tensor, torch.Tensor], batch_id: int) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def train_epoch(self, train_data_loader: torch.utils.data.DataLoader) -> float: ...

    @abc.abstractmethod
    def evaluate(
            self,
            data_loader: torch.utils.data.DataLoader
    ) -> isml_train.structures.EvalResult:
        ...

    @abc.abstractmethod
    def train(
            self,
            train_data_loader: torch.utils.data.DataLoader,
            test_data_loader: torch.utils.data.DataLoader,
    ) -> TrainMetrics:
        ...

    @property
    def model(self) -> _N:
        return self._nn_model

    @property
    def optimizer(self) -> _O:
        return self._optimizer

    @property
    def loss(self) -> torch.nn.Module:
        return self._loss_f

    def __call__(self, *args, **kwargs) -> TrainMetrics:
        return self.train(*args, **kwargs)
