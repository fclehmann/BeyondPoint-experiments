import functools
import inspect
import typing

import torch.optim

from isml_train.config import TorchOptimizersConfig, TorchLRSchedulerConfig


class TorchOptimizerLoader:
    config: TorchOptimizersConfig

    def __init__(self, config: TorchOptimizersConfig):
        self.config = config

    def get_optimizer_cls(self) -> typing.Type[torch.optim.Optimizer]:
        return getattr(torch.optim, self.config.optimizer)

    def get_optimizer_kwargs(self) -> dict[str, typing.Any]:
        optimizer_param_names = [
            k
            for k in inspect.signature(self.get_optimizer_cls().__init__).parameters.keys() if
            k not in ("self", "params")
        ]
        return {
            name: getattr(self.config, name) for name in optimizer_param_names
        }

    def get_optimizer_config(self) -> dict[str, typing.Any]:
        return {
            "type": self.config.optimizer,
            "params": self.get_optimizer_kwargs()
        }

    def __call__(self, parameters: typing.Iterator[torch.nn.Parameter]) -> torch.optim.Optimizer:
        return self.get_optimizer_cls()(parameters, **self.get_optimizer_kwargs())


class TorchLRSchedulerLoader:
    config: TorchLRSchedulerConfig

    def __init__(self, config: TorchLRSchedulerConfig):
        self.config = config

    def get_lrs_cls(self) -> typing.Optional[typing.Type[torch.optim.lr_scheduler.LRScheduler]]:
        if self.config.lr_scheduler is not None:
            # noinspection PyTypeChecker
            return getattr(torch.optim.lr_scheduler, self.config.lr_scheduler)

    def get_lrs_kwargs(self):
        if self.config.lr_scheduler is not None:
            lrs_param_names = [
                k
                for k in inspect.signature(self.get_lrs_cls().__init__).parameters.keys() if
                k not in ("self", "last_epoch", "optimizer", "verbose")
            ]
            return {
                name: getattr(self.config, f"lrs_{name}") for name in lrs_param_names
            }

    def get_lrs_config(self):
        return {
            "type": self.config.lr_scheduler,
            "params": self.get_lrs_kwargs()
        }

    def __call__(self, optimizer: torch.optim.Optimizer) -> typing.Optional[torch.optim.lr_scheduler.LRScheduler]:
        if self.config.lr_scheduler is not None:
            return self.get_lrs_cls()(optimizer, **self.get_lrs_kwargs())
