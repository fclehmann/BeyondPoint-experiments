import enum

import spock
import typing


class LRScheduler(enum.Enum):
    exponentialLR = "ExponentialLR"
    stepLR = "StepLR"
    reduceLROnPlateau = "ReduceLROnPlateau"


@spock.spock
class TorchOptimizersConfig:
    # Optimizer settings
    optimizer: str = "AdamW"
    lr: float = 0.02
    lr_decay: float = 0.0
    weight_decay: float = 0.0
    initial_accumulator_value: float = 0.0
    betas: typing.Tuple[float, float] = (0.9, 0.999)
    etas: typing.Tuple[float, float] = (0.5, 1.2)  # Rprop
    step_sizes: typing.Tuple[int, int] = (int(1e-5), 50)
    # step_sizes: float = 0.9
    eps: float = 1e-8
    amsgrad: bool = False
    max_iter: int = 20
    max_eval: typing.Optional[int] = None
    tolerance_grad: float = 1e-07
    tolerance_change: float = 1e-09
    history_size: int = 100
    alpha: float = 0.99
    momentum: float = 0.0
    dampening: float = 0.0
    lambd: float = 0.1
    rho: float = 0.9
    t0: float = 0.1
    nesterov: bool = False
    centered: bool = False
    foreach: typing.Optional[bool] = None
    maximize: bool = False
    differentiable: bool = False
    capturable: bool = False
    fused: typing.Optional[bool] = None


@spock.spock
class TorchLRSchedulerConfig:
    lr_scheduler: typing.Optional[LRScheduler] = None
    lrs_gamma: typing.Optional[float] = None
    lrs_step_size: typing.Optional[int] = None
    lrs_mode: str = "min"
    lrs_factor: float = 0.1
    lrs_patience: int = 10
    lrs_threshold: float = 1e-4
    lrs_min_lr: float = 0.0
    lrs_cooldown: int = 0
    lrs_threshold_mode: str = "rel"
    lrs_eps: float = 1e-8
    lrs_verbose: bool = False


@spock.spock
class DataSaverConfig:
    save_results: bool = False
    save_model: bool = False
    out_path: str
    out_file_string: str = "result"
