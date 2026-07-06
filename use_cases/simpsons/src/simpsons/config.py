import enum
import typing

import spock

import isml_train.config


class TrainTestSplitAlgorithm(enum.Enum):
    bootstrap = "bootstrap"
    cross_validation = "cross-validation"
    split = "split"


class TargetMetric(enum.Enum):
    loss = "loss"
    error_rate = "error_rate"


class SimpsonsModel(enum.Enum):
    simpsons_net_conv = "SimpsonsNetConv"
    vgg16 = "vgg16"


@spock.spock
class ExperimentConfig(
    isml_train.config.TorchOptimizersConfig,
    isml_train.config.TorchLRSchedulerConfig,
    isml_train.config.DataSaverConfig
):
    training_data_root: str
    short_name: typing.Optional[str]
    seed: typing.Optional[int]
    slurm_tracking: bool = False
    show_step: int = 1
    num_epochs: int = 100
    batch_size: int = 4

    # Train test split settings
    train_test_split_algorithm: TrainTestSplitAlgorithm = "split"
    train_test_split_algorithm_seed: typing.Optional[int]
    test_size: float = 0.25
    n_folds: int = 5
    stratified: bool = True
    val_size: float = 0.2
    stratified_test_split: bool = True

    target_metric: TargetMetric = "error_rate"

    # Data transform settings
    apply_resize: bool = False
    augmentation_enabled: bool = True
    # Data augmentation seed
    # If None - defaults to general seed
    augmentation_seed: typing.Optional[int]

    # Model settings
    model: SimpsonsModel = "SimpsonsNetConv"

    # Transfer learning settings
    # vgg16
    feature_layers_freeze: int = 10

    # Weights init settings
    # If none - governed by general seed
    weights_init_seed: typing.Optional[int]

    # Dropout settings
    # If none - governed by general seed
    dropout_seed: typing.Optional[int]
