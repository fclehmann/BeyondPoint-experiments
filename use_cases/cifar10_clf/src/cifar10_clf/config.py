import enum
import typing
from uuid import uuid4

import spock

import isml_train.config


class TrainValSplitAlgorithm(enum.Enum):
    bootstrap = "bootstrap"
    split = "split"


class TestSampleAlgorithm(enum.Enum):
    bootstrap = "bootstrap"
    split = "split"


@spock.spock
class ExperimentConfig(
    isml_train.config.TorchOptimizersConfig,
    isml_train.config.TorchLRSchedulerConfig,
    isml_train.config.DataSaverConfig,
):
    training_data_root: str
    seed: typing.Optional[int]
    slurm_tracking: bool = False

    batch_size: int = 4

    run_uid: typing.Optional[str] = None

    # Settings for sampling test subset
    test_sample_with_replacement: bool = True
    test_sample_stratified: bool = True
    test_sample_seed: typing.Optional[int] = None
    test_sample_size: int = 10

    # Train val split settings
    train_val_split_algorithm: TrainValSplitAlgorithm = "bootstrap"
    train_val_split_algorithm_seed: typing.Optional[int]
    n_folds: int = 5
    stratified: bool = True
    train_sample_size: int = 10
    val_sample_size: int = 10

    target_metric: str = "val_MulticlassF1Score"

    # Data transform settings
    augmentation_enabled: bool = True
    # Data augmentation seed
    # If None - defaults to general seed
    augmentation_seed: typing.Optional[int]

    # Model settings
    # model: SimpsonsModel = "SimpsonsNetConv"

    # Transfer learning settings
    # vgg16
    # feature_layers_freeze: int = 10
    dropout_p: float = 0.5

    # Weights init settings
    # If none - governed by general seed
    weights_init_seed: typing.Optional[int]

    # Dropout settings
    # If none - governed by general seed
    dropout_seed: typing.Optional[int]

    # Trainer settings
    tf32: bool = False
    device: str = "cuda:0"
    compile_model: bool = False
    model_checkpoint_strategy: str = "last_epoch"
    show_step: int = 1
    num_epochs: int = 100
    deterministic: bool = True

    # Data loader settings
    # 0 -> 26s, 1 -> 15s, 2 -> 10s, 3 -> 7s, 6 -> 6s/epoch
    loader_num_workers: int = 6
    data_shuffle_seed: typing.Optional[int] = None

    # tracking
    tracking_type: str = "mlflow"
    mlflow_tracking_uri: str = "/data/horse/ws/s4610340-isml/mlruns"
    mlflow_experiment: str = "isml/cifar10_clf/debug"
    mlflow_run_name: typing.Optional[str] = None
    mlflow_log_system_metrics: bool = False
    mlflow_artifact_location: typing.Optional[str] = None

    def __post_hook__(self):
        if self.run_uid is None:
            object.__setattr__(self, "run_uid", str(uuid4()))
