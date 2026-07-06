import functools
import logging
import os
from pathlib import Path
from typing import Callable
from attr import asdict
import isml_train.tracking
import isml_train.utils
import mlflow
import mlflow.entities
import mlflow.entities.run
import torchinfo
import torchmetrics
import torchmetrics.classification
import torchvision.datasets
import torchvision.transforms.v2 as transforms
from cifar10_clf.config import ExperimentConfig
from cifar10_clf.dataset import LazyCIFAR10Combined
from cifar10_clf.trainer import Trainer
from cifar10_clf.folds_gen import TrainValBootstapSamplesGeneratorSingleFold
from cifar10_clf.utils import dicts_mean, modify_dict_keys, untensor_dict
from cifar10_clf.metrics import HConfusionMatrix
from cifar10_clf.models import VGG11BN
from cifar10_clf.tracking import FileSystemLogger, MlFlowLogger
import lightning.pytorch.loggers

import torch.utils.data
import sklearn.utils
import isml_train.pipeline
import isml_train.data_loader_creator

_logger = logging.getLogger(__name__)


def load_transformations(
    config: ExperimentConfig,
) -> tuple[transforms.Compose, transforms.Compose]:
    transforms_common_end = [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.49154913, 0.4821251, 0.44642678), (0.24703223, 0.24348513, 0.26158784)
        ),
    ]
    transforms_augment = []

    if config.augmentation_enabled:
        transforms_augment = [
            transforms.RandomCrop(
                size=(32, 32), padding=4, pad_if_needed=True
            ),
            transforms.RandomHorizontalFlip(p=0.5),
        ]

    return transforms.Compose(
        transforms_augment + transforms_common_end
    ), transforms.Compose(transforms_common_end)


def train_test_split(targets: list[int], config: ExperimentConfig):
    """Uses stratified resampling to generate split

    :param targets: targets
    :param config: experiment config
    :return: Out-of-bootstrap indexes, sample indexes
    """
    index = list(range(len(targets)))

    idx_test: list[int] = sklearn.utils.resample(
        index,
        replace=config.test_sample_with_replacement,
        n_samples=config.test_sample_size,
        random_state=config.test_sample_seed,
        stratify=targets,
    )  # type: ignore

    idx_train = [idx for idx in index if idx not in idx_test]

    return idx_train, idx_test


def train(config: ExperimentConfig) -> float:
    # Support for different tracking backends
    if config.tracking_type == "mlflow":
        tracking_logger = MlFlowLogger(
            run_name=config.mlflow_run_name,
            experiment_name=config.mlflow_experiment,
            tracking_uri=config.mlflow_tracking_uri,
            artifact_location=config.mlflow_artifact_location,
        )
    elif config.tracking_type == "filesystem_cluster":
        tracking_logger = FileSystemLogger(out_path=Path(config.out_path), run_uid=config.run_uid)  # type: ignore
    else:
        raise ValueError()

    tracking_logger.log_hyperparams(asdict(config))

    # Obtain and log slurm job info if running with omniopt
    slurm_job_info = isml_train.utils.get_slurm_job_info()
    _logger.info(f"SLURM job info: {slurm_job_info}")
    tracking_logger.log_hyperparams({"SLURM_tracking": slurm_job_info})
    # Obtain and log omniopt seed if possible
    tracking_logger.log_hyperparams({"omniopt_seed": isml_train.utils.get_omniopt_seed()})

    # Set overall seed
    if config.seed is not None:
        _logger.info(f"Set default seed to {config.seed}")
        isml_train.utils.seed_everything(config.seed, enable_deterministic=config.deterministic)
        if config.deterministic:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    else:
        _logger.warning(f"Default seed is not set => deterministic behavior disabled!")


    # Create datasets
    transform_train, transform_inference = load_transformations(config)

    # Dataset with train transforms
    dataset_train = LazyCIFAR10Combined(
        config.training_data_root,
        transform=transform_train,
        loader=functools.cache(torchvision.datasets.folder.default_loader),
    )
    _logger.info(f"Train+val dataset:\n{dataset_train}")

    # Dataset with inference transforms
    dataset_inference = LazyCIFAR10Combined(
        config.training_data_root,
        transform=transform_inference,
        loader=functools.cache(torchvision.datasets.folder.default_loader),
    )
    _logger.info(f"Inference dataset:\n{dataset_inference}")

    # Hardware
    device = torch.device(config.device)
    _logger.info(f"Selected device: {device}")

    # Loss
    loss = torch.nn.CrossEntropyLoss()
    _logger.info(f"Loaded loss: {loss}")

    # Setup metrics
    metrics = torchmetrics.MetricCollection(
        [
            torchmetrics.Accuracy(
                task="multiclass", num_classes=len(dataset_train.class_to_idx)
            ),
            torchmetrics.F1Score(
                task="multiclass",
                num_classes=len(dataset_train.class_to_idx),
                average="macro",
            ),
            HConfusionMatrix(
                labels=sorted(
                    list(dataset_train.class_to_idx.keys()),
                    key=lambda c: dataset_train.class_to_idx[c],
                ),
                num_classes=len(dataset_train.class_to_idx),
            ),
        ]
    ).to(device)

    # Get train+val and test indices
    idx_train_val, idx_test = train_test_split(dataset_train.targets, config)

    # Generate folds
    if config.train_val_split_algorithm_seed is not None:
        split_algorithm_seed = config.train_val_split_algorithm_seed
        _logger.info(f"Set train-val split seed to {split_algorithm_seed}")
    else:
        split_algorithm_seed = config.seed
        _logger.warning(
            f"Set train-val split seed to the general seed: {split_algorithm_seed}"
        )

    folds_gen = TrainValBootstapSamplesGeneratorSingleFold(
        n_samples_train=config.train_sample_size,
        n_samples_val=config.val_sample_size,
        with_replacement=True,
        stratified=True,
        split_algorithm_seed=split_algorithm_seed,
    )

    metrics_by_fold = []
    models_by_fold = []

    # Train
    for fold_i, (idx_train, idx_val) in enumerate(
        folds_gen(
            elements=idx_train_val,
            targets=[dataset_train.targets[i] for i in idx_train_val],
        )
    ):

        # Init model
        # NOTE: When n folds > 1 the model weights are initialized identically for each fold,
        # beause the default rng of torch gets reset every time
        model = VGG11BN(
            num_classes=len(dataset_train.class_to_idx),
            weights_init_seed=config.weights_init_seed,
            initial_seed=config.seed,
            dropout=config.dropout_p,
        )
        model = model.to(device)

        if fold_i == 0:
            model_summary = torchinfo.summary(
                model=model,
                verbose=0,
                input_size=(config.batch_size, *dataset_train[0][0].shape),
            )
            _logger.info(f"Loaded model:\n {model_summary}")

        # Init optimizer
        optimizer = isml_train.pipeline.TorchOptimizerLoader(config)(model.parameters())
        _logger.info(f"Loaded optimizer: {optimizer}")

        # Init LR scheduler
        lr_scheduler = isml_train.pipeline.TorchLRSchedulerLoader(config)(optimizer)
        _logger.info(f"Loaded lr-scheduler: {lr_scheduler}")

        # Init trainer
        metric_postfix = f"_fold-{fold_i}" if folds_gen.n_folds > 1 else ""
        trainer = Trainer(
            loss=loss,
            optimizer=optimizer,
            device=device,
            metrics_train=metrics.clone(prefix="train_", postfix=metric_postfix),
            metrics_val=metrics.clone(prefix="val_", postfix=metric_postfix),
            metrics_test=metrics.clone(prefix="test_", postfix=metric_postfix),
            lr_scheduler=lr_scheduler,
            show_step=1,
            config=config,
            tracking_logger=tracking_logger,
        )

        # Create data loaders
        data_loader_creator = isml_train.data_loader_creator.DataLoaderCreator(
            augmentation_seed=config.augmentation_seed,
            data_shuffle_seed=config.data_shuffle_seed,
        )
        data_loader_train = data_loader_creator(
            dataset=torch.utils.data.Subset(dataset_train, idx_train),  # type: ignore
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.loader_num_workers,
            persistent_workers=config.loader_num_workers > 0,
            drop_last=False,
        )
        data_loader_val = data_loader_creator(
            dataset=torch.utils.data.Subset(dataset_inference, idx_val),  # type: ignore
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.loader_num_workers,
            persistent_workers=config.loader_num_workers > 0,
            drop_last=False,
        )

        # Train and validate
        # NOTE: When n folds > 1 the dropout impact is identical for each fold,
        # beause the default rng of torch gets reset every time
        if config.dropout_seed is not None:
            _logger.info(
                f"Setting torch seed (to affect dropout) to dropout seed: {config.dropout_seed}"
            )
            torch.cuda.manual_seed_all(config.dropout_seed)
            torch.manual_seed(config.dropout_seed)

        fold_val_metrics = trainer.fit(
            train_data_loader=data_loader_train,
            val_data_loader=data_loader_val,
            model=model,
        )
        fold_val_metrics = untensor_dict(fold_val_metrics)

        if config.dropout_seed is not None:
            _logger.info(
                f"Setting torch seed (affected dropout) back to default seed: {config.seed}"
            )
            torch.cuda.manual_seed_all(config.seed)  # type: ignore
            torch.manual_seed(config.seed)

        _logger.info(fold_val_metrics)

        # Remove postfixes and collect
        if folds_gen.n_folds > 1:
            metrics_by_fold.append(
                modify_dict_keys(
                    fold_val_metrics, lambda s: s.replace(metric_postfix, "")
                )
            )
            models_by_fold.append(model)

    # Aggregate folds metrics
    if folds_gen.n_folds > 1:
        metrics_avg = dicts_mean(metrics_by_fold)
        metrics_avg = modify_dict_keys(metrics_avg, lambda s: f"{s}_folds_avg")
        tracking_logger.log_metrics(metrics_avg, step=config.num_epochs-1)

    # Test
    # TODO: Implement test on best model according to selected val metric
    trainer.test(
        torch.utils.data.DataLoader(
            dataset=torch.utils.data.Subset(dataset_inference, idx_test),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.loader_num_workers,
        ),
    )

    tracking_logger.save()
    
    if folds_gen.n_folds > 1:
        return metrics_avg[f"{config.target_metric}_folds_avg"]
    else:
        return fold_val_metrics[config.target_metric] # type: ignore
