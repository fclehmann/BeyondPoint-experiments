"""
Resources:
srun -p alpha -A p_scads --nodes=1 --ntasks=1 --cpus-per-task=8 --hint=nomultithread --mem-per-cpu=100000 --gres=gpu:1 --time=04:00:00 --pty bash
module load modenv/hiera GCCcore/10.3.0 Python/3.9.5 GCC/10.3.0  CUDA/11.3.1  OpenMPI/4.1.1 PyTorch/1.10.0

Test run:
~/.local/bin/poetry run train-simpsons-model -c /beegfs/ws/0/clehm-machine-learning-projects/statistics-in-machine-learning/inferentialStatisticsMLmetrics/projects/simpsons/configs/example_config_taurus.yaml
"""

import functools
import logging
import os
import typing

import sklearn.model_selection

import torch.nn
import torch.optim
import torch.utils.data
import torchvision.transforms

from simpsons.models.simpsons_net_conv import SimpsonsNetConv
from simpsons.dataset import SimpsonsDataset
from simpsons.trainer import SimpsonsClassifierModelTrainer
from simpsons.config import ExperimentConfig, SimpsonsModel

import isml_train.train_test_algorithm
import isml_train.metric
import isml_train.results_saver
import isml_train.structures
import isml_train.utils
import isml_train.data_loader_creator
import isml_train.pipeline

_logger = logging.getLogger(__name__)


def track_modules_environment() -> dict[str, typing.Any]:
    """
    Gets module list
    https://doc.zih.tu-dresden.de/software/modules/
    :return: serializable dict
    """
    import subprocess
    import shutil

    module_list: bytes = b''
    try:
        if shutil.which("module") is None:
            raise Exception("module command does not exist")
        _logger.info("Loading module list ...")
        module_list: bytes = subprocess.check_output("module list", shell=True)
    except Exception as e:
        _logger.warning("Taurus modules information is not available.")
        pass

    return {
        "taurus_modules": module_list.decode()
    }


def get_omniopt_seed() -> typing.Optional[float]:
    """
    Gets omniopt(->hyperopt) seed from slurm job name if possible.
    :return: serializable dict
    """

    import re

    job_name: typing.Optional[str] = os.environ.get("SLURM_JOB_NAME", None)

    if job_name is None:
        return None

    seed_regex = r"SEED_(\d*\.\d+|\d+)"
    match = re.search(seed_regex, job_name)

    if match is None:
        return None
    else:
        return float(match.group(1))


_N = typing.TypeVar("_N", SimpsonsNetConv, torchvision.models.vgg.VGG)


def init_model(config: ExperimentConfig, dataset: SimpsonsDataset) -> _N:
    if config.model == SimpsonsModel.simpsons_net_conv.value:
        return SimpsonsNetConv(
            input_shape=(64, 64, 3),
            num_classes=len(dataset.classes)
        )

    elif config.model == SimpsonsModel.vgg16.value:
        model = torchvision.models.vgg16(pretrained=True)

        # freeze the layers
        feature_layers_freeze = config.feature_layers_freeze
        for param in model.features[:feature_layers_freeze].parameters():
            param.requires_grad = False

        # Set correct number of classes in classifier
        num_features = model.classifier[6].in_features
        features = list(model.classifier.children())[:-1]  # Remove last layer
        features.extend([torch.nn.Linear(num_features, len(dataset.classes))])
        model.classifier = torch.nn.Sequential(*features)

        return model

    else:
        raise Exception(f"Unsupported model {config.model}")


def load_transformations(config: ExperimentConfig, input_size: tuple[int, int]) -> tuple[
    torchvision.transforms.Compose, torchvision.transforms.Compose]:
    transforms_resize: list[typing.Callable] = []

    if config.apply_resize:
        transforms_resize.append(torchvision.transforms.Resize(input_size))

    transforms_common_start = transforms_resize + [torchvision.transforms.CenterCrop(input_size)]
    transforms_common_end = [torchvision.transforms.ToTensor()]
    transforms_augment = []

    if config.augmentation_enabled:
        transforms_augment = [
            torchvision.transforms.RandomChoice([
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ColorJitter(contrast=0.9),
                torchvision.transforms.ColorJitter(brightness=0.1),
                torchvision.transforms.RandomApply(
                    [
                        torchvision.transforms.RandomHorizontalFlip(p=1),
                        torchvision.transforms.ColorJitter(contrast=0.9)
                    ],
                    p=0.5
                ),
                torchvision.transforms.RandomApply(
                    [
                        torchvision.transforms.RandomHorizontalFlip(p=1),
                        torchvision.transforms.ColorJitter(brightness=0.1)
                    ],
                    p=0.5
                ),
            ])
        ]

    transforms_train = transforms_common_start + transforms_augment + transforms_common_end
    transforms_test = transforms_common_start + transforms_common_end

    return torchvision.transforms.Compose(transforms_train), torchvision.transforms.Compose(transforms_test)


def init_model_weights(
        model: _N,
        config: ExperimentConfig
):
    # Set torch seed to config.weights_init_seed
    if config.weights_init_seed is not None:
        _logger.info(f"Setting torch seed to config.weights_init_seed={config.weights_init_seed}...")
        torch.manual_seed(config.weights_init_seed)
        torch.cuda.manual_seed(config.weights_init_seed)
        torch.cuda.manual_seed_all(config.weights_init_seed)

    # Init weights
    _logger.info(f"Init weights...")
    if config.model == SimpsonsModel.simpsons_net_conv.value:
        layers = model.children()
    elif config.model == SimpsonsModel.vgg16.value:
        layers = [list(model.classifier.children())[-1]]
    else:
        raise Exception(f"Unsupported model by weights-init: {config.model}")

    _logger.debug(f"Setting weights in layers {layers} for model {config.model} ...")

    for layer in layers:
        if hasattr(layer, 'weight') and layer.weight.requires_grad:
            torch.nn.init.xavier_uniform_(layer.weight)
        if hasattr(layer, 'bias') and layer.weight.requires_grad:
            torch.nn.init.zeros_(layer.bias)

    # Set torch seed back to config.seed
    if config.weights_init_seed is not None and config.seed is not None:
        _logger.info(f"Setting torch seed back to config.seed={config.seed}...")
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)


def init_model_and_weights(config: ExperimentConfig, dataset: SimpsonsDataset) -> torch.nn.Module:
    model = init_model(config, dataset)
    init_model_weights(model, config)
    return model


def load_train_test_split_config(config: ExperimentConfig):
    """
    Loads train-test split config from parsed cli args.
    :param config: parsed cli args
    :return: train-test split config
    """
    return {
        "algorithm": config.train_test_split_algorithm,
        "options": {
            "n_folds": config.n_folds,
            "val_size": config.val_size,
            "stratified": config.stratified,
            "train_test_split_algorithm_seed": config.train_test_split_algorithm_seed
        }
    }


def create_trainer(
        config: ExperimentConfig,
        dataset: SimpsonsDataset,
        device: torch.device,
        optimizer_loader: isml_train.pipeline.TorchOptimizerLoader,
        lrs_loader: isml_train.pipeline.TorchLRSchedulerLoader,
) -> SimpsonsClassifierModelTrainer:
    # Load model
    model = init_model(config, dataset)
    model.to(device)
    _logger.info(f"Model:\n{model}", )

    # Init model weights
    init_model_weights(model, config)

    # Load optimizer
    optimizer = optimizer_loader(model.parameters())
    _logger.info(f"Optimizer:\n{optimizer}")

    # Load loss
    loss = torch.nn.CrossEntropyLoss()
    _logger.info(f"Loss function:\n{loss}")

    # Create lr-scheduler
    lr_scheduler = lrs_loader(optimizer)
    _logger.info(f"LR scheduler:\n{lr_scheduler}")

    return SimpsonsClassifierModelTrainer(
        base_nn_model=model,
        optimizer=optimizer,
        loss_function=loss,
        class_to_indexes=dataset.class_to_idx,
        n_epochs=config.num_epochs,
        device=device,
        show_step=config.show_step,
        lr_scheduler=lr_scheduler
    )


def create_and_split_datasets(
        config: ExperimentConfig,
        dataset_train: SimpsonsDataset,
        dataset_test: SimpsonsDataset,
) -> tuple[
    tuple[torch.utils.data.Subset, typing.Sequence[int]],
    tuple[torch.utils.data.Subset, typing.Sequence[int]],
    tuple[torch.utils.data.Subset, typing.Sequence[int]]
]:
    """
    Splits dataset on train-val and test. Input train and test dataset refer to correspondingly
    *whole* dataset with train transformations applied (data augmentation)
    and test transformations applied.
    :param config: Experiment config
    :param dataset_train: *whole* dataset with train transformations applied
    :param dataset_test: *whole* dataset with test transformations applied
    :return: (
        (train-val dataset with train transformations applied, train-val class indexes),
        (train-val dataset with test transformations applied, train-val class indexes)
        (test dataset with test transformations applied, test class indexes)
    )
    """
    _logger.info("Creating train-val and test subsets ...")

    stratify = None
    if config.stratified_test_split is not None:
        stratify = dataset_train.targets

    if config.test_size > 0:
        train_val_idx, test_idx = sklearn.model_selection.train_test_split(
            dataset_train.index,
            test_size=config.test_size,
            shuffle=True,
            stratify=stratify,
            random_state=config.train_test_split_algorithm_seed
        )
    else:
        train_val_idx = dataset_train.index
        test_idx = []

    train_subset = torch.utils.data.Subset(dataset_train, train_val_idx)
    val_subset = torch.utils.data.Subset(dataset_test, train_val_idx)
    test_subset = torch.utils.data.Subset(dataset_test, test_idx)

    train_val_targets = [dataset_train.targets[ix] for ix in train_val_idx]
    test_targets = [dataset_train.targets[ix] for ix in test_idx]

    return (
        (train_subset, train_val_targets),
        (val_subset, train_val_targets),
        (test_subset, test_targets)
    )


def train_simpsons_model(config: ExperimentConfig, experiment_uid: str) -> float:
    _logger.info(f"Simpson model train with config:\n {config}")
    _logger.info(f"Experiment UID: {experiment_uid}")

    # Set global seed
    if config.seed is None:
        _logger.warning("Seed is not set!!! seed=None => global seeds not set.")
    else:
        _logger.info(f"Set seed={config.seed}")
        isml_train.utils.seed_everything(seed=config.seed)

    # Load opt config
    opt_loader = isml_train.pipeline.TorchOptimizerLoader(config)
    opt_config = opt_loader.get_optimizer_config()
    _logger.info(f"Loaded optimizer settings: {opt_config}")

    # Load train-test-split config
    train_test_split_settings = load_train_test_split_config(config)
    _logger.info(f"Loaded train-test split settings: {train_test_split_settings}")

    # Load LR-scheduler config
    lrs_loader = isml_train.pipeline.TorchLRSchedulerLoader(config)
    lrs_config = lrs_loader.get_lrs_config()
    _logger.info(f"Loaded lr scheduler settings: {lrs_config}")

    SLURM_track = None

    # define variables for SLURM tracking
    if config.slurm_tracking:
        _logger.info(f'SLURM tracking: {config.slurm_tracking}')
        slurm_tracking_vars = ['SLURM_JOBID', 'SLURM_JOB_NAME', 'SLURM_JOB_PARTITION', 'SLURM_NODELIST']
        SLURM_track = {'SLURM_tracking': {k: os.getenv(key=k) for k in slurm_tracking_vars}}

    # Load train and test transforms
    model_to_input_size = {
        SimpsonsModel.simpsons_net_conv.value: (64, 64),
        SimpsonsModel.vgg16.value: (224, 224)
    }
    transforms_train, transforms_test = load_transformations(config, model_to_input_size[config.model])

    # collect experiment setting
    # TODO: Don't repeat config
    experiment_settings = {
        'description_label': config.short_name,
        'train_data_path': config.training_data_root,
        'out_path': config.out_path,
        'optimizer': opt_config,
        'lr_scheduler': lrs_config,
        'num_epochs': config.num_epochs,
        'batch_size': config.batch_size,
        'seed': config.seed,
        'slurm_tracking': config.slurm_tracking,
        'train_test_split_config': train_test_split_settings,
        'target_metric': config.target_metric,
        'transforms': {
            'train': str(transforms_train),
            'test': str(transforms_test)
        }
    }

    # Create train, validation and test datasets

    _logger.debug("Creating dataset ...")
    dataset_train: SimpsonsDataset = SimpsonsDataset(
        config.training_data_root,
        transform=transforms_train,
        loader=functools.cache(torchvision.datasets.folder.default_loader)
    )
    dataset_test: SimpsonsDataset = SimpsonsDataset(
        config.training_data_root,
        transform=transforms_test,
        loader=functools.cache(torchvision.datasets.folder.default_loader)
    )

    _logger.info(f"Train dataset to be used:\n{dataset_train}")
    _logger.info(f"Test dataset to be used:\n{dataset_test}")

    (subset_train, targets_train), (subset_val, _), (subset_test, targets_test) = create_and_split_datasets(
        config=config,
        dataset_train=dataset_train,
        dataset_test=dataset_test
    )

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _logger.info(f"DEVICE to be used: {device}")

    # Create trainer
    trainer = create_trainer(
        config=config,
        dataset=dataset_train,
        device=device,
        optimizer_loader=opt_loader,
        lrs_loader=lrs_loader
    )

    if experiment_settings["train_test_split_config"]["options"]["train_test_split_algorithm_seed"] is None:
        _logger.info("Train-test algorithm seed is not set => global seed will be used.")
    else:
        _logger.info(
            f'Train-test algorithm seed is set to {experiment_settings["train_test_split_config"]["options"]["train_test_split_algorithm_seed"]}')

    train_test_class: typing.Type[isml_train.train_test_algorithm.AbstractTrainTestAlgo] = next(
        a for a in [
            isml_train.train_test_algorithm.BootstrapTrainTestSplit,
            isml_train.train_test_algorithm.SimpleTrainTestSplit,
            isml_train.train_test_algorithm.CVTrainTestSplit
        ]
        if a.algo_name() == config.train_test_split_algorithm
    )
    _logger.info(f"Train-test-algo class:\n{train_test_class}")

    train_test_algo: isml_train.train_test_algorithm.AbstractTrainTestAlgo = train_test_class(
        trainer=trainer,
        train_data_loader_creator=isml_train.data_loader_creator.DataLoaderCreator(
            augmentation_seed=config.augmentation_seed,
            batch_size=config.batch_size,
            pin_memory=False,
            shuffle=True
        ),
        test_data_loader_creator=isml_train.data_loader_creator.DataLoaderCreator(
            augmentation_seed=config.augmentation_seed,
            batch_size=config.batch_size,
            shuffle=False
        ),
        reset_trainer=lambda: create_trainer(
            config=config,
            dataset=dataset_train,
            device=device,
            optimizer_loader=opt_loader,
            lrs_loader=lrs_loader
        ),
        **train_test_split_settings["options"]
    )

    if config.dropout_seed is not None:
        _logger.info(f"Setting torch seed (to affect dropout) to dropout seed: {config.dropout_seed}")
        torch.cuda.manual_seed_all(config.dropout_seed)
        torch.manual_seed(config.dropout_seed)

    train_metrics = train_test_algo(
        dataset_train=subset_train,
        dataset_test=subset_val,
        targets=targets_train
    )

    if config.dropout_seed is not None:
        _logger.info(f"Setting torch seed (affected dropout) back to default seed: {config.seed}")
        torch.cuda.manual_seed_all(config.seed)
        torch.manual_seed(config.seed)

    target_metric_class: typing.Type[isml_train.metric.AbstractTargetMetric] = next(
        m for m in [
            isml_train.metric.ErrorRateTargetMetric,
            isml_train.metric.LossTargetMetric
        ] if m.metric_name() == config.target_metric
    )

    target_metric_value: float = target_metric_class(subset="val")(train_metrics)

    test_data_loader = isml_train.data_loader_creator.DataLoaderCreator(
        augmentation_seed=config.augmentation_seed,
        batch_size=config.batch_size,
        shuffle=False
    )(subset_test)

    test_metrics: typing.Optional[isml_train.structures.EvalResult] = None
    if config.test_size > 0:
        test_metrics = trainer.evaluate(data_loader=test_data_loader)

    _logger.info(f"Evaluation result on test subset: {test_metrics}")

    if config.save_results:
        isml_train.results_saver.save_results(
            train_metrics=train_metrics,
            target_metric_value=target_metric_value,
            test_evaluation_result=test_metrics,
            model_setup=isml_train.structures.ModelSetup(
                model=trainer.model,
                optimizer=trainer.optimizer,
                criterion=trainer.loss
            ),
            experiment_settings=experiment_settings,
            data_saver_config=config,
            SLURM_track=SLURM_track,
            environment={
                **track_modules_environment(),
                "omniopt_seed": get_omniopt_seed()
            },
            experiment_uid=experiment_uid
        )

    return target_metric_value
