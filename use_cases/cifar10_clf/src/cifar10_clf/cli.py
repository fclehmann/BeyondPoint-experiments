import logging
import pathlib
import sys
import uuid
import warnings

import isml_train
import isml_train.utils
import mlflow
import spock

from cifar10_clf.config import ExperimentConfig
from cifar10_clf.train import train

_logger = logging.getLogger(__name__)


def main():
    builder = spock.SpockBuilder(ExperimentConfig, desc="Experiment config")

    config: ExperimentConfig = builder.generate().ExperimentConfig

    path_run = pathlib.Path(config.out_path) / config.run_uid  # type: ignore

    builder.save(
        file_name="config",
        create_save_path=True,
        user_specified_path=path_run,  # type: ignore
    )

    isml_train.utils.setup_logging(
        log_level=logging.DEBUG, log_file_path=path_run / "logs.log"
    )
    logging.getLogger("PIL.PngImagePlugin").setLevel("WARNING")
    warnings.filterwarnings(action="ignore", module=r".*torch\.autograd")

    try:
        val_metric = train(config)
    except Exception as e:
        _logger.exception(e)
        raise e

    print(f"RESULT: {1 - val_metric}")


if __name__ == "__main__":
    main()
