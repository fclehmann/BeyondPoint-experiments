import logging
import pathlib
import sys
import uuid

import spock

from simpsons.config import ExperimentConfig
from simpsons.pipeline import train_simpsons_model

_logger = logging.getLogger(__name__)


def main():
    experiment_uid: str = str(uuid.uuid4())

    builder = spock.SpockBuilder(
        ExperimentConfig, desc='Experiment config', salt=experiment_uid
    )

    config: ExperimentConfig = builder.generate().ExperimentConfig

    builder.save(
        file_name=experiment_uid,
        create_save_path=True,
        user_specified_path=pathlib.Path(config.out_path) / experiment_uid
    )

    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(
                pathlib.Path(config.out_path) / experiment_uid / f"log-{config.out_file_string}-{experiment_uid}.log"),
            logging.StreamHandler(stream=sys.stdout)
        ]
    )
    logging.root.setLevel(logging.NOTSET)

    try:
        target_metric_value = train_simpsons_model(config, experiment_uid=experiment_uid)
    except Exception as e:
        _logger.exception(e)
        raise e

    print(f"RESULT: {target_metric_value}")


if __name__ == '__main__':
    main()
