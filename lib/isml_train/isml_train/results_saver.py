import json
import pathlib
import typing
import uuid

import torch

import isml_train.config
from isml_train.structures import ModelSetup

import logging

_logger = logging.getLogger(__name__)


def save_results(
        train_metrics: typing.List[typing.Dict],
        target_metric_value: float,
        model_setup: ModelSetup,
        experiment_settings: typing.Dict,
        environment: dict,
        data_saver_config: isml_train.config.DataSaverConfig,
        SLURM_track: typing.Optional[typing.Dict] = None,
        experiment_uid: typing.Optional[str] = None,
        test_evaluation_result: typing.Optional[isml_train.structures.EvalResult] = None
) -> None:
    # acc_rates_df.to_csv(acc_file_string, index = False)

    if experiment_uid is None:
        experiment_uid = str(uuid.uuid4())

    experiment_output_path = pathlib.Path(data_saver_config.out_path) / experiment_uid
    experiment_output_path.mkdir(exist_ok=True)

    if data_saver_config.save_model:
        _logger.info(f'Saving model to {experiment_output_path}')
        path_to_trained_model = experiment_output_path / f"model-{data_saver_config.out_file_string}-{experiment_uid}.pt"
        torch.save(model_setup.model.state_dict(), path_to_trained_model)
    else:
        path_to_trained_model = ""

    result_data = {
        'uid': experiment_uid,
        'experiment_setup': {
            **experiment_settings,
            'path_to_trained_model': str(path_to_trained_model),
            'model': str(model_setup.model),
            'loss_function': str(model_setup.criterion)
        },
        'result_data': {
            "train_results": train_metrics,
            "target_metric_value": target_metric_value,
            "test_metrics": test_evaluation_result
        },
        'environment': environment
    }

    if SLURM_track is not None:
        result_data = {**result_data, **SLURM_track}

    # json.dumps(result_data, indent=4)

    result_path = pathlib.Path(
        experiment_output_path / f"result-{data_saver_config.out_file_string}-{experiment_uid}.json"
    )
    json_str = json.dumps(result_data, indent=4) + '\n'

    result_path.write_text(json_str, encoding='utf-8')
    _logger.info(f'Results written to json file: {result_path}')
