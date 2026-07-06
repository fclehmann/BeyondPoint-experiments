import logging
import os
from pathlib import Path
import optuna
from dataclasses import asdict

_logger = logging.getLogger(__name__)

def clear_experiment(exp_conf: dict):
    # Parameters
    db_port = exp_conf["LAUNCHER"]["db_port"]
    db_host = exp_conf["LAUNCHER"]["db_host"]
    db_pass = os.environ.get("DB_PASS")
    db_uri = f"postgresql://postgres:{db_pass}@{db_host}:{db_port}"
    storage = optuna.storages.RDBStorage(url=db_uri)

    seeds = exp_conf["DATA"]["seed"]
    if isinstance(exp_conf["DATA"]["seed"], int):
         seeds = [exp_conf["DATA"]["seed"]]
    
    for s in seeds:
        study_name = f'{exp_conf["ENV_PARAMS"]["base_params"]["experiment_name"]}_SEED_{s}'
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
        except Exception as e:
            _logger.warning(f"Failed to delete study: {study_name}", exc_info=e)