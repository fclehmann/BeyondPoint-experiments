import dataclasses
import pathlib
import typing
from typing import Literal, Optional, Union, Any, TypeVar, Generic
import pydantic.dataclasses
import tomlkit


@pydantic.dataclasses.dataclass
class BaseExperimentEnvParams:
    command_name: str
    experiments_path: str
    objective_program_name: str
    experiment_config_path: str
    experiment_name: str = None
    work_dir: str = None
    out_path_option_name: str = "ExperimentConfig.out_path"
    explicit_boolean_options: bool = False

    def __post_init__(self):
        file_name = pathlib.Path(self.experiment_config_path).stem
        self.experiment_name = file_name.replace("-", "_").replace(".", " ").replace(" ", "")
        self.work_dir = str(pathlib.Path(self.experiments_path) / self.experiment_name)


@pydantic.dataclasses.dataclass
class SingularityExperimentEnvParams:
    env_type: Literal["singularity"]
    base_params: BaseExperimentEnvParams
    singularity_image_file: str


@pydantic.dataclasses.dataclass
class PreCreatedVenvExperimentEnvParams:
    env_type: Literal["pre-created-venv"]

    base_params: BaseExperimentEnvParams

    venv_path: str


@pydantic.dataclasses.dataclass
class VenvExperimentEnvParams:
    env_type: Literal["venv"]

    base_params: BaseExperimentEnvParams

    commit_sha: str
    rel_project_path: str


@pydantic.dataclasses.dataclass
class ExtraHardwareParams:
    num_cpus_per_worker: int
    reservation: typing.Optional[str] = None


@pydantic.dataclasses.dataclass
class OmnioptData:
    seed: Union[int, list[int]]
    number_of_workers: int
    precision: int
    account: str
    partition: str
    enable_gpus: int
    mem_per_cpu: int
    computing_time: int
    max_evals: int
    algo_name: str
    range_generator_name: str
    projectname: str = None
    num_gpus_per_worker: typing.Optional[int] = None


@pydantic.dataclasses.dataclass
class RangeDimension:
    name: str
    range_generator: Literal["hp.uniformint", "hp.loguniform", "hp.uniform"]
    min: Union[float, int]
    max: Union[float, int]


@pydantic.dataclasses.dataclass
class ChoiceDimension:
    name: str
    range_generator: Literal["hp.choice"]
    options: list[Any]


@pydantic.dataclasses.dataclass
class OmnioptDebugConf:
    debug_xtreme: int
    debug: int
    info: int
    warning: int
    success: int
    stack: int
    show_live_output: int
    sbatch_or_srun = "sbatch"
    debug_sbatch_srun: int


@pydantic.dataclasses.dataclass
class OmnioptMongoConf:
    worker_last_job_timeout: int
    poll_interval: int
    kill_after_n_no_results: int


_ENV_TYPE = TypeVar("_ENV_TYPE", VenvExperimentEnvParams, PreCreatedVenvExperimentEnvParams,
                    SingularityExperimentEnvParams)

@pydantic.dataclasses.dataclass
class ExperimentConfig(Generic[_ENV_TYPE]):
    FIXED_EXPERIMENT_PARAMS: dict[str, Any]

    EXTRA_HARDWARE_PARAMS: ExtraHardwareParams

    DATA: OmnioptData

    DIMENSIONS: list[Union[RangeDimension, ChoiceDimension]]

    DEBUG: OmnioptDebugConf

    MONGODB: OmnioptMongoConf

    ENV_PARAMS: _ENV_TYPE = pydantic.Field(discriminator="env_type")

    LAUNCHER: Optional[dict[str, Any]] = None

    def __post_init_post_parse__(self):
        self.FIXED_EXPERIMENT_PARAMS[self.ENV_PARAMS.base_params.out_path_option_name] = f"{self.ENV_PARAMS.base_params.work_dir}/results/"
        self.DATA.projectname = self.ENV_PARAMS.base_params.experiment_name


def read_experiment_config(config_file_path: pathlib.Path) -> ExperimentConfig:
    """
    Reads and validates experiment config from toml file
    :param config_file_path: path to config
    :return: config
    """
    with open(config_file_path) as conf_file:
        conf = tomlkit.loads(config_file_path.read_text()).value
        conf["ENV_PARAMS"]["base_params"]["experiment_config_path"] = str(config_file_path)
        return ExperimentConfig(**conf)
