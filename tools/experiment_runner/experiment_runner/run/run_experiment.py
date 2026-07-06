# cli options, params to opt, run script -> config_file_base64
# run script should be already on taurus
# run script should install (coin) experiment package OR (coin) package is
# already installed (CI?)
import abc
import base64
import configparser
import dataclasses
import io
import logging
import os
import pathlib
import subprocess
import shutil
from typing import NamedTuple, Any, Union, TypeVar, Generic, Type
import json
import jinja2

from experiment_runner.run.experiment_config import (
    VenvExperimentEnvParams,
    PreCreatedVenvExperimentEnvParams,
    SingularityExperimentEnvParams,
    ExperimentConfig,
    RangeDimension,
    ChoiceDimension,
    read_experiment_config
)
from tomlkit import parse
from experiment_runner.launcher.main import launch_sweepers
from experiment_runner.launcher.clear_experiment import clear_experiment

_logger = logging.getLogger(__name__)


class Dimension(NamedTuple):
    i: int
    name: str


_ENV_PARAMS_TYPE = TypeVar(
    "_ENV_PARAMS_TYPE",
    VenvExperimentEnvParams,
    PreCreatedVenvExperimentEnvParams,
    SingularityExperimentEnvParams
)


class ExperimentDeployerBase(abc.ABC, Generic[_ENV_PARAMS_TYPE]):
    experiment_config: ExperimentConfig[_ENV_PARAMS_TYPE]

    def __init__(self, experiment_config: ExperimentConfig[_ENV_PARAMS_TYPE]):
        self.experiment_config = experiment_config

    @property
    def objective_program_template_name(self) -> str:
        return "objective_program.jinja2"

    def build_objective_program_template_command_context(
            self, fixed_experiment_params: dict[str, Union[str, int, float]]
    ) -> dict[str, Any]:
        return {
            "command": {
                "name": self.experiment_config.ENV_PARAMS.base_params.command_name,
                "options": [{"name": name, "value": value} for name, value in fixed_experiment_params.items()]
            }
        }

    @abc.abstractmethod
    def build_objective_program_template_env_context(self) -> dict[str, Any]:
        ...

    def create_objective_program_script(self, jinja_env: jinja2.Environment) -> str:
        """
        Creates objective program script
        :param jinja_env: jinja env (<=>~ templates path)
        :return: objective program script as string
        """
        template: jinja2.Template = jinja_env.get_template(self.objective_program_template_name)
        context = {
            **self.build_objective_program_template_env_context(),
            **self.build_objective_program_template_command_context(self.experiment_config.FIXED_EXPERIMENT_PARAMS)
        }
        return template.render(context)

    @staticmethod
    def parse_dimensions_settings(dimensions: list[Union[RangeDimension, ChoiceDimension]]) -> list[Dimension]:
        """
        Parses DIMENSIONS section of experiment config
        :param dimensions: DIMENSIONS section of experiment config
        :return: parsed DIMENSIONS section
        """
        return [Dimension(i=i, name=d.name) for i, d in enumerate(dimensions)]

    @staticmethod
    def create_objective_program_option(
            objective_program_path: pathlib.Path,
            dimensions: list[Dimension]
    ) -> str:
        """
        Creates objective_program setting for experiment config
        :param objective_program_path: Path to objective program on taurus
        :param dimensions: parsed DIMENSIONS section
        :return: objective_program setting for experiment config
        """
        command = f"bash {str(objective_program_path)}"
        for dimension in dimensions:
            command += f' --{dimension.name}=($x_{dimension.i})'

        return command

    @staticmethod
    def encode_omniopt_config_ini(omniopt_config: dict[str, Any]) -> bytes:
        config_ini = configparser.ConfigParser()

        # Parse DIMENSIONS
        dimensions_ini: dict[str, Any] = dict()
        for dim_i, dim in enumerate(omniopt_config["DIMENSIONS"]):
            if "min" in dim:
                dimensions_ini[f"dim_{dim_i}_name"] = dim["name"]
                dimensions_ini[f"range_generator_{dim_i}"] = dim["range_generator"]
                dimensions_ini[f"min_{dim_i}"] = str(dim["min"])
                dimensions_ini[f"max_{dim_i}"] = str(dim["max"])
            elif "options" in dim:
                dimensions_ini[f"dim_{dim_i}_name"] = dim["name"]
                dimensions_ini[f"range_generator_{dim_i}"] = dim["range_generator"]
                dimensions_ini[f"options_{dim_i}"] = ",".join(str(v) for v in dim["options"])
        dimensions_ini["dimensions"] = len(omniopt_config["DIMENSIONS"])
        config_ini["DIMENSIONS"] = dimensions_ini

        # Parse DATA
        data_ini = dict()
        for key, value in omniopt_config["DATA"].items():
            if key == "seed":
                data_ini["seed"] = ",".join(str(i) for i in value)
            elif type(value) is bool:
                data_ini[key] = int(value)
            else:
                data_ini[key] = value

        config_ini["DATA"] = data_ini

        # Parse DEBUG
        debug_ini = dict()
        for key, value in omniopt_config["DEBUG"].items():
            if type(value) is bool:
                debug_ini[key] = int(value)
            else:
                debug_ini[key] = value

        config_ini["DEBUG"] = debug_ini

        # Parse MONGODB
        config_ini["MONGODB"] = omniopt_config["MONGODB"]

        config_io: io.StringIO = io.StringIO()
        config_ini.write(config_io)
        return base64.b64encode(bytes(config_io.getvalue(), "utf-8"))

    @staticmethod
    def encode_omniopt_config(omniopt_config: dict[str, Any]) -> bytes:
        """
        Encodes omniopt config in base64
        :param omniopt_config: omniopt config
        :return: omniopt config in base64 encoding
        """
        config_io: io.StringIO = io.StringIO()
        json.dump(omniopt_config, config_io)
        return base64.b64encode(bytes(config_io.getvalue(), "utf-8"))

    def create_omniopt_config(
            self,
            objective_program_path: pathlib.Path
    ) -> dict[str, Any]:
        """
        Creates omniopt config from experiment config
        :param objective_program_path: path to objective program on taurus
        :return: omniopt config
        """

        # noinspection PyDataclass
        omniopt_conf: dict[str, Any] = dataclasses.asdict(self.experiment_config)
        del omniopt_conf["FIXED_EXPERIMENT_PARAMS"]
        del omniopt_conf["ENV_PARAMS"]
        del omniopt_conf["EXTRA_HARDWARE_PARAMS"]

        omniopt_conf["DATA"]["objective_program"] = ExperimentDeployerBase.create_objective_program_option(
            objective_program_path=objective_program_path,
            dimensions=ExperimentDeployerBase.parse_dimensions_settings(self.experiment_config.DIMENSIONS)
        )

        return omniopt_conf

    def sbatch_context_from_config(self) -> dict[str, Union[int, str]]:
        """
        Creates sbatch template context from experiment config
        :return: sbatch template context
        """
        total_gpus: int = self.experiment_config.DATA.number_of_workers * self.experiment_config.DATA.num_gpus_per_worker
        context = {
            "project_name": self.experiment_config.ENV_PARAMS.base_params.experiment_name,
            "cpus_per_task": self.experiment_config.EXTRA_HARDWARE_PARAMS.num_cpus_per_worker,
            "reservation": self.experiment_config.EXTRA_HARDWARE_PARAMS.reservation,
            "gres": total_gpus,
            "gpus_per_task": self.experiment_config.DATA.num_gpus_per_worker,
            "n_tasks": self.experiment_config.DATA.number_of_workers,
            "max_computing_time": self.experiment_config.DATA.computing_time,
            "mem_per_cpu": self.experiment_config.DATA.mem_per_cpu,
            "partition": self.experiment_config.DATA.partition
        }
        return context

    def create_sbatch_command(
            self,
            jinja_env: jinja2.Environment,
            sbatch_template_name: str = "sbatch_command.jinja2"
    ) -> str:
        """
        Creates omniopt-sbatch command from experiment config
        :param jinja_env: jinja env (<=>~ templates path)
        :param sbatch_template_name: template name of sbatch command
        :return: sbatch command as string
        """
        template: jinja2.Template = jinja_env.get_template(sbatch_template_name)
        return template.render(self.sbatch_context_from_config()).replace("\n", "")

    def create_omniopt_command(
            self,
            omniopt_config_b64: bytes,
            sbatch_command_b64: bytes,
            run_omniopt_command_template_name: str,
            jinja_env: jinja2.Environment,
            submit_jobs: bool,
    ) -> str:
        """
        Creates command which executes omniopt_script.
        :param omniopt_config_b64: omniopt config in base64 encoding
        :param sbatch_command_b64: sbatch command in base64 encoding
        :param run_omniopt_command_template_name: template name for this command
        :param jinja_env: jinja env (<=>~ templates path)
        :param submit_jobs: whether to submit jobs
        :return: command as string
        """
        template: jinja2.Template = jinja_env.get_template(run_omniopt_command_template_name)
        context = {
            "project_name": self.experiment_config.ENV_PARAMS.base_params.experiment_name,
            "omniopt_config_b64": omniopt_config_b64.decode(),
            "sbatch_command_b64": sbatch_command_b64.decode(),
            "workdir": self.experiment_config.ENV_PARAMS.base_params.work_dir,
            "submit_jobs": submit_jobs
        }
        return template.render(context, trim_blocks=True, lstrip_blocks=True)

    @staticmethod
    def execute_commands(
            *commands: str,
            show_logs: bool = True
    ) -> subprocess.CompletedProcess:
        """
        Executes bash commands sequentially
        :param commands: bash commands
        :param show_logs: if stdout is printed
        :return: object
        """
        for command in commands:
            if show_logs:
                return subprocess.run(command, check=True, shell=True)
            else:
                return subprocess.run(command, check=True, shell=True, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)

    @abc.abstractmethod
    def install_use_case(self, log_level: Union[int, str]) -> None:
        ...

    def execute_omniopt_command(self, omniopt_command: str) -> None:
        self.execute_commands(omniopt_command)

    def deploy(self, log_level: Union[int, str], submit_jobs: bool, clean_workdir: bool):
        # Config logging
        logging.basicConfig()
        _logger.setLevel(log_level)

        templates_path: pathlib.Path = pathlib.Path(f"{os.path.dirname(os.path.realpath(__file__))}/../templates")
        sbatch_template_name: str = "sbatch_command.jinja2"
        run_omniopt_command_template_name: str = "run_omniopt_script.jinja2"

        # Prepare jinja env
        jinja_env: jinja2.Environment = jinja2.Environment(loader=jinja2.FileSystemLoader(templates_path))

        # Build path to objective program on taurus
        objective_program_path = pathlib.Path(
            self.experiment_config.ENV_PARAMS.base_params.work_dir
        ) / self.experiment_config.ENV_PARAMS.base_params.objective_program_name

        # Create objective program script
        _logger.info(f"Creating objective program ...")
        objective_program: str = self.create_objective_program_script(jinja_env=jinja_env)

        _logger.debug(objective_program)

        # Create and encode omniopt config
        _logger.info(f"Creating and encoding omniopt config ...")
        omniopt_config_b64: bytes = self.encode_omniopt_config(self.create_omniopt_config(
            objective_program_path=objective_program_path
        ))

        _logger.debug(base64.b64decode(omniopt_config_b64).decode())

        # Create and encode sbatch command
        _logger.info(f"Creating and encoding sbatch command ...")
        sbatch_command_b64: bytes = base64.b64encode(bytes(
            self.create_sbatch_command(jinja_env=jinja_env, sbatch_template_name=sbatch_template_name),
            encoding="utf-8"
        ))

        _logger.debug(base64.b64decode(sbatch_command_b64).decode())

        # Create omniopt_script command
        _logger.info(f"Creating omniopt command ...")
        omniopt_command: str = self.create_omniopt_command(
            omniopt_config_b64=omniopt_config_b64,
            sbatch_command_b64=sbatch_command_b64,
            run_omniopt_command_template_name=run_omniopt_command_template_name,
            jinja_env=jinja_env,
            submit_jobs=submit_jobs
        )

        _logger.debug(omniopt_command)

        work_dir = self.experiment_config.ENV_PARAMS.base_params.work_dir

        # Create workdir if absent
        _logger.info(f"Creating working directory ({work_dir}) ...")
        self.execute_commands(f"mkdir -p {work_dir};")

        # Clean * in workdir on taurus if desired
        if clean_workdir:
            _logger.info(f"Cleaning working directory ({work_dir}) ...")
            self.execute_commands(f"rm -rf {work_dir}/*")

        # Create results directory
        _logger.info(
            f"Creating results (out_path) directory ({self.experiment_config.FIXED_EXPERIMENT_PARAMS[self.experiment_config.ENV_PARAMS.base_params.out_path_option_name]}) ...")
        self.execute_commands(
            f"mkdir -p {self.experiment_config.FIXED_EXPERIMENT_PARAMS[self.experiment_config.ENV_PARAMS.base_params.out_path_option_name]};")

        # Coppy objective program into workdir
        _logger.info(
            f"Copying objective program to taurus: {objective_program_path} ...")
        with pathlib.Path(objective_program_path).open(mode="w") as file:
            file.write(objective_program)
        self.execute_commands(
            f"chmod +x {objective_program_path}"    
        )

        # Copy configuration file to results directory
        _logger.info(
            f"Copying configuration file {self.experiment_config.ENV_PARAMS.base_params.experiment_config_path} to results directory: {work_dir} ...")
        shutil.copy(pathlib.Path(self.experiment_config.ENV_PARAMS.base_params.experiment_config_path),
                    pathlib.Path(work_dir) / pathlib.Path(
                        self.experiment_config.ENV_PARAMS.base_params.experiment_config_path).name)

        self.install_use_case(log_level=log_level)

        _logger.info(f"submit_jobs={submit_jobs}")

        if self.experiment_config.LAUNCHER is None:
            # Execute omniopt_script on taurus
            _logger.info(f"Executing omniopt command on taurus ...")
            self.execute_omniopt_command(omniopt_command)
        else:
            if clean_workdir:
                _logger.info("Deleting optuna studies of experiment...")
                clear_experiment(dataclasses.asdict(self.experiment_config))
            _logger.info(f"Starting launcher ...")
            launch_sweepers(pathlib.Path(self.experiment_config.ENV_PARAMS.base_params.experiment_config_path))


class ExperimentDeployerVenv(ExperimentDeployerBase[VenvExperimentEnvParams]):

    def install_use_case(self, log_level: Union[int, str]):
        repo_name = 'statistics-in-machine-learning'
        repo_url = f"git@gitlab.hrz.tu-chemnitz.de:clehm--tu-dresden.de/{repo_name}"

        work_dir = self.experiment_config.ENV_PARAMS.base_params.work_dir

        _logger.info(f"Cloning repository on commit {self.experiment_config.ENV_PARAMS.commit_sha} ...")
        self.execute_commands(
            f"cd {work_dir};"
            f"git clone {repo_url}; cd {repo_name}; git checkout {self.experiment_config.ENV_PARAMS.commit_sha}",
            show_logs=log_level <= logging.DEBUG
        )

        _logger.info("Create venv ...")
        self.execute_commands(
            f"module load modenv/hiera GCCcore/10.3.0 Python/3.9.5 GCC/10.3.0  CUDA/11.3.1  OpenMPI/4.1.1 PyTorch/1.10.0;"
            f"cd {work_dir}/{repo_name}/{self.experiment_config.ENV_PARAMS.rel_project_path};"
            f"python -m venv .venv;"
            f"~/.local/bin/poetry build --format wheel;"
            f".venv/bin/python -m pip install --force-reinstall ./dist/*",
            show_logs=log_level <= logging.DEBUG
        )

    repo_name: str

    def build_objective_program_template_env_context(self) -> dict[str, Any]:
        workdir = pathlib.Path(self.experiment_config.ENV_PARAMS.base_params.work_dir)
        return {
            "venv_path": str(workdir / self.repo_name / self.experiment_config.ENV_PARAMS.rel_project_path / ".venv"),
            "explicit_boolean_options": self.experiment_config.ENV_PARAMS.base_params.explicit_boolean_options
        }


class ExperimentDeployerPreMadeVenv(ExperimentDeployerBase[PreCreatedVenvExperimentEnvParams]):

    def install_use_case(self, log_level: Union[int, str]):
        _logger.info(f"Skipping repository cloning because env_type={self.experiment_config.ENV_PARAMS.env_type}")
        _logger.info(f"Skipping venv creation because env_type={self.experiment_config.ENV_PARAMS.env_type}")

    def build_objective_program_template_env_context(self) -> dict[str, Any]:
        return {
            "venv_path": self.experiment_config.ENV_PARAMS.venv_path,
            "explicit_boolean_options": self.experiment_config.ENV_PARAMS.base_params.explicit_boolean_options
        }


class ExperimentDeployerSingularity(ExperimentDeployerBase[SingularityExperimentEnvParams]):

    @property
    def objective_program_template_name(self) -> str:
        return "objective_program_singularity.jinja2"

    def install_use_case(self, log_level: Union[int, str]) -> None:
        _logger.info(f"Skipping repository cloning because env_type={self.experiment_config.ENV_PARAMS.env_type}")
        _logger.info(f"Skipping venv creation because env_type={self.experiment_config.ENV_PARAMS.env_type}")

    def run_on_host_per_ssh(self, command: str):
        # user = os.environ.get("USER")
        # host = os.environ.get("HOST")

        # if user is None or host is None:
        #     raise Exception("$USER or $HOST not set.")

        command = f"cd {self.experiment_config.ENV_PARAMS.base_params.work_dir}; {command}"

        # ExperimentDeployerBase.execute_commands(f"ssh {user}@{host} '{command}'")
        ExperimentDeployerBase.execute_commands(command)

    def execute_omniopt_command(self, omniopt_command: str) -> None:
        self.run_on_host_per_ssh(omniopt_command)

    def build_objective_program_template_env_context(self) -> dict[str, Any]:
        return {
            "gpus_enabled": bool(self.experiment_config.DATA.enable_gpus),
            "singularity_image_file": self.experiment_config.ENV_PARAMS.singularity_image_file,
            "explicit_boolean_options": self.experiment_config.ENV_PARAMS.base_params.explicit_boolean_options
        }


def run_experiment(
        experiment_config_path: pathlib.Path,
        clean_workdir: bool,
        submit_jobs: bool,
        log_level: int
) -> None:
    """
    Runs experiment defined by given experiment config.

    :param experiment_config_path: Path to experiment config
    :param clean_workdir: flag whether to delete * in workdir
    :param submit_jobs: flag whether to submit jobs
    :param log_level: log level as int, see logging.
    :return: None
    """

    # Config logging
    logging.basicConfig()
    _logger.setLevel(log_level)

    _logger.info(f"Starting experiment {experiment_config_path} deploy ...")

    _logger.info(f"Reading experiment config from {experiment_config_path} ...")
    config = read_experiment_config(experiment_config_path)

    env_type_to_deployer_cls: dict[str, Type[ExperimentDeployerBase]] = {
        "venv": ExperimentDeployerVenv,
        "pre-created-venv": ExperimentDeployerPreMadeVenv,
        "singularity": ExperimentDeployerSingularity
    }

    env_type_to_deployer_cls[config.ENV_PARAMS.env_type](experiment_config=config).deploy(
        log_level=log_level,
        clean_workdir=clean_workdir,
        submit_jobs=submit_jobs
    )

def continue_experiment(experiment_path: pathlib.Path):
    logging.basicConfig()
    _logger.setLevel(logging.DEBUG)

    _logger.info(f"Continue experiment {experiment_path}")

    path_experiment_config = next(experiment_path.glob("*.toml"))
    config = read_experiment_config(path_experiment_config)

    env_type_to_deployer_cls: dict[str, Type[ExperimentDeployerBase]] = {
        "venv": ExperimentDeployerVenv,
        "pre-created-venv": ExperimentDeployerPreMadeVenv,
        "singularity": ExperimentDeployerSingularity
    }

    deployer = env_type_to_deployer_cls[config.ENV_PARAMS.env_type](experiment_config=config)

    work_dir = pathlib.Path(config.ENV_PARAMS.base_params.work_dir)

    for sbatch_command_path in work_dir.glob("omniopt/sbatch_commands/*/0.sbatch"):
        raw_sbatch_command = sbatch_command_path.read_text()
        sbatch_command = f"cd {str(work_dir / 'omniopt')}; {raw_sbatch_command}"
        deployer.execute_omniopt_command(sbatch_command)


    