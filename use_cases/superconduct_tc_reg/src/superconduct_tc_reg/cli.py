# Defines omniopt-compatible CLI
import logging
import os
import pathlib
import sys
import hydra
import isml_train.utils
import retry
import typer
import isml_train
import superconduct_tc_reg.train

_logger = logging.getLogger(__name__)

app = typer.Typer()


def _reformat_cli_arg(s: str):
    if s.startswith("--"):
        return s[2:]
    elif s.startswith("-"):
        return s[1:]
    else:
        return s


@app.command(
    "train",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def train(ctx: typer.Context):
    # Read and format cli args to satisfy the hydra format
    formatted_cli_args = list(map(_reformat_cli_arg, ctx.args))

    # Init hydra config
    hydra.initialize(
        version_base=None,
        config_path="../../configs",
    )
    cfg = hydra.compose(config_name="train", overrides=formatted_cli_args)

    # Create run dir
    path_run = pathlib.Path(cfg.out_path) / cfg.run_uid
    path_run.mkdir(exist_ok=True, parents=True)
    os.chdir(path_run)

    isml_train.utils.setup_logging(
        log_level=logging.DEBUG, log_file_path=path_run / "logs.log"
    )

    _logger.info(f"Run path: {path_run}")

    train_with_retry = retry.retry(
        exceptions=(Exception,), tries=3, delay=1, logger=_logger
    )(superconduct_tc_reg.train.train)

    # Train
    try:
        val_loss = train_with_retry(cfg) # type: ignore
    except Exception as e:
        _logger.exception(e)
        raise e

    print(f"RESULT: {val_loss}")


if __name__ == "__main__":
    app()
