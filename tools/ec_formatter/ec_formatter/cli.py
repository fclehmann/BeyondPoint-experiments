import logging
import os
import pathlib
import typing

import pandas as pd
import typer

import ec_formatter.ini_to_toml

_logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command(name="ini-to-toml")
def ini_ec_to_toml(
        ini_config_path: pathlib.Path,
        output_file_path: typing.Optional[pathlib.Path] = typer.Argument(None)
) -> None:
    """
    Converts EC ini to toml
    """

    logging.basicConfig(level=logging.DEBUG)

    if output_file_path is None:
        name = ini_config_path.stem
        path = ini_config_path.parent
        output_file_path = path / f"{name}.toml"

    _logger.info(f"Output file path: {output_file_path}")

    ec_formatter.ini_to_toml.convert_ini_ec_to_toml(input_path=ini_config_path, output_path=output_file_path)


if __name__ == "__main__":
    app()
