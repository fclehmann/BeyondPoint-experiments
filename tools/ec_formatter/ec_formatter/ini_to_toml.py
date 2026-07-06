import configparser
import json
import pathlib
from collections import defaultdict
from typing import NamedTuple, Union, Any
import re
import tomlkit

ConfigValue = Union[str, int, float]


def convert_value_correct_data_type(x: str) -> Union[int, float, str]:
    # int
    if x.isnumeric():
        return int(x)
    # float
    try:
        return float(x)
    except ValueError:
        return x


def convert_config_values_to_correct_data_types(config: dict[str, dict[str, str]]) -> dict[str, dict[str, ConfigValue]]:
    for section_name, section_content in config.items():
        for k, v in section_content.items():
            config[section_name][k] = convert_value_correct_data_type(v)

    # Will work on float or int seeds
    if "seed" in config["DATA"]:
        s = config["DATA"]["seed"]
        config["DATA"]["seed"] = eval(f"[{s}]")

    return config


def read_ini_config(path: pathlib.Path) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(path)
    return config


def ini_config_to_dict(config: configparser.ConfigParser) -> dict[str, Any]:
    return {sect: dict(config.items(sect)) for sect in config.sections()}


def convert_ini_dimensions_to_json(dimensions_ini: dict[str, str]) -> list[dict[str, ConfigValue]]:
    dimensions = defaultdict(dict)

    pattern_dim_name = re.compile("dim_(\d)_name")
    pattern_range_generator = re.compile("range_generator_(\d)")
    pattern_min = re.compile("min_dim_(\d)")
    pattern_max = re.compile("max_dim_(\d)")
    pattern_options = re.compile("options_(\d)")

    for dimension_spec in dimensions_ini.items():
        name, value = dimension_spec
        if name == "dimensions":
            continue
        if pattern_dim_name.match(name):
            i = int(name[4])
            dimensions[i]["name"] = value
        elif pattern_range_generator.match(name):
            i = int(name[-1])
            dimensions[i]["range_generator"] = value
        elif pattern_min.match(name):
            i = int(name[-1])
            dimensions[i]["min"] = value
        elif pattern_max.match(name):
            i = int(name[-1])
            dimensions[i]["max"] = value
        elif pattern_options.match(name):
            i = int(name[-1])
            dimensions[i]["options"] = value

    return list(dimensions.values())


def convert_ini_ec_to_toml(input_path: pathlib.Path, output_path: pathlib.Path) -> None:
    config_dict = ini_config_to_dict(read_ini_config(input_path))

    convert_config_values_to_correct_data_types(config_dict)

    config_dict["DIMENSIONS"] = convert_ini_dimensions_to_json(config_dict["DIMENSIONS"])

    toml_config = tomlkit.document()
    for section_name, section_content in config_dict.items():
        toml_config.append(section_name, section_content)

    with open(output_path, "w") as file:
        tomlkit.dump(toml_config, file)
