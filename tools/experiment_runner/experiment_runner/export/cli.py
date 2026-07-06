import json
import pathlib
import experiment_runner.export.distributions as distributions 

import typer

from typing import Annotated, Optional, List

app = typer.Typer()

@app.command("export-data-simpsons")
def export_data_soil_analysis(
    output_path: pathlib.Path,
    path_experiments: pathlib.Path,
    experiments: Annotated[List[str], typer.Option()] = [
        "10_simpsons_hopt_tpe_SGDwlrs_seeds",
        "11_simpsons_hopt_tpe_AdamW_seeds",
        "12_simpsons_hopt_tpe_Adamax_seeds",
        "13_simpsons_hopt_tpe_Adam_seeds",
        "14_simpsons_hopt_tpe_Adagrad_seeds",
        "15_simpsons_hopt_tpe_RMSprop_seeds",
        "16_simpsons_opt_split_AdamW",
        "17_simpsons_opt_augment_AdamW",
        "18_simpsons_opt_tl_vgg16_augment_AdamW",
        "19_simpsons_opt_tl_vgg16_split_AdamW",
        "21_simpsons_opt_weights_AdamW",
        "22_simpsons_opt_split_cv_AdamW",
        "23_simpsons_opt_split_bootstrap_AdamW",
        "24_simpsons_opt_tl_vgg16_weights_AdamW",
        "25_simpsons_opt_tl_vgg16_split_cv_AdamW",
        "26_simpsons_opt_dropout_AdamW",
        "27_simpsons_opt_tl_vgg16_dropout_AdamW",
        "20_simpsons_opt_split_augment_AdamW"
    ],
):
    if output_path.is_dir():
        raise ValueError("Output path must not be a directory!")
    
    exp_paths = [path_experiments / n for n in experiments]
    data = distributions.export_data(exp_paths)
    with open(output_path, "w") as f:
        json.dump(data, f)
    return data

@app.command("export-data-lrl_sorb")
def export_data_lrl_sorb(
    output_path: pathlib.Path,
    path_experiments: pathlib.Path,
    experiments: Annotated[List[str], typer.Option()] = [
        "001_lrl_sorb_hopt_tl_tpe_adamw_hf_seeds",
        "002_lrl_sorb_hopt_tl_tpe_paged_adamw_32bit_seeds",
        "003_lrl_sorb_hopt_tl_tpe_paged_adamw_8bit_seeds",
        "004_lrl_sorb_hopt_tl_tpe_paged_lion_32bit_seeds",
        "005_lrl_sorb_hopt_tl_tpe_paged_lion_8bit_seeds",
        "006_lrl_sorb_czech_hsb_hopt_tl_tpe_adamw_hf_seeds",
        "007_lrl_sorb_robeczech_csb_hopt_tl_tpe_adamw_hf_seeds",
        "008_lrl_sorb_robeczech_hsb_hopt_tl_tpe_adamw_hf_seeds",
        "009_lrl_sorb_czech_csb_opt_tl_adamw_hf_split",
        "010_lrl_sorb_czech_csb_opt_tl_adamw_hf_cv",
        "011_lrl_sorb_czech_csb_opt_tl_adamw_hf_bootstrap",
        "019_lrl_sorb_pl_rob_cs_opt_tl_adamw_torch_bootstrap",
        "021_lrl_sorb_pl_rob_cs_opt_tl_adamw_torch_split",
        "022_lrl_sorb_pl_bert_cs_opt_tl_adamw_torch_split",
    ],
):
    if output_path.is_dir():
        raise ValueError("Output path must not be a directory!")
    
    exp_paths = [path_experiments / n for n in experiments]
    data = distributions.export_data(exp_paths)
    with open(output_path, "w") as f:
        json.dump(data, f)
    return data

@app.command("export-data-cifar10")
def export_data_cifar10(
    output_path: pathlib.Path,
    path_experiments: pathlib.Path,
    experiments: Annotated[List[str], typer.Option()] = [
        "003_cifar10_clf_opt_split_bootstrap",
        "004_cifar10_clf_opt_augment",
        "005_cifar10_clf_opt_dropout",
        "006_cifar10_clf_opt_weights_init",
        "007_cifar10_clf_opt_data_order",
        "008_cifar10_clf_hopt_tpe_SGDwlrs_bs128_s"
    ],
):
    if output_path.is_dir():
        raise ValueError("Output path must not be a directory!")
    
    exp_paths = [path_experiments / n for n in experiments]
    data = distributions.export_data(exp_paths)
    with open(output_path, "w") as f:
        json.dump(data, f)
    return data


@app.command("export-data-soil_analysis")
def export_data_soil_analysis(
    output_path: pathlib.Path,
    path_experiments: pathlib.Path,
    experiments: Annotated[List[str], typer.Option()] = [
        "001_soil_analysis_opt_split",
        "002_soil_analysis_opt_weights_init",
        "003_soil_analysis_hopt_tpe_SGD_lrs",
    ],
):
    if output_path.is_dir():
        raise ValueError("Output path must not be a directory!")
    
    exp_paths = [path_experiments / n for n in experiments]
    data = distributions.export_data(exp_paths)
    with open(output_path, "w") as f:
        json.dump(data, f)
    return data

@app.command("export-data-superconduct_tc_reg")
def export_data_soil_analysis(
    output_path: pathlib.Path,
    path_experiments: pathlib.Path,
    experiments: Annotated[List[str], typer.Option()] = [
        "001_superconduct_tc_reg_dnn_opt_split_random",
        "004_superconduct_tc_reg_dnn_opt_weights_init",
        "005_superconduct_tc_reg_dnn_opt_dropout",
        "006_superconduct_tc_reg_dnn_opt_train_order",
        "014_superconduct_tc_reg_dnn_opt_split_bs",
        "015_superconduct_tc_reg_dnn_opt_split_cv",
        "002_superconduct_tc_reg_gbdt_opt_split_random",
        "003_superconduct_tc_reg_gbdt_opt_model_seed",
        "007_superconduct_tc_reg_gbdt_opt_train_order",
        "008_superconduct_tc_reg_gbdt_hopt_tpe",
        "009_superconduct_tc_reg_gbdt_subs_opt_split_r",
        "010_superconduct_tc_reg_gbdt_subsample_opt_model_seed",
        "011_superconduct_tc_reg_gbdt_subs_opt_order",
        "012_superconduct_tc_reg_gbdt_subs_opt_split_bs",
        "013_superconduct_tc_reg_gbdt_subs_opt_split_cv"
    ],
):
    if output_path.is_dir():
        raise ValueError("Output path must not be a directory!")
    
    exp_paths = [path_experiments / n for n in experiments]
    data = distributions.export_data(exp_paths)
    with open(output_path, "w") as f:
        json.dump(data, f)
    return data


@app.command("join-data")
def join_data(
    path_output: pathlib.Path,
    path_data_simpsons: pathlib.Path,
    path_data_lrl_sorb: pathlib.Path,
):
    if path_output.is_dir():
        raise ValueError("Output path must not be a directory!")
   
    data_1 = json.loads(path_data_simpsons.read_text())
    data_2 = json.loads(path_data_lrl_sorb.read_text())
   
    with open(path_output, "w") as f:
        json.dump(data_1 + data_2, f)


@app.command(hidden=True)
def _hidden():
    pass


if __name__ == "__main__":
    app()
