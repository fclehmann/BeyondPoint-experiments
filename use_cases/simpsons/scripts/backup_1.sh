#!/bin/bash

module load release/23.04 GCCcore/10.3.0 Python/3.9.5 GCC/10.3.0

source /data/horse/ws/s4610340-isml/dev/statistics-in-machine-learning/tools/experiment_runner/.venv/bin/activate

python -m experiment_runner backup \
 -o /data/horse/ws/s4610340-isml/archive \
 -e "/data/horse/ws/s4610340-isml/simpsons/experiments/10_simpsons_hopt_tpe_SGDwlrs_seeds/10-simpsons-hopt-tpe-SGDwlrs-seeds.toml" \
 -e "/data/horse/ws/s4610340-isml/simpsons/experiments/11_simpsons_hopt_tpe_AdamW_seeds/11-simpsons-hopt-tpe-AdamW-seeds.toml" \
 -e "/data/horse/ws/s4610340-isml/simpsons/experiments/12_simpsons_hopt_tpe_Adamax_seeds/12-simpsons-hopt-tpe-Adamax-seeds.toml" \
 -e "/data/horse/ws/s4610340-isml/simpsons/experiments/13_simpsons_hopt_tpe_Adam_seeds/13-simpsons-hopt-tpe-Adam-seeds.toml" \
 -e "/data/horse/ws/s4610340-isml/simpsons/experiments/15_simpsons_hopt_tpe_RMSprop_seeds/15-simpsons-hopt-tpe-RMSprop-seeds.toml" \
 -rp "*.json" \
 -rp "*.yaml" \
 -hp "omniopt/projects/*/csv/*.csv" \
 -c "xz" \
 --lazy \
 --resolve_path_from_ec \
 --as_job