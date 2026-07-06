#!/bin/bash

module load release/23.04 GCCcore/10.3.0 Python/3.9.5 GCC/10.3.0

source /data/horse/ws/s4610340-isml/dev/statistics-in-machine-learning/tools/experiment_runner/.venv/bin/activate

for experiment_path in /data/horse/ws/s4610340-isml/simpsons/experiments/*; do
    

    if [[ $experiment_path == *"_hopt_"* ]]; then
        workers=32
        mem="32GB"
        time="24:00:00"
    else
        workers=16
        mem="16GB"
        time="00:10:00"
    fi

    python -m experiment_runner process $experiment_path \
        -m val_macro_F1 \
        -m val_loss \
        -m val_accuracy \
        --as_job \
        --time $time \
        --workers $workers \
        --mem $mem
done