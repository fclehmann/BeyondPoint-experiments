# Simpsons classifier
Train pipeline of simpsons image classifier.

## Install
You will need python version between 3.9 and 3.11 (inclusive).
1. Create venv and install dependencies: 
    ```bash
    poetry install
    ```
2. Activate venv:
    ```bash
    poetry shell
    ```

## Prepare data
1. Download data
    ```bash
    chmod +x ./scripts/download_data.sh; ./scripts/download_data.sh ./data
    ```
2. Process data
    ```bash
    python -m simpsons.process_data ./data
    ```

## Train
**Note: The last revision where simpsons train pipeline is guaranteed to work is `7f22022`**. So it might be necessary to execute these preliminary steps:
1. Checkout the last working version:
    ```
    git checkout 7f22022
    ```
2. Reinstall dependencies:
    ```
    poetry install
    ```

The configuration is managed by spock framework. The default configuration is stored under `./configs`. You have to override the output dir and data dir:


Train VGG16 model overriding directories: 
```bash
python -m simpsons.cli -c configs/optimal_config_taurus-vgg16.yaml --ExperimentConfig.training_data_root=./data/processed --ExperimentConfig.out_path=./outputs
```

## References
- https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset/versions/3