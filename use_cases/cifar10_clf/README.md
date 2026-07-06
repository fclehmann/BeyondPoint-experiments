# CIFAR-10 classifier
Image classification using VGG11 trained on CIFAR-10. 

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
```bash
    python -m cifar10_clf.prepare_data ./data
```

## Train
The configuration is managed by spock framework. The default configuration is stored under `./configs`. You will have to override the output dir and data dir:


Launch train of VGG11 model overriding directories: 
```bash
python -m cifar10_clf.cli -c configs/optimal_config_cluster.yaml --ExperimentConfig.training_data_root=./data/processed --ExperimentConfig.out_path=./outputs
```

## References
- https://www.cs.toronto.edu/~kriz/cifar.html