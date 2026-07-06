import pathlib
import unittest
import uuid

import mlflow
import spock

import cifar10_clf.config
import cifar10_clf.cli
import cifar10_clf.train
import logging
import socket


def is_on_taurus():
    return "taurus" in socket.gethostname()


class TestReproducibility(unittest.TestCase):

    def setUp(self) -> None:
        current_path = pathlib.Path(__file__).parent.resolve()
        self.config_path = current_path / "configs" / "example_config_cluster.yaml"
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        logging.root.setLevel(logging.DEBUG)
        logging.getLogger("PIL.PngImagePlugin").setLevel("WARNING")

    def test_reproducibility_no_dropout(self):
        config = (
            spock.SpockBuilder(
                cifar10_clf.config.ExperimentConfig,
                desc="",
                configs=[self.config_path],
                no_cmd_line=True,
            )
            .generate()
            .ExperimentConfig
        )

        object.__setattr__(config, "dropout_p", 0.0)

        metrics = []

        for i in range(10):
            val_metric = cifar10_clf.train.train(config)
            metrics.append(val_metric)

        for v in metrics[1:]:
            print(f"{v} ?= {metrics[0]}")
            self.assertEqual(metrics[0], v)

    def test_reproducibility_with_dropout(self):
        config = (
            spock.SpockBuilder(
                cifar10_clf.config.ExperimentConfig,
                desc="",
                configs=[self.config_path],
                no_cmd_line=True,
            )
            .generate()
            .ExperimentConfig
        )

        object.__setattr__(config, "dropout_p", 0.1)

        metrics = []

        for i in range(10):
            val_metric = cifar10_clf.train.train(config)
            metrics.append(val_metric)

        for v in metrics[1:]:
            print(f"{v} ?= {metrics[0]}")
            self.assertEqual(metrics[0], v)

    def test_reproducibility_tf32(self):
        config = (
            spock.SpockBuilder(
                cifar10_clf.config.ExperimentConfig,
                desc="",
                configs=[self.config_path],
                no_cmd_line=True,
            )
            .generate()
            .ExperimentConfig
        )

        object.__setattr__(config, "tf32", True)

        metrics = []

        for i in range(3):
            val_metric = cifar10_clf.train.train(config)
            metrics.append(val_metric)

        for v in metrics[1:]:
            print(f"{v} ?= {metrics[0]}")
            self.assertAlmostEqual(metrics[0], v, delta=1e-4)
