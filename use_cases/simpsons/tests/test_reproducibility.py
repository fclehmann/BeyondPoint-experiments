import pathlib
import unittest
import uuid

import spock

import simpsons.config
import simpsons.cli
import logging
import socket


def is_on_taurus():
    return "taurus" in socket.gethostname()


class TestReproducibility(unittest.TestCase):

    def setUp(self) -> None:
        self.run_id = str(uuid.uuid4())
        results_parent_path = pathlib.Path("./results")
        results_parent_path.mkdir(parents=True, exist_ok=True)
        results_path = results_parent_path / self.run_id
        results_path.mkdir(exist_ok=False)
        self.results_path = results_path
        self.on_taurus = is_on_taurus()
        if self.on_taurus:
            self.config_path = './configs/optimal_config_taurus.yaml'
        else:
            self.config_path = './configs/example_config.yaml'

    def test_reproducibility_metric_1(self):
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logging.root.setLevel(logging.NOTSET)
        config = spock.SpockBuilder(
            simpsons.config.ExperimentConfig,
            desc="",
            configs=[self.config_path],
            no_cmd_line=True
        ).generate().ExperimentConfig

        object.__setattr__(config, "seed", 125)
        object.__setattr__(config, "save_results", False)
        object.__setattr__(config, "num_epochs", 30)

        uid = str(uuid.uuid4())
        metrics = []

        for i in range(5):
            metrics.append(simpsons.cli.train_simpsons_model(config, self.run_id))

        for metric in metrics[1:]:
            self.assertAlmostEqual(metrics[0], metric, places=5)

    def tearDown(self) -> None:
        self.results_path.rmdir()
