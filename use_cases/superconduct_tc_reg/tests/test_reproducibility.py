import os
import pathlib
import unittest
import uuid

import hydra
import hydra.core.hydra_config
import mlflow

import superconduct_tc_reg.train
import logging
import socket


class TestReproducibility(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        logging.root.setLevel(logging.DEBUG)
        hydra.initialize(
            version_base="1.3",
            config_path="../configs", # type: ignore
        )

    def setUp(self) -> None:
        pass

    def test_reproducibility_dnn(self):
        # Init hydra config
        cfg = hydra.compose(
            config_name="train", 
            overrides=[
                "tracking=filesystem_cluster",
                "pipeline.trainer.max_epochs=5",
                "pipeline.do_checkpointing=False",
                "seed=1",
                "pipeline=nn",
                "pipeline.trainer.deterministic=True",
            ])
        workdir = f"{cfg.out_path}/{cfg.run_uid}"
        os.mkdir(workdir)
        os.chdir(workdir)


        metrics = []

        for i in range(10):
            val_loss = superconduct_tc_reg.train.train(cfg)
            metrics.append(val_loss)

        print("test_reproducibility_dnn", metrics)
        for metric in metrics[1:]:
            self.assertEqual(metrics[0], metric)

    def test_reproducibility_gbdt(self):
        # Init hydra config
        cfg = hydra.compose(
            config_name="train", 
            overrides=[
                "tracking=filesystem_cluster",
                "seed=1",
                "pipeline=xgb"
            ])
        workdir = f"{cfg.out_path}/{cfg.run_uid}"
        os.mkdir(workdir)
        os.chdir(workdir)

        metrics = []

        for i in range(10):
            val_loss = superconduct_tc_reg.train.train(cfg)
            metrics.append(val_loss)

        print("test_reproducibility_gbdt", metrics)
        for metric in metrics[1:]:
            self.assertEqual(metrics[0], metric)
