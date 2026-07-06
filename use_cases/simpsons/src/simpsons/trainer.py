import logging
import typing

import numpy as np
import sklearn.metrics
import torch
import torch.optim.lr_scheduler
import torch.utils.data
import torchvision

import isml_train.trainer
import isml_train.structures

from simpsons.models.simpsons_net_conv import SimpsonsNetConv

import isml_train.data_loader_creator

_logger = logging.getLogger(__name__)

_O = typing.TypeVar("_O", bound=torch.optim.Optimizer)
_N = typing.TypeVar("_N", SimpsonsNetConv, torchvision.models.vgg.VGG)


class SimpsonsClassifierModelTrainer(isml_train.trainer.AbstractClassifierModelTrainer[_O, _N]):

    def train_batch(self, batch: torch.Tensor, batch_id: int) -> torch.Tensor:
        train_features, train_labels = batch
        # _logger.info(f"Batch id: {batch_id}")
        # _logger.info(f"Free GPU memory before train batch on gpu: {torch.cuda.mem_get_info()[0]*10e-6}")
        train_features: torch.Tensor = train_features.to(self._device)
        # _logger.info(f"Free GPU memory after train batch on gpu: {torch.cuda.mem_get_info()[0] * 10e-6}")
        train_labels: torch.Tensor = train_labels.to(self._device)
        # _logger.info(f"Free GPU memory after train and labels batch on gpu: {torch.cuda.mem_get_info()[0] * 10e-6}")
        # Forward pass
        train_outputs: torch.Tensor = self._nn_model.forward(train_features)
        # _logger.info(f"Free GPU memory after forward on gpu: {torch.cuda.mem_get_info()[0] * 10e-6}")
        return self._loss_f(train_outputs, train_labels)

    def train_epoch(self, train_data_loader: torch.utils.data.DataLoader) -> float:

        self._nn_model.train(True)

        running_loss = 0.0

        for i, batch in enumerate(train_data_loader):
            train_loss = self.train_batch(batch, i)

            # Backward and optimize
            train_loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()

            running_loss += float(train_loss.item())

            del train_loss

        return running_loss

    def calculate_metrics_over_dataset(
            self,
            data_loader: torch.utils.data.DataLoader,
            device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        preds_batched: list[torch.Tensor] = []
        labels_batched: list[torch.Tensor] = []

        loss = 0.0

        for features_batch, labels_batch in data_loader:
            features_batch: torch.Tensor = features_batch.to(device)  # (B, C, H, W)
            labels_batch: torch.Tensor = labels_batch.to(device)  # (B,)

            outputs_batch = self._nn_model.forward(features_batch)  # (B, N_c)

            loss += float(self._loss_f(outputs_batch, labels_batch).item())

            class_probs, class_idx_batch = torch.max(input=outputs_batch, dim=1)

            preds_batched.append(class_idx_batch.detach().cpu())
            labels_batched.append(labels_batch.detach().cpu())

            del features_batch, outputs_batch, labels_batch

        preds_dataset = torch.hstack(preds_batched)
        labels_dataset = torch.hstack(labels_batched)
        acc = int((preds_dataset == labels_dataset).sum()) / len(labels_dataset)
        return preds_dataset, labels_dataset, loss, acc

    def evaluate(
            self,
            data_loader: torch.utils.data.DataLoader,
    ) -> isml_train.structures.EvalResult:
        self._nn_model.train(False)
        with torch.no_grad():
            preds, labels, loss, acc = self.calculate_metrics_over_dataset(
                data_loader,
                device=self._device
            )
            conf_matrix: np.ndarray = sklearn.metrics.confusion_matrix(
                y_true=labels,
                y_pred=preds,
                labels=list(self._class_to_indexes.values())
            )
            return {
                "loss": loss,
                "acc": acc,
                "confusion_matrix": conf_matrix.tolist()
            }

    def train(
            self,
            train_data_loader: torch.utils.data.DataLoader,
            test_data_loader: torch.utils.data.DataLoader,
    ) -> isml_train.structures.TrainMetrics:

        reports = []

        for epoch in range(self._n_epochs):

            _logger.info(f"Training epoch {epoch} ...")

            train_loss = self.train_epoch(train_data_loader=train_data_loader)

            if self._lr_scheduler is not None:
                self._lr_scheduler.step()

            _logger.info(f"Train loss in epoch={epoch} (sum): {train_loss}")

            torch.cuda.empty_cache()

            if (epoch + 1) % self._show_step == 0:
                report = {
                    "epoch": epoch + 1,
                    "train": self.evaluate(train_data_loader),
                    "val": self.evaluate(test_data_loader)
                }
                _logger.info(
                    f'Epoch: {epoch + 1}, '
                    f'train loss = {train_loss:.4f}, '
                    f'train acc: ## {100 * report["train"]["acc"]:.2f}, '
                    f'val loss: {report["val"]["loss"]:.4f}, '
                    f'val acc: ** {100 * report["val"]["acc"]:.2f}%'
                )
                reports.append(report)

        return reports
