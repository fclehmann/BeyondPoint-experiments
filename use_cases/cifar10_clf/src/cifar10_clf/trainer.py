import logging
from typing import Any, Callable, Literal, Optional
from git import Union
import mlflow
import pandas as pd
import torch.utils.data
import isml_train.trainer
import isml_train.structures
import torchmetrics
import tqdm
import lightning.pytorch.loggers
from cifar10_clf.config import ExperimentConfig
from cifar10_clf.tracking import TrackingProtocol

_logger = logging.getLogger(__name__)


class Trainer:

    def __init__(
        self,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        show_step: int,
        metrics_val: torchmetrics.MetricCollection,
        metrics_test: torchmetrics.MetricCollection,
        metrics_train: torchmetrics.MetricCollection,
        config: ExperimentConfig,
        tracking_logger: TrackingProtocol,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        self._device = device
        self._show_step = show_step
        self._lr_scheduler = lr_scheduler
        self._loss = loss
        self._optimizer = optimizer
        self._metrics: dict[str, torchmetrics.MetricCollection] = {
            "train": metrics_train,
            "val": metrics_val,
            "test": metrics_test,
        }
        self._config = config
        self._tracking_logger = tracking_logger

    def replace_metrics(
        self,
        metrics: torchmetrics.MetricCollection,
        stage: Literal["train", "val", "test"],
    ):
        self._metrics[stage] = metrics

    def _log_metrics(self, metrics: dict[str, Any], step: int):
        for k, v in metrics.items():
            if type(v) is torch.Tensor:
                if len(v.shape) == 0:
                    self._tracking_logger.log_metrics({k: v.item()}, step)
                else:
                    raise ValueError(
                        "Logging of tensors with len(shape) > 0 is not supported."
                    )
            elif type(v) is pd.DataFrame:
                self._tracking_logger.log_table(
                    data=v.reset_index(), artifact_file=f"{k}-{step}.json"
                )

    def _forward_process_batch(
        self, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        train_features, train_labels = batch
        train_features: torch.Tensor = train_features.to(self._device)

        # (B,) true class indexes
        train_labels: torch.Tensor = train_labels.to(self._device)

        # (B, C) logits
        model_outputs = self._model.forward(train_features)

        return train_features, model_outputs, train_labels

    def train_batch(self, batch: torch.Tensor) -> float:
        self._optimizer.zero_grad()

        _, train_outputs, train_labels = self._forward_process_batch(batch)

        loss = self._loss(train_outputs, train_labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)  # type: ignore
        self._optimizer.step()

        with torch.no_grad():
            self._metrics["train"](train_outputs, train_labels)

        return loss.detach().item()

    def train_epoch(self, data_loader: torch.utils.data.DataLoader, epoch: int):
        self._model.train()

        loss = 0.0
        n_batches = 0

        for batch in data_loader:
            loss += self.train_batch(batch)
            n_batches += 1

        loss /= n_batches

        self._tracking_logger.log_metrics({"train_loss": loss}, epoch)
        metrics_train = self._metrics["train"].compute()
        self._log_metrics(metrics_train, epoch)

        self._metrics["train"].reset()

        if self._lr_scheduler is not None:
            if type(self._lr_scheduler) is torch.optim.lr_scheduler.ReduceLROnPlateau:
                self._lr_scheduler.step(loss)
            else:
                self._lr_scheduler.step()

            self._last_lr = self._lr_scheduler._last_lr[0]  # type: ignore
            self._tracking_logger.log_metrics({"lr": self._last_lr}, epoch)

        return loss, metrics_train

    @torch.no_grad()
    def val_step(self, batch: torch.Tensor, stage: Literal["val", "test"]) -> None:
        _, train_outputs, train_labels = self._forward_process_batch(batch)
        self._metrics[stage](train_outputs, train_labels)

    @torch.no_grad()
    def val_epoch(
        self,
        data_loader: torch.utils.data.DataLoader,
        epoch: int,
        stage: Literal["val", "test"],
        model: Optional[torch.nn.Module] = None,
    ) -> dict[str, float]:
        assert self._model is not None or model is not None

        if model is not None:
            self._model = model

        self._model.eval()

        for batch in data_loader:
            self.val_step(batch, stage)

        # Compute metrics
        metrics = self._metrics[stage].compute()
        self._log_metrics(metrics, epoch)
        self._metrics[stage].reset()

        return metrics

    def _build_progress_bar_description(
        self,
        loss_train: float,
        metrics_val: dict[str, Any],
        metrics_train: dict[str, Any],
    ):
        train_part = f"loss: {loss_train:.2f}; acc: {metrics_train['train_MulticlassAccuracy']:.2g}"
        val_part = f"val_acc: {metrics_val['val_MulticlassAccuracy']:.2g}"
        lr_part = f"" if self._lr_scheduler is None else f"lr: {self._last_lr:.2g}"
        return f"{train_part}; {val_part}; {lr_part}"

    @torch.no_grad()
    def _log_model(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        epoch: int,
    ):
        self._tracking_logger.log_model(model, f"models/model-epoch={epoch}.pt", None)

    def fit(
        self,
        train_data_loader: torch.utils.data.DataLoader,
        val_data_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
    ):
        self._model = model

        if self._config.compile_model:
            _logger.debug("Model will be compiled.")
            self._model = torch.compile(model, dynamic=False)

        if self._config.tf32:
            _logger.info("Enabling tf32 precision")
            # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
            # in PyTorch 1.12 and later.
            torch.backends.cuda.matmul.allow_tf32 = True
            # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        progress_bar = tqdm.tqdm(range(self._config.num_epochs), unit="epoch")

        for epoch in progress_bar:
            loss_train, metrics_train = self.train_epoch(
                data_loader=train_data_loader, epoch=epoch
            )
            metrics_val = self.val_epoch(
                data_loader=val_data_loader, epoch=epoch, stage="val"
            )

            progress_bar.set_description(
                self._build_progress_bar_description(
                    loss_train, metrics_val, metrics_train
                )
            )

        if (
            self._config.save_model
            and self._config.model_checkpoint_strategy == "last_epoch"
        ):
            self._log_model(model, train_data_loader, epoch=epoch)

        return metrics_val

    @torch.no_grad()
    def test(
        self,
        data_loader: torch.utils.data.DataLoader,
        model: Optional[torch.nn.Module] = None,
    ) -> dict[str, float]:
        return self.val_epoch(
            data_loader=data_loader,
            epoch=self._config.num_epochs,
            stage="test",
            model=model,
        )
