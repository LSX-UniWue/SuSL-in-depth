from typing import Any

from lightning import LightningModule
from torch import Tensor
from torch.optim import Adam, Optimizer
from torchmetrics import MetricCollection

from .gmm_dgm import GaussianMixtureDeepGenerativeModel
from .losses import GaussianMixtureDeepGenerativeLoss


class LightningGMMModel(LightningModule):
    def __init__(
        self,
        model: GaussianMixtureDeepGenerativeModel,
        loss_fn: GaussianMixtureDeepGenerativeLoss,
        val_metrics: MetricCollection,
        test_metrics: MetricCollection,
        lr: float = 1e-3,
        loss_fn_step_step_size: float = 0.02,
        loss_fn_step_max_value: float = 1.0,
    ) -> None:
        super().__init__()
        self.__model = model
        self.__lr = lr
        self.__loss_fn = loss_fn
        self.__val_metrics = val_metrics
        self.__test_metrics = test_metrics
        self.__loss_fn_step_step_size = loss_fn_step_step_size
        self.__loss_fn_step_max_value = loss_fn_step_max_value

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.__model.parameters(), lr=self.__lr)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        pred = self.__model(batch)
        loss = self.__loss_fn(pred, None)
        self.log_dict({f"train_{k}": v for k, v in pred.items()}, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        loss = self.__metric_step(batch=batch, batch_idx=batch_idx, prefix="val")
        self.__val_metrics.update(self.__model.predict(batch["x_l"]), batch["y_l"])
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.__val_metrics.compute(), prog_bar=True)
        self.__val_metrics.reset()

    def test_step(self, batch: Any, batch_idx: int) -> Tensor:
        loss = self.__metric_step(batch=batch, batch_idx=batch_idx, prefix="test")
        self.__test_metrics.update(self.__model.predict(batch["x_l"]), batch["y_l"])
        return loss

    def __metric_step(self, batch: Any, batch_idx: int, prefix: str) -> Tensor:
        pred = self.__model(batch)
        loss = self.__loss_fn(pred, None)
        self.log_dict({f"{prefix}_{k}": v for k, v in pred.items()}, prog_bar=True)
        self.log(f"{prefix}_loss", loss, prog_bar=True)
        return loss

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.__test_metrics.compute(), prog_bar=True)
        self.__test_metrics.reset()

    def on_train_epoch_end(self) -> None:
        self.__loss_fn.step(step_size=self.__loss_fn_step_step_size, max_value=self.__loss_fn_step_max_value)
