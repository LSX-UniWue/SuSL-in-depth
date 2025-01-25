from typing import Any, Tuple

from torch import Tensor
from torchmetrics import Metric
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.functional.clustering.utils import calculate_contingency_matrix
from torchmetrics.utilities import dim_zero_cat


class ClusterMetric(Metric):
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            **kwargs,
        )
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        if preds.ndim == 2:
            self.preds.append(preds.argmax(dim=-1))
        else:
            self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Tuple[Tensor, Tensor]:
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        # Get mode for each cluster -> dim=0
        modes = calculate_contingency_matrix(preds=preds, target=target).argmax(dim=0)
        _, idx = preds.unique(sorted=True, return_inverse=True)
        return modes[idx], target


class ClusterAccuracy(MulticlassAccuracy):
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            **kwargs,
        )
        self.__cluster_metric = ClusterMetric()

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.__cluster_metric.update(preds, target)

    def compute(self) -> Tensor:
        preds, target = self.__cluster_metric.compute()
        super().update(preds=preds, target=target)
        return super().compute()
