from typing import Dict

from torch import Tensor, tensor
from torch.nn import Module


class GaussianMixtureDeepGenerativeLoss(Module):
    def __init__(self, alpha: float = 1.1, gamma: float = 0.5, reduction: str = "mean") -> None:
        super().__init__()
        self.__reduction = reduction
        self._alpha = alpha
        self._gamma = gamma

    def step(self, step_size: float = 0.02, max_value: float = 1.0) -> None:
        pass

    def forward(self, y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor] = None) -> Tensor:
        tmp = (
            y_pred.get("elbo_u", tensor(0.0))
            + y_pred.get("elbo_l", tensor(0.0))
            + self._alpha * y_pred.get("reg_labelled", tensor(0.0))
            + self._gamma * y_pred.get("reg_unlabelled", tensor(0.0))
        )

        match self.__reduction:
            case "mean":
                return -tmp.mean()
            case "sum":
                return -tmp.sum()
            case _:
                return -tmp


class EntropyGaussianMixtureDeepGenerativeLoss(GaussianMixtureDeepGenerativeLoss):
    def __init__(self, alpha: float = 1.1, gamma: float = 0.5, reduction: str = "mean") -> None:
        super().__init__(alpha=alpha, gamma=gamma, reduction=reduction)
        self.__temperature = 1e-5
        self.__original_gamma = gamma
        self._gamma = self.__temperature * gamma

    def step(self, step_size: float = 0.02, max_value: float = 1.0) -> None:
        self.__temperature = min(max_value, self.__temperature + step_size)
        self._gamma = self.__temperature * self.__original_gamma
