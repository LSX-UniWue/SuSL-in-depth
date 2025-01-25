from typing import Tuple

from torch import Tensor
from torch.nn import Module


class Reshape(Module):
    def __init__(self, shape: Tuple[int, ...]) -> None:
        super().__init__()
        self.__shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(self.__shape)
