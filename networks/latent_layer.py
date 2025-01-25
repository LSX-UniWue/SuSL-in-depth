from torch import Tensor, hstack
from torch.nn import Module, Identity


class LatentLayer(Module):
    DEFAULT_INIT_STD = 0.001

    def __init__(self, pre_module: Module, post_module: Module = Identity()) -> None:
        super().__init__()
        self.__pre_module = pre_module
        self.__post_module = post_module

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        latent = self.__pre_module(x)
        return self.__post_module(hstack((latent, y)))


class WillettsLatentLayer(LatentLayer):
    def __init__(self, module: Module) -> None:
        super().__init__(pre_module=Identity(), post_module=module)
