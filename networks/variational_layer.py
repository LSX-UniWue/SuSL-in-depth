from torch import Tensor
from torch.distributions import Distribution, Normal, Bernoulli
from torch.nn import Module, Linear


class VariationalLayer(Module, Distribution):
    def __init__(self, feature_extractor: Module) -> None:
        Module.__init__(self)
        self.__feature_extractor = feature_extractor

    def forward(self, x: Tensor, y: Tensor = None) -> None:
        if y is None:
            latent = self.__feature_extractor(x)
        else:
            latent = self.__feature_extractor(x, y)
        return latent


class GaussianVariationalLayer(VariationalLayer, Normal):
    def __init__(self, feature_extractor: Module, module_init=Linear, **kwargs) -> None:
        VariationalLayer.__init__(self, feature_extractor)
        Normal.__init__(self, loc=0, scale=1)
        self.__mean = module_init(**kwargs)
        self.__log_var = module_init(**kwargs)

    def forward(self, x: Tensor, y: Tensor = None) -> None:
        latent = VariationalLayer.forward(self, x, y)
        Normal.__init__(self, loc=self.__mean(latent), scale=self.__log_var(latent).exp().sqrt())


class BernoulliVariationalLayer(VariationalLayer, Bernoulli):
    def __init__(self, feature_extractor: Module, module_init=Linear, **kwargs) -> None:
        VariationalLayer.__init__(self, feature_extractor)
        Bernoulli.__init__(self, logits=0)
        self.__logits = module_init(**kwargs)

    def forward(self, x: Tensor, y: Tensor = None) -> None:
        latent = VariationalLayer.forward(self, x, y)
        Bernoulli.__init__(self, logits=self.__logits(latent))
