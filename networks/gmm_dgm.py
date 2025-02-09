from abc import abstractmethod, ABC
from typing import Dict

from torch import Tensor, tensor, eye, arange, no_grad
from torch.distributions import Normal
from torch.nn import Module
from torch.nn.functional import cross_entropy, one_hot

from .variational_layer import VariationalLayer


class GaussianMixtureDeepGenerativeModel(Module, ABC):
    def __init__(
        self,
        n_y: int,
        n_z: int,
        n_x: int,
        q_y_x_module: Module,
        p_x_z_module: VariationalLayer,
        q_z_xy_module: VariationalLayer,
        p_z_y_module: VariationalLayer,
        markov_chain_samples: int = 1,
        log_priors: Tensor = None,
    ) -> None:
        super().__init__()
        self._markov_chain_samples = markov_chain_samples
        if log_priors is None:
            from torch import full

            self.register_buffer("log_priors", full(size=(n_y,), fill_value=1.0 / n_y).float().log())
        else:
            self.register_buffer("log_priors", log_priors)
        self._n_y = n_y
        self._n_z = n_z
        self._n_x = n_x
        self._q_y_x_module = q_y_x_module
        self._p_x_z_module = p_x_z_module
        self._q_z_xy_module = q_z_xy_module
        self._p_z_y_module = p_z_y_module

    def _forward_impl(
        self, x_u: Tensor, x_l: Tensor, y_l: Tensor, x_u_target: Tensor, x_l_target: Tensor
    ) -> Dict[str, Tensor]:
        # No unlabelled data?
        if x_u is None:
            elbo_u = tensor(0.0, device=x_l.device)
            reg_unlabelled = tensor(0.0, device=x_l.device)
        else:
            elbo_u = self._unlabelled_loss(x_u, x_u_target).mean()
            reg_unlabelled = self._unlabelled_regularisation(x_u).mean()
        # No labelled data?
        if x_l is None:
            elbo_l = tensor(0.0, device=x_u.device)
            reg_labelled = tensor(0.0, device=x_u.device)
        else:
            elbo_l = self._labelled_loss(x_l, y_l, x_l_target).mean()
            reg_labelled = self._labelled_regularisation(x_l, y_l).mean()

        return {
            "elbo_u": elbo_u,
            "reg_unlabelled": reg_unlabelled,
            "elbo_l": elbo_l,
            "reg_labelled": reg_labelled,
        }

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x_u = data.get("x_u", None)
        x_l = data.get("x_l", None)
        y_l = data.get("y_l", None)
        if y_l is not None:
            y_l = one_hot(y_l, self._n_y).float()
        x_u_target = data.get("x_u_target", x_u)
        x_l_target = data.get("x_l_target", x_l)
        return self._forward_impl(x_u=x_u, x_l=x_l, y_l=y_l, x_l_target=x_l_target, x_u_target=x_u_target)

    def predict(self, x: Tensor) -> Tensor:
        return self._q_y_x_module(x).softmax(dim=-1)

    @no_grad()
    def get_z_y_embedding(self, x: Tensor = None, y: Tensor = None) -> Tensor:
        if y is None:
            y = self.predict(x)
        self._p_z_y_module(y)
        return self._p_z_y_module.mean

    @no_grad()
    def get_z_xy_embedding(self, x: Tensor, y: Tensor = None) -> Tensor:
        if y is None:
            y = self.predict(x)
        self._q_z_xy_module(x, y)
        return self._q_z_xy_module.mean

    def _labelled_regularisation(self, x: Tensor, y: Tensor) -> Tensor:
        return -cross_entropy(target=y, input=self._q_y_x_module(x))

    @abstractmethod
    def _unlabelled_regularisation(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Method must be implemented in subclass.")

    def _labelled_loss(self, x: Tensor, y: Tensor, x_target: Tensor) -> Tensor:
        self._q_z_xy_module(x, y)
        # VAE reparam
        z = self._q_z_xy_module.rsample(sample_shape=(self._markov_chain_samples,))
        self._p_z_y_module(y)
        # dimension repeats, 2 for MLP, 4 for CNN
        repetitions = x.dim() * [1]
        x_ = x_target.unsqueeze(0).repeat(self._markov_chain_samples, *repetitions)
        y_ = y.unsqueeze(0).repeat(self._markov_chain_samples, 1, 1)
        return self._lower_bound(x_, y_, z)

    def _unlabelled_loss(self, x: Tensor, x_target: Tensor) -> Tensor:
        qy_l = self.predict(x)
        y_u = eye(self._n_y, device=qy_l.device)
        y_idx = arange(y_u.shape[0], device=qy_l.device).repeat_interleave(qy_l.shape[0])
        x_idx = arange(qy_l.shape[0], device=qy_l.device).repeat(y_u.shape[0])
        lb_u = self._labelled_loss(x[x_idx], y_u[y_idx], x_target[x_idx]).reshape(self._n_y, x.shape[0]).transpose(0, 1)
        lb_u = (qy_l * lb_u).sum(dim=-1)
        qy_entropy = (qy_l * (1e-10 + qy_l).log()).sum(dim=-1)
        return lb_u - qy_entropy

    def _lower_bound(self, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        self._p_x_z_module(z)
        # sum dims, -1 for MLP, -1,-2,-3 for CNN
        dim = [-i for i in range(1, x.ndim - 1)]
        l_px = self._p_x_z_module.log_prob(x).sum(dim=dim)
        l_py = (y * self.log_priors).sum(dim=-1)
        l_pz = self._p_z_y_module.log_prob(z).sum(dim=-1)
        l_qz = self._q_z_xy_module.log_prob(z).sum(dim=-1)
        return (l_px + l_py + l_pz - l_qz).mean(dim=0)


class EntropyRegularizedGaussianMixtureDeepGenerativeModel(GaussianMixtureDeepGenerativeModel):
    def __init__(
        self,
        n_y: int,
        n_z: int,
        n_x: int,
        q_y_x_module: Module,
        p_x_z_module: VariationalLayer,
        q_z_xy_module: VariationalLayer,
        p_z_y_module: VariationalLayer,
        markov_chain_samples: int = 1,
        log_priors: Tensor = None,
    ) -> None:
        super().__init__(
            n_y=n_y,
            n_z=n_z,
            n_x=n_x,
            q_y_x_module=q_y_x_module,
            p_x_z_module=p_x_z_module,
            q_z_xy_module=q_z_xy_module,
            p_z_y_module=p_z_y_module,
            markov_chain_samples=markov_chain_samples,
            log_priors=log_priors,
        )

    def _unlabelled_regularisation(self, x: Tensor) -> Tensor:
        return -cross_entropy(target=self.predict(x), input=self._q_y_x_module(x))


# L2 implementation from https://github.com/MatthewWilletts/GM-DGM
class L2RegularizedGaussianMixtureDeepGenerativeModel(GaussianMixtureDeepGenerativeModel):
    def __init__(
        self,
        n_y: int,
        n_z: int,
        n_x: int,
        q_y_x_module: Module,
        p_x_z_module: VariationalLayer,
        q_z_xy_module: VariationalLayer,
        p_z_y_module: VariationalLayer,
        markov_chain_samples: int = 1,
        log_priors: Tensor = None,
    ) -> None:
        super().__init__(
            n_y=n_y,
            n_z=n_z,
            n_x=n_x,
            q_y_x_module=q_y_x_module,
            p_x_z_module=p_x_z_module,
            q_z_xy_module=q_z_xy_module,
            p_z_y_module=p_z_y_module,
            markov_chain_samples=markov_chain_samples,
            log_priors=log_priors,
        )

    def _unlabelled_regularisation(self, x: Tensor) -> Tensor:
        from torch.nn import Linear

        res = tensor(0.0, device=x.device)
        normal_distribution = Normal(loc=0.0, scale=1.0)
        for layer in filter(lambda z: isinstance(z, Linear), self.modules()):
            res += normal_distribution.log_prob(layer.weight).sum()
        return res
