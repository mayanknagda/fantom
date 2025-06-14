import torch
import torch.nn as nn
from torch.distributions import Dirichlet


class DirPathwiseGrad(nn.Module):
    """
    A class for the Dirichlet pathwise gradient sampler.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, alpha: torch.tensordot) -> torch.Tensor:
        """
        Forward pass through the pathwise gradient sampler.

        Args:
            alpha (torch.Tensor): The input tensor.

        Returns:
            sample (torch.Tensor): The output tensor.
            kl_div (torch.Tensor): The KL divergence between the input and the prior.
        """
        sample = Dirichlet(alpha).rsample()
        return sample


class DirRSVI(nn.Module):
    """
    A class for Dirichlet RSVI (Rejection Sampling Variational Inference) sampler.
    ref: https://jmlr.org/papers/volume20/18-569/18-569.pdf
    """

    def __init__(self) -> None:
        super().__init__()

    def calc_epsilon(self, p, alpha):
        sqrt_alpha = torch.sqrt(9 * alpha - 3)
        powza = torch.pow(p / (alpha - 1 / 3), 1 / 3)
        return sqrt_alpha * (powza - 1)

    def gamma_h_boosted(self, epsilon, u, alpha):
        """
        Reparameterization for gamma rejection sampler with shape augmentation.
        """
        B = u.shape[0]
        K = alpha.shape[1]
        r = torch.arange(B, device=alpha.device)
        rm = torch.reshape(r, (-1, 1, 1)).float()
        alpha_vec = torch.tile(alpha, (B, 1)).reshape((B, -1, K)) + rm
        u_pow = torch.pow(u, 1.0 / alpha_vec) + 1e-10
        return torch.prod(u_pow, axis=0) * self.gamma_h(epsilon, alpha + B)

    def gamma_h(self, eps, alpha):
        b = alpha - 1 / 3
        c = 1 / torch.sqrt(9 * b)
        v = 1 + (eps * c)
        return b * (v**3)

    def rsvi(self, alpha):
        B = 10
        gam = torch.distributions.Gamma(alpha + B, 1).sample().to(alpha.device)
        eps = self.calc_epsilon(gam, alpha + B).detach().to(alpha.device)
        u = torch.rand((B, alpha.shape[0], alpha.shape[1]), device=alpha.device)
        doc_vec = self.gamma_h_boosted(eps, u, alpha)
        # normalize
        gam = doc_vec
        doc_vec = gam / torch.reshape(torch.sum(gam, dim=1), (-1, 1))
        z = doc_vec.reshape(alpha.shape)
        return z

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the RSVI sampler.

        Args:
            alpha (torch.Tensor): The input tensor.

        Returns:
            sample (torch.Tensor): The output tensor.
            kl_div (torch.Tensor): The KL divergence between the input and the prior.
        """
        sample = self.rsvi(alpha)
        return sample
