import torch
import torch.nn as nn
from torch.distributions import kl_divergence
from abc import ABC, abstractmethod


class Loss(ABC, nn.Module):
    """
    An abstract class for the loss function.
    """

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass through the loss function.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: The output tensor.
        """
        raise NotImplementedError


class NLLLoss(Loss):
    """
    A class for the negative log likelihood loss.
    """

    def __init__(self) -> None:
        """Initialize the NLL loss."""
        super().__init__()

    def forward(self, bow: torch.Tensor, bow_recon: torch.Tensor) -> torch.Tensor:
        """Forward pass through the NLL loss.

        Args:
            bow (torch.Tensor): The input BoW tensor.
            bow_recon (torch.Tensor): The reconstructed BoW tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        bow_recon = bow_recon + 1e-6
        return -torch.sum(bow * bow_recon, dim=1).mean()


class KLDivLoss(Loss):
    """
    A class for the KL divergence loss.
    """

    def __init__(self) -> None:
        """Initialize the KL divergence loss."""
        super().__init__()

    def forward(self, dist, prior_dist) -> torch.Tensor:
        """Forward pass through the KL divergence loss.

        Args:
            dist: The distribution.
            prior_dist: The prior distribution.

        Returns:
            torch.Tensor: The output tensor.
        """
        return kl_divergence(dist, prior_dist).mean()
