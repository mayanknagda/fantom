import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC


class Encoder(nn.Module, ABC):
    """
    Initializing a base class for the VAE encoder.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize the base encoder.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
        """
        super().__init__()
        self.enc_base = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout(0.25),
        )


class DirichletEncoder(Encoder):
    """
    A VAE encoder for Topic Models.
    """

    def __init__(self, in_features: int, num_topics: int) -> None:
        """
        Initialize the encoder.

        The encoder is based on: https://jmlr.org/papers/volume20/18-569/18-569.pdf

        Args:
            in_features (int): The number of input features.
            num_topics (int): The number of topics.
        """
        super().__init__(in_features, 512)

        self.net = nn.Linear(512, num_topics)
        self.batch_norm = nn.BatchNorm1d(
            num_topics, eps=0.001, momentum=0.001, affine=True
        )
        self.batch_norm.weight.data.copy_(torch.ones(num_topics))
        self.batch_norm.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            alpha (torch.Tensor): The dirichlet parameters.
        """
        alpha = torch.max(
            F.softplus(self.batch_norm(self.net(self.enc_base(x)))),
            torch.tensor(1e-5, device=x.device),
        )
        return alpha
