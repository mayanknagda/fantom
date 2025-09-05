"""\nModel definitions and helpers for topic models.\n\nThis module is part of the `tomo` topic modeling library.\n"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    A class for the VAE topic model decoder.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize the base decoder.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features (usually the vocab size).
        """
        super().__init__()
        self.dec_base = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(
            out_features, eps=0.001, momentum=0.001, affine=True
        )
        self.batch_norm.weight.data.copy_(torch.ones(out_features))
        self.batch_norm.weight.requires_grad = False
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through the decoder.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.log_softmax(self.batch_norm(self.dec_base(x)))

    def get_topics(self, **kwargs) -> torch.Tensor:
        """Get the topics from the decoder.

        Returns:
            torch.Tensor: The topics.
        """
        return self.dec_base.weight.detach().cpu().numpy().T


class ETMDecoder(nn.Module):
    """
    A class for the ETM topic model decoder.
    """

    def __init__(
        self, in_features: int, out_features: int, vocab_embeddings: torch.Tensor
    ) -> None:
        """Initialize the ETM decoder.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features (usually the vocab size).
        """
        super().__init__()
        self.topic_embeddings = nn.Linear(
            in_features, vocab_embeddings.shape[1], bias=False
        )
        self.word_embeddings = nn.Linear(
            vocab_embeddings.shape[1], out_features, bias=False
        )
        self.word_embeddings.weight.data.copy_(vocab_embeddings)
        self.decoder_norm = nn.BatchNorm1d(
            num_features=out_features, eps=0.001, momentum=0.001, affine=True
        )
        self.decoder_norm.weight.data.copy_(torch.ones(out_features))
        self.decoder_norm.weight.requires_grad = False

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through the decoder.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        topic_embeddings = self.topic_embeddings(x)
        word_embeddings = self.word_embeddings.weight
        x_recon = torch.matmul(
            topic_embeddings, word_embeddings.T
        )  # (batch_size, vocab_size)
        x_recon = torch.nn.functional.log_softmax(
            self.decoder_norm(x_recon), dim=1
        )  # (batch_size, vocab_size)
        return x_recon

    def get_topics(self, **kwargs) -> torch.Tensor:
        """Get the topics from the decoder.

        Returns:
            torch.Tensor: The topics.
        """
        topic_embeddings = self.topic_embeddings.weight.data.cpu().numpy().T  # (K, E)
        word_embeddings = self.word_embeddings.weight.data.cpu().numpy().T  # (E, V)
        topics = topic_embeddings @ word_embeddings  # (K, V)
        return topics


class SharedETMDecoder(nn.Module):
    """
    A class for the ETM topic model decoder.
    """

    def __init__(self, out_features: int) -> None:
        """Initialize the ETM decoder.

        Args:
            out_features (int): The number of output features (usually the vocab size).
        """
        super().__init__()
        author_embeddings = torch.zeros(out_features, 300)
        self.author_embeddings = nn.Linear(
            author_embeddings.shape[1], out_features, bias=False
        )
        self.author_embeddings.weight.data.copy_(author_embeddings)
        self.decoder_norm = nn.BatchNorm1d(
            num_features=out_features, eps=0.001, momentum=0.001, affine=True
        )
        self.decoder_norm.weight.data.copy_(torch.ones(out_features))
        self.decoder_norm.weight.requires_grad = False

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through the decoder.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        topic_embeddings = kwargs["te"](x)
        author_embeddings = self.author_embeddings.weight
        x_recon = torch.matmul(
            topic_embeddings, author_embeddings.T
        )  # (batch_size, vocab_size)
        x_recon = torch.nn.functional.log_softmax(
            self.decoder_norm(x_recon), dim=1
        )  # (batch_size, vocab_size)
        return x_recon

    def get_topics(self, **kwargs) -> torch.Tensor:
        """Get the topics from the decoder.

        Returns:
            torch.Tensor: The topics.
        """
        topic_embeddings = kwargs["te"].weight.data.cpu().numpy().T  # (K, E)
        author_embeddings = self.author_embeddings.weight.data.cpu().numpy().T  # (E, V)
        topics = topic_embeddings @ author_embeddings  # (K, V)
        return topics
