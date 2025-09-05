"""\nModel definitions and helpers for topic models.\n\nThis module is part of the `tomo` topic modeling library.\n"""

import torch
import torch.nn as nn
from .vae_helper import DirichletEncoder, Decoder, DirPathwiseGrad, DirRSVI, ETMDecoder
from ._vae import VAE
from ._scholar import SCHOLAR
from ._lda import LDA
from ._berttopic import BTopic
from ..optim._loss import KLDivLoss, NLLLoss


def get_model(train_texts=None, **kwargs):
    """
    Get the model class based on the model name.
    The elements of the model should be divided by a hyphen (-).
    """
    ## (model_type)-(labels, authors,etc)-(encoder-sampler-decoder)
    ## vae_based: encoder-sampler-decoder
    model_name = kwargs["model_name"]
    elements = model_name.split("-")
    kld_loss = KLDivLoss()
    nll_loss = NLLLoss()
    if elements[0] == "vae":
        vocab_size = len(kwargs["i2w"])
        ## encoder
        if "context" in model_name:
            en_in_features = vocab_size + kwargs["doc_emb_dim"]
        elif "llm" in model_name:
            en_in_features = kwargs["doc_emb_dim"]
        else:
            en_in_features = vocab_size
        if (
            (elements[-3] == "lin")
            or (elements[-3] == "context")
            or (elements[-3] == "llm")
        ):
            encoder = DirichletEncoder(
                in_features=en_in_features, num_topics=kwargs["num_topics"]
            )
        else:
            raise ValueError(f"Encoder not found for {elements[-3]}")

        if elements[-2] == "dir_pathwise":
            sampler = DirPathwiseGrad()
        elif elements[-2] == "dir_rsvi":
            sampler = DirRSVI()

        if elements[-1] == "lin":
            decoder = Decoder(in_features=kwargs["num_topics"], out_features=vocab_size)
        elif elements[-1] == "etm":
            decoder = ETMDecoder(
                in_features=kwargs["num_topics"],
                out_features=vocab_size,
                vocab_embeddings=kwargs["vocab_embeddings"],
            )
        else:
            raise ValueError(f"Decoder not found for {elements[-1]}")
        return VAE(
            encoder,
            sampler,
            decoder,
            nll_loss,
            kld_loss,
            **kwargs,
        )
    elif elements[0] == "scholar":
        vocab_size = len(kwargs["i2w"])
        ## encoder
        en_in_features = vocab_size + len(kwargs["label_names"])
        if elements[-3] == "lin":
            encoder = DirichletEncoder(
                in_features=en_in_features, num_topics=kwargs["num_topics"]
            )
        else:
            raise ValueError(f"Encoder not found for {elements[-3]}")

        if elements[-2] == "dir_pathwise":
            sampler = DirPathwiseGrad()
        elif elements[-2] == "dir_rsvi":
            sampler = DirRSVI()

        if elements[-1] == "lin":
            decoder = Decoder(in_features=kwargs["num_topics"], out_features=vocab_size)
        elif elements[-1] == "etm":
            decoder = ETMDecoder(
                in_features=kwargs["num_topics"],
                out_features=vocab_size,
                vocab_embeddings=kwargs["vocab_embeddings"],
            )
        else:
            raise ValueError(f"Decoder not found for {elements[-1]}")
        return SCHOLAR(
            encoder,
            sampler,
            decoder,
            nll_loss,
            kld_loss,
            **kwargs,
        )
    elif elements[0] == "lda":
        model = LDA(train_texts=train_texts, **kwargs)
        return model
    elif elements[0] == "bertopic":
        model = BTopic(train_texts=train_texts, **kwargs)
        return model
    else:
        raise ValueError(f"Model not found for {elements[0]}")
