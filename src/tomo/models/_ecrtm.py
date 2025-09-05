"""\nModel definitions and helpers for topic models.\n\nThis module is part of the `tomo` topic modeling library.\n"""

import torch
import torch.nn as nn


def get_ecr_cost(topic_embeddings, word_embeddings):
    """\n    Function `get_ecr_cost`.\n\n    Args:\n    topic_embeddings: Description.\n    word_embeddings: Description.\n\n    Returns: Description.\n"""
    cost = pairwise_euclidean_distance(topic_embeddings, word_embeddings)
    return cost


def pairwise_euclidean_distance(x, y):
    """\n    Function `pairwise_euclidean_distance`.\n\n    Args:\n    x: Description.\n    y: Description.\n\n    Returns: Description.\n"""
    cost = (
        torch.sum(x**2, axis=1, keepdim=True)
        + torch.sum(y**2, dim=1)
        - 2 * torch.matmul(x, y.t())
    )
    return cost


class ECR(nn.Module):
    """\n    Class `ECR`.\n\n    Args:\n    nn.Module: Description.\n\n    Returns: Description.\n"""

    def __init__(
        self, weight_loss_ECR=250, sinkhorn_alpha=20, OT_max_iter=1000, stopThr=0.5e-2
    ):
        """\n        Function `__init__`.\n    \n        Args:\n        weight_loss_ECR: Description.\n        sinkhorn_alpha: Description.\n        OT_max_iter: Description.\n        stopThr: Description.\n    \n        Returns: Description.\n"""
        super().__init__()

        self.sinkhorn_alpha = sinkhorn_alpha
        self.OT_max_iter = OT_max_iter
        self.weight_loss_ECR = weight_loss_ECR
        self.stopThr = stopThr
        self.epsilon = 1e-16

    def forward(self, M):
        """\n        Function `forward`.\n    \n        Args:\n        M: Description.\n    \n        Returns: Description.\n"""
        # M: KxV
        # a: Kx1
        # b: Vx1
        device = M.device

        # Sinkhorn's algorithm
        a = (torch.ones(M.shape[0]) / M.shape[0]).unsqueeze(1).to(device)
        b = (torch.ones(M.shape[1]) / M.shape[1]).unsqueeze(1).to(device)

        u = (torch.ones_like(a) / a.size()[0]).to(device)  # Kx1

        K = torch.exp(-M * self.sinkhorn_alpha)
        err = 1
        cpt = 0
        while err > self.stopThr and cpt < self.OT_max_iter:
            v = torch.div(b, torch.matmul(K.t(), u) + self.epsilon)
            u = torch.div(a, torch.matmul(K, v) + self.epsilon)
            cpt += 1
            if cpt % 50 == 1:
                bb = torch.mul(v, torch.matmul(K.t(), u))
                err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float("inf"))

        transp = u * (K * v.T)

        loss_ECR = torch.sum(transp * M)
        loss_ECR *= self.weight_loss_ECR

        return loss_ECR
