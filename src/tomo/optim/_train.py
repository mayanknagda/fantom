"""\nOptimization helpers (training loops, loss functions).\n\nThis module is part of the `tomo` topic modeling library.\n"""

import os
import torch
import torch.nn as nn
from typing import Callable, Dict, Any, Union, Tuple, List, Optional
from abc import ABC, abstractmethod


class Trainer(ABC):
    """
    A general trainer class for training a model.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dl: torch.utils.data.DataLoader,
        val_dl: torch.utils.data.DataLoader,
        optimizer: Callable = torch.optim.Adam,
        device: Union[str, torch.device] = "cpu",
        exp_path: Optional[str] = None,
    ) -> None:
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.optimizer = optimizer
        self.model = model
        self.device = device
        self.exp_path = exp_path
        self.best_val_loss = float("inf")

    def _save_model(self, **kwargs):
        """\n        Function `_save_model`.\n    \n        Returns: Description.\n"""
        torch.save(self.model.state_dict(), os.path.join(self.exp_path, "model.pt"))

    @abstractmethod
    def _train_one_epoch(self, **kwargs):
        """\n        Function `_train_one_epoch`.\n    \n        Returns: Description.\n"""
        pass

    @abstractmethod
    def _val_one_epoch(self, **kwargs):
        """\n        Function `_val_one_epoch`.\n    \n        Returns: Description.\n"""
        pass

    def _run_epoch(self, **kwargs):
        """\n        Function `_run_epoch`.\n    \n        Returns: Description.\n"""
        self._train_one_epoch()
        self._val_one_epoch()

    @abstractmethod
    def run(self, **kwargs):
        """\n        Function `run`.\n    \n        Returns: Description.\n"""
        pass
