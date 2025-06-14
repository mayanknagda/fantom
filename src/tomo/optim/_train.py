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
        torch.save(self.model.state_dict(), os.path.join(self.exp_path, "model.pt"))

    @abstractmethod
    def _train_one_epoch(self, **kwargs):
        pass

    @abstractmethod
    def _val_one_epoch(self, **kwargs):
        pass

    def _run_epoch(self, **kwargs):
        self._train_one_epoch()
        self._val_one_epoch()

    @abstractmethod
    def run(self, **kwargs):
        pass
