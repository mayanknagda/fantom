"""\nOptimization helpers (training loops, loss functions).\n\nThis module is part of the `tomo` topic modeling library.\n"""

import torch
import torch.nn as nn
from typing import Callable, Union, Optional
from ._vae import TrainerVAE
from ._scholar import TrainerSCHOLAR

register_trainer = {
    "vae": TrainerVAE,
    "scholar": TrainerSCHOLAR,
}


def get_trainer(
    model_name: str,
    model: nn.Module,
    train_dl: torch.utils.data.DataLoader,
    val_dl: torch.utils.data.DataLoader,
    optimizer: Callable,
    device: Union[str, torch.device] = "cpu",
    exp_path: Optional[str] = None,
    **kwargs,
) -> TrainerVAE:
    trainer_type = model_name.split("-")[0]
    if trainer_type in register_trainer:
        return register_trainer[trainer_type](
            model_name, model, train_dl, val_dl, optimizer, device, exp_path, **kwargs
        )
    else:
        raise ValueError(f"Trainer type {trainer_type} is not supported")
