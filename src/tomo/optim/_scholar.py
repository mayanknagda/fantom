import os
import wandb
import torch
import torch.nn as nn
from typing import Any, Callable, List, Optional, Tuple, Union
from ._train import Trainer


class TrainerSCHOLAR(Trainer):
    def __init__(
        self,
        model_name: str,
        model: nn.Module,
        train_dl: torch.utils.data.DataLoader,
        val_dl: torch.utils.data.DataLoader,
        optimizer: Callable,
        device: Union[str, torch.device] = "cpu",
        out_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(model, train_dl, val_dl, optimizer, device, out_dir)
        self.model_name = model_name
        self.beta = kwargs["beta"]
        self.current_epoch = 0
        self.summary = {
            "train/loss": [],
            "val/loss": [],
            "train/kl": [],
            "val/kl": [],
            "train/nll": [],
            "val/nll": [],
        }

    def _train_one_epoch(self):
        self.model.train()
        train_loss = 0
        epoch_kl_loss = 0
        epoch_nll_loss = 0
        for batch in self.train_dl:
            # Get the batch
            bow = batch["bow"].to(self.device)
            kwargs = {}
            kwargs["labels"] = batch["labels"].to(self.device)
            if "authors" in self.model_name:
                kwargs["authors"] = batch["authors"].to(self.device)
            x, dist_params, z, x_hat, kl_d, nll_loss = self.model(bow, **kwargs)
            loss = nll_loss + self.beta * kl_d
            # Backpropagate the loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            epoch_kl_loss += kl_d.item()
            epoch_nll_loss += nll_loss.item()

        self.summary["train/loss"].append(train_loss / len(self.train_dl))
        self.summary["train/kl"].append(epoch_kl_loss / len(self.train_dl))
        self.summary["train/nll"].append(epoch_nll_loss / len(self.train_dl))

        # wandb log
        wandb.log(
            {"train/loss": self.summary["train/loss"][-1]}, step=self.current_epoch
        )
        wandb.log({"train/kl": self.summary["train/kl"][-1]}, step=self.current_epoch)
        wandb.log({"train/nll": self.summary["train/nll"][-1]}, step=self.current_epoch)

    def _val_one_epoch(self):
        self.model.eval()
        val_loss = 0.0
        epoch_kl_loss = 0
        epoch_nll_loss = 0
        with torch.no_grad():
            for batch in self.val_dl:
                # Get the batch
                bow = batch["bow"].to(self.device)
                kwargs = {}
                kwargs["labels"] = batch["labels"].to(self.device)
                if "authors" in self.model_name:
                    kwargs["authors"] = batch["authors"].to(self.device)
                x, dist_params, z, x_hat, kl_d, nll_loss = self.model(bow, **kwargs)
                loss = nll_loss + self.beta * kl_d
                val_loss += loss.item()
                epoch_kl_loss += kl_d.item()
                epoch_nll_loss += nll_loss.item()

        self.summary["val/loss"].append(val_loss / len(self.val_dl))
        self.summary["val/kl"].append(epoch_kl_loss / len(self.val_dl))
        self.summary["val/nll"].append(epoch_nll_loss / len(self.val_dl))

        # wandb log
        wandb.log({"val/loss": self.summary["val/loss"][-1]}, step=self.current_epoch)
        wandb.log({"val/kl": self.summary["val/kl"][-1]}, step=self.current_epoch)
        wandb.log({"val/nll": self.summary["val/nll"][-1]}, step=self.current_epoch)

    def run(self, **kwargs):
        for _ in range(kwargs["epochs"]):
            self.current_epoch += 1
            self._run_epoch()
            curr_train_loss = self.summary["train/loss"][-1]
            curr_val_loss = self.summary["val/loss"][-1]
            if curr_val_loss < self.best_val_loss:
                self.best_val_loss = curr_val_loss
                self._save_model()
            print(
                f"Epoch: {self.current_epoch:02d} | Train Loss: {curr_train_loss:.3f} | Val Loss: {curr_val_loss:.3f}"
            )
