import sys, os
import logging

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from datetime import timedelta
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.nn import Linear
import torch

from ..utils import load_processed_dataset


class GNNBase(LightningModule):
    
    """
    The base class for a combined edge classification AND track parameter regression GNN model. 

    This class abstracts out the training and validation behaviour of edge classification + track parameter regression GNNs. Note: It is specific to edge classification + track parameter regression.
    
    Todo:
        * Rewrite edge classification to include track parameter regression loss function
        * Handle truth for unpooled edges
        * Calculate truth pT for a pooled node

    """

    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        
        Args:
            hparams (dict): The hyperparameter dictionary, preferably loaded from a yaml config file. Can also be loaded from a saved lightning checkpoint.
            
        """
        self.save_hyperparameters()

        # Assign hyperparameters
        self.hparams = hparams

    def setup(self, stage):
        if stage == "fit":
            # Handle any subset of [train, val, test] data split, assuming that ordering
            input_dirs = [None, None, None]
            input_dirs[: len(self.hparams["datatype_names"])] = [
                os.path.join(self.hparams["input_dir"], datatype)
                for datatype in self.hparams["datatype_names"]
            ]
            self.trainset, self.valset, self.testset = [
                load_processed_dataset(input_dir, self.hparams["datatype_split"][i])
                for i, input_dir in enumerate(input_dirs)
            ]

    def train_dataloader(self):
        if self.trainset is not None:
            return DataLoader(self.trainset, batch_size=1, num_workers=1)
        else:
            return None

    def val_dataloader(self):
        if self.valset is not None:
            return DataLoader(self.valset, batch_size=1, num_workers=1)
        else:
            return None

    def test_dataloader(self):
        if self.testset is not None:
            return DataLoader(self.testset, batch_size=1, num_workers=1)
        else:
            return None

    def configure_optimizers(self):
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=(self.hparams["lr"]),
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True,
            )
        ]
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer[0],
                    step_size=self.hparams["patience"],
                    gamma=self.hparams["factor"],
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimizer, scheduler

    def training_step(self, batch, batch_idx):

        weight = (
            torch.tensor(self.hparams["weight"])
            if ("weight" in self.hparams)
            else torch.tensor((~batch.y_pid.bool()).sum() / batch.y_pid.sum())
        )

        output = (
            self(
                torch.cat([batch.cell_data, batch.x], axis=-1), batch.edge_index
            ).squeeze()
            if ("ci" in self.hparams["regime"])
            else self(batch.x, batch.edge_index).squeeze()
        )
        
        # Fix this!
        edge_truth = (
            (batch.pid[batch.edge_index[0]] == batch.pid[batch.edge_index[1]]).float()
            if "pid" in self.hparams["regime"]
            else batch.y
        )

        loss = F.binary_cross_entropy_with_logits(output, edge_truth, pos_weight=weight)

        self.log("train_loss", loss)

        return loss

    def shared_evaluation(self, batch, batch_idx):

        weight = (
            torch.tensor(self.hparams["weight"])
            if ("weight" in self.hparams)
            else torch.tensor((~batch.y_pid.bool()).sum() / batch.y_pid.sum())
        )

        output = (
            self(
                torch.cat([batch.cell_data, batch.x], axis=-1), batch.edge_index
            ).squeeze()
            if ("ci" in self.hparams["regime"])
            else self(batch.x, batch.edge_index).squeeze()
        )
        
        # Fix this!
        truth = (
            (batch.pid[batch.edge_index[0]] == batch.pid[batch.edge_index[1]]).float()
            if "pid" in self.hparams["regime"]
            else batch.y
        )

        loss = F.binary_cross_entropy_with_logits(output, truth.float())

        # Edge filter performance
        preds = F.sigmoid(output) > self.hparams["edge_cut"]
        edge_positive = preds.sum().float()

        edge_true = truth.sum().float()
        edge_true_positive = (truth.bool() & preds).sum().float()

        logging.info(
            "True: {}, Positive: {}, TP: {}".format(
                edge_true, edge_positive, edge_true_positive
            )
        )

        eff = edge_true_positive / edge_true
        pur = edge_true_positive / max(edge_positive, 1)


        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log_dict(
            {
                "val_loss": loss,
                "edge_eff": eff,
                "edge_pur": pur,
                "current_lr": current_lr,
            }
        )

        return {
            "loss": loss,
            "true_positive": (truth.bool() & preds).float().cpu().numpy(),
            "true": truth.float().cpu().numpy(),
            "positive": preds.float().cpu().numpy(),
            "preds": preds.float().cpu().numpy()
        }

    def validation_step(self, batch, batch_idx):

        outputs = self.shared_evaluation(batch, batch_idx)

        return outputs["loss"]

    def test_step(self, batch, batch_idx):

        outputs = self.shared_evaluation(batch, batch_idx)

        return outputs

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure=None,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # warm up lr
        if (self.hparams["warmup"] is not None) and (
            self.trainer.global_step < self.hparams["warmup"]
        ):
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["lr"]

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
