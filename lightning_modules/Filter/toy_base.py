# System imports
import sys, os

# 3rd party imports
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
import numpy as np
import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"

# Local imports
from .utils import graph_intersection, load_dataset


class FilterBase(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different filter training regimes
        """
        self.save_hyperparameters()
        # Assign hyperparameters
        self.hparams = hparams

    def setup(self, stage):
        # Handle any subset of [train, val, test] data split, assuming that ordering
        if stage == "fit":
            input_dirs = [None, None, None]
            input_dirs[: len(self.hparams["datatype_names"])] = [
                os.path.join(self.hparams["input_dir"], datatype)
                for datatype in self.hparams["datatype_names"]
            ]
            self.trainset, self.valset, self.testset = [
                load_dataset(input_dir, self.hparams["datatype_split"][i])
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
        #         scheduler = [
        #             {
        #                 'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer[0], factor=self.hparams["factor"], patience=self.hparams["patience"]),
        #                 'monitor': 'val_loss',
        #                 'interval': 'epoch',
        #                 'frequency': 1
        #             }
        #         ]
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

        emb = (
            None if (self.hparams["emb_channels"] == 0) else batch.embedding
        )  # Does this work??

        if self.hparams["ratio"] != 0:
            num_true, num_false = batch.y.bool().sum(), (~batch.y.bool()).sum()
            fake_indices = torch.where(~batch.y.bool())[0][
                torch.randint(num_false, (num_true.item() * self.hparams["ratio"],))
            ]
            true_indices = torch.where(batch.y.bool())[0]
            combined_indices = torch.cat([true_indices, fake_indices])
            # Shuffle indices:
            combined_indices = combined_indices[torch.randperm(len(combined_indices))]
            positive_weight = (
                torch.tensor(self.hparams["weight"])
                if ("weight" in self.hparams)
                else torch.tensor(self.hparams["ratio"])
            )

        else:
            combined_indices = torch.range(batch.edge_index.shape[1])
            positive_weight = (
                torch.tensor(self.hparams["weight"])
                if ("weight" in self.hparams)
                else torch.tensor((~batch.y.bool()).sum() / batch.y.sum())
            )

        output = (
            self(
                torch.cat([batch.cell_data, batch.x], axis=-1),
                batch.edge_index[:, combined_indices],
                emb,
            ).squeeze()
            if ("ci" in self.hparams["regime"])
            else self(batch.x, batch.edge_index[:, combined_indices], emb).squeeze()
        )

        if "weighting" in self.hparams["regime"]:
            manual_weights = batch.weights[combined_indices]
            manual_weights[batch.y[combined_indices] == 0] = 1
        else:
            manual_weights = None

        if "pid" in self.hparams["regime"]:
            y_pid = (
                batch.pid[batch.edge_index[0, combined_indices]]
                == batch.pid[batch.edge_index[1, combined_indices]]
            )
            loss = F.binary_cross_entropy_with_logits(
                output, y_pid.float(), weight=manual_weights, pos_weight=positive_weight
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                output,
                batch.y[combined_indices].float(),
                weight=manual_weights,
                pos_weight=weight,
            )

        self.log("train_loss", loss)

        return result

    def shared_evaluation(self, batch, batch_idx, log=False):

        """
        This method is shared between validation steps and test steps
        """

        output = self(batch.x, batch.edge_index).squeeze()
        scores = torch.sigmoid(output)

        y_pid = batch.pid[batch.edge_index[0]] == batch.pid[batch.edge_index[1]]
        val_loss = F.binary_cross_entropy_with_logits(output, y_pid.float())

        cut_list = scores > self.hparams["val_filter_cut"]

        # Edge filter performance
        edge_positive = max(cut_list.sum().float(), 1)
        edge_true = y_pid.sum()
        edge_true_positive = (y_pid & cut_list).sum().float()

        current_lr = self.optimizers().param_groups[0]["lr"]

        if log:
            self.log_dict(
                {
                    "edge_eff": edge_true_positive / edge_true,
                    "edge_pur": edge_true_positive / edge_positive,
                    "val_loss": val_loss,
                    "current_lr": current_lr,
                }
            )

        return {
            "loss": val_loss,
            "true_positive": (y_pid & cut_list).float().cpu().numpy(),
            "true": y_pid.float().cpu().numpy(),
            "positive": cut_list.float().cpu().numpy(),
        }

    def validation_step(self, batch, batch_idx):

        outputs = self.shared_evaluation(batch, batch_idx, log=True)

        return outputs["loss"]

    def test_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        outputs = self.shared_evaluation(batch, batch_idx, log=False)

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
        """
        Use this to manually enforce warm-up. In the future, this may become built-into PyLightning
        """
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


class FilterBaseBalanced(FilterBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different filter training regimes
        """

    def training_step(self, batch, batch_idx):

        input_edges = self.random_sample(batch)

        output = self(batch.x, input_edges).squeeze()

        if "pid" in self.hparams["regime"]:
            y_pid = batch.pid[input_edges[0]] == batch.pid[input_edges[1]]
            loss = F.binary_cross_entropy_with_logits(
                output, y_pid.float(), pos_weight=torch.tensor(self.hparams["weight"])
            )

        self.log("train_loss", loss)

        return loss

    def random_sample(self, batch):

        bidir_edges = torch.cat([batch.edge_index, batch.edge_index.flip(0)], axis=-1)

        if "n_edges" in self.hparams:
            bidir_edges = bidir_edges[
                :, torch.randperm(bidir_edges.shape[1])[: self.hparams["n_edges"]]
            ]

        if "hnm" in self.hparams["regime"]:
            bidir_edges = self.find_hard_negatives(bidir_edges, batch)

        y = batch.pid[bidir_edges[0]] == batch.pid[bidir_edges[1]]

        num_true, num_false = y.bool().sum(), (~y.bool()).sum()
        fake_indices = torch.where(~y.bool())[0][torch.randperm(num_false)[:num_true]]
        true_indices = torch.where(y.bool())[0]
        combined_indices = torch.cat([true_indices, fake_indices])

        #         print("Num true: {}, num fake: {}".format(true_indices.shape[0], fake_indices.shape[0]))

        # Shuffle indices:
        subgraph = bidir_edges[
            :, combined_indices[torch.randperm(len(combined_indices))]
        ]

        return subgraph

    def find_hard_negatives(self, bidir_edges, batch):

        with torch.no_grad():

            output = self(batch.x, bidir_edges).squeeze()

            cut = F.sigmoid(output) > self.hparams["train_filter_cut"]
            y = batch.pid[bidir_edges[0]] == batch.pid[bidir_edges[1]]

            true_indices = torch.where(y.bool())[0]
            hard_negatives = cut & ~y.bool()
            easy_negatives = ~cut & ~y.bool()
            hard_indices = torch.where(hard_negatives)[0]
            easy_indices = torch.where(easy_negatives)[0][
                torch.randperm(easy_negatives.sum())
            ][: hard_negatives.sum()]

            #             print("Num true: {}, num hard: {}, num easy: {}".format(true_indices.shape[0], hard_negatives.sum(), easy_negatives.sum()))
            combined_indices = torch.cat([true_indices, hard_indices, easy_indices])
            #             combined_indices = torch.cat([true_indices, hard_indices])

            return bidir_edges[:, combined_indices]
