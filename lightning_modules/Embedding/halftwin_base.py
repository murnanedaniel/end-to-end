# System imports
import sys
import os
import logging

# 3rd party imports
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch
from torch.nn import Linear
from torch_geometric.data import DataLoader
from torch_cluster import radius_graph
import numpy as np

# Local Imports
from ..utils import (
    graph_intersection,
    split_datasets,
    load_processed_dataset,
    build_edges,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


class HalfTwinEmbeddingBase(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        self.save_hyperparameters()
        # Assign hyperparameters
        self.hparams = hparams

    def setup(self, stage):
        if stage == "fit":
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
        if len(self.trainset) > 0:
            return DataLoader(self.trainset, batch_size=1, num_workers=1)
        else:
            return None

    def val_dataloader(self):
        if len(self.valset) > 0:
            return DataLoader(self.valset, batch_size=1, num_workers=1)
        else:
            return None

    def test_dataloader(self):
        if len(self.testset):
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
        #         scheduler = [torch.optim.lr_scheduler.StepLR(optimizer[0], step_size=1, gamma=0.3)]
        return optimizer, scheduler

    def training_step(self, batch, batch_idx):

        """
        Example:
        TODO - Explain how the embedding training step works by example!

        Args:
            batch (``list``, required): A list of ``torch.tensor`` objects
            batch (``int``, required): The index of the batch

        Returns:
            ``torch.tensor`` The loss function as a tensor
        """

        # Forward pass of model, handling whether Cell Information (ci) is included
        if "ci" in self.hparams["regime"]:
            source_spatial, target_spatial = self(
                torch.cat([batch.cell_data, batch.x], axis=-1)
            )
        else:
            source_spatial, target_spatial = self(batch.x)

        # Instantiate bidirectional truth (since KNN prediction will be bidirectional)
        if "pid" in self.hparams["regime"]:
            truth_graph = batch.pid_true_edges
        else:
            truth_graph = torch.cat(
                [batch.layerless_true_edges, batch.layerless_true_edges.flip(0)],
                axis=-1,
            )

        # Instantiate empty prediction edge list
        e_spatial = torch.empty([2, 0], dtype=torch.int64, device=self.device)

        # Append random edges pairs (rp) for stability
        if "rp" in self.hparams["regime"]:
            n_random = int(self.hparams["randomisation"] * truth_graph.shape[1])
            e_spatial = torch.cat(
                [
                    e_spatial,
                    torch.randint(
                        truth_graph.min(),
                        truth_graph.max(),
                        (2, n_random),
                        device=self.device,
                    ),
                ],
                axis=-1,
            )

        # Append Hard Negative Mining (hnm) with KNN graph
        if "hnm" in self.hparams["regime"]:
            e_spatial = torch.cat(
                [
                    e_spatial,
                    build_edges(
                        source_spatial,
                        self.hparams["r_train"],
                        self.hparams["knn"],
                        target_spatial=target_spatial,
                    ),
                ],
                axis=-1,
            )

        # Calculate truth from intersection between Prediction graph and Truth graph
        if "weighting" in self.hparams["regime"]:
            weights_bidir = torch.cat([batch.weights, batch.weights])
            e_spatial, y_cluster, new_weights = graph_intersection(
                e_spatial, e_bidir, using_weights=True, weights_bidir=weights_bidir
            )
            new_weights = (
                new_weights.to(self.device) * self.hparams["weight"]
            )  # Weight positive examples
        else:
            if "pid" in self.hparams["regime"]:
                y_cluster = batch.pid[e_spatial[0]] == batch.pid[e_spatial[1]]

            else:
                e_spatial, y_cluster = graph_intersection(e_spatial, truth_graph)
            new_weights = y_cluster.to(self.device) * self.hparams["weight"]
            # Append all positive examples and their truth and weighting
            e_spatial = torch.cat([e_spatial.to(self.device), truth_graph], axis=-1)
            y_cluster = torch.cat(
                [
                    y_cluster.to(self.device),
                    torch.ones(truth_graph.shape[1], device=self.device),
                ]
            )

        if "weighting" in self.hparams["regime"]:
            new_weights = torch.cat(
                [new_weights, weights_bidir * self.hparams["weight"]]
            )
        else:
            new_weights = torch.cat(
                [
                    new_weights,
                    torch.ones(truth_graph.shape[1], device=self.device)
                    * self.hparams["weight"],
                ]
            )

        reference = source_spatial.index_select(0, e_spatial[0])
        neighbors = target_spatial.index_select(0, e_spatial[1])
        d = torch.sum((reference - neighbors) ** 2, dim=-1)

        new_weights[
            y_cluster == 0
        ] = 1  # Give negative examples a weight of 1 (note that there may still be TRUE examples that are weightless)
        d = d * new_weights

        hinge = y_cluster.float()
        hinge[hinge == 0] = -1

        loss = torch.nn.functional.hinge_embedding_loss(
            d, hinge, margin=self.hparams["margin"], reduction="mean"
        )

        self.log("train_loss", loss)

        return loss

    def shared_evaluation(self, batch, batch_idx, knn_radius, knn_num, log=False):

        if "ci" in self.hparams["regime"]:
            source_spatial, target_spatial = self(
                torch.cat([batch.cell_data, batch.x], axis=-1)
            )
        else:
            source_spatial, target_spatial = self(batch.x)

        if "pid" in self.hparams["regime"]:
            truth_graph = batch.pid_true_edges
        else:
            truth_graph = torch.cat(
                [batch.layerless_true_edges, batch.layerless_true_edges.flip(0)],
                axis=-1,
            )

        # Build whole KNN graph
        e_spatial = build_edges(
            source_spatial, knn_radius, knn_num, target_spatial=target_spatial
        )

        if "weighting" in self.hparams["regime"]:
            weights_bidir = torch.cat([batch.weights, batch.weights])
            e_spatial, y_cluster, new_weights = graph_intersection(
                e_spatial, truth_graph, using_weights=True, weights_bidir=weights_bidir
            )
            new_weights = (
                new_weights.to(self.device) * self.hparams["weight"]
            )  # Weight positive examples
        else:
            if "pid" in self.hparams["regime"]:
                y_cluster = batch.pid[e_spatial[0]] == batch.pid[e_spatial[1]]
            else:
                e_spatial, y_cluster = graph_intersection(e_spatial, truth_graph)
            new_weights = y_cluster.to(self.device) * self.hparams["weight"]

        e_spatial = e_spatial.to(self.device)
        reference = source_spatial.index_select(0, e_spatial[0])
        neighbors = target_spatial.index_select(0, e_spatial[1])
        d = torch.sum((reference - neighbors) ** 2, dim=-1)

        new_weights[y_cluster == 0] = 1
        d = d * new_weights

        hinge = y_cluster.float().to(self.device)
        hinge[hinge == 0] = -1

        loss = torch.nn.functional.hinge_embedding_loss(
            d, hinge, margin=self.hparams["margin"], reduction="mean"
        )

        cluster_true = truth_graph.shape[1]
        cluster_true_positive = y_cluster.sum()
        cluster_positive = len(e_spatial[0])

        eff = torch.tensor(cluster_true_positive / cluster_true)
        pur = torch.tensor(cluster_true_positive / cluster_positive)

        current_lr = self.optimizers().param_groups[0]["lr"]
        if log:
            self.log_dict(
                {"val_loss": loss, "eff": eff, "pur": pur, "current_lr": current_lr}
            )
        logging.info("Efficiency: {}".format(eff))
        logging.info("Purity: {}".format(pur))
        logging.info(batch.event_file)

        return {
            "loss": loss,
            "preds": e_spatial.cpu().numpy(),
            "truth": y_cluster.cpu().numpy(),
            "truth_graph": truth_graph.cpu().numpy(),
        }

    def validation_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """

        outputs = self.shared_evaluation(
            batch, batch_idx, self.hparams["r_val"], 100, log=True
        )

        return outputs["loss"]

    def test_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        outputs = self.shared_evaluation(
            batch, batch_idx, self.hparams["r_test"], 500, log=False
        )

        return outputs

    def get_true_edge_number(self, batch):
        """
        For PID Truth, return ideal number of edges
        """
        _, counts = np.unique(batch.pid.cpu().numpy(), return_counts=True)
        count_list = np.array([count * (count - 1) for count in counts])
        return count_list.sum()

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
