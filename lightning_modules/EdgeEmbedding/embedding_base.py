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
    build_edges,
    load_processed_dataset,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingBase(LightningModule):
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

        """
        TODO
        [X] Get HitE features
        [X] Run through model
        [X] Get random pairs
        [X] Get KNN
        [X] Build true edge list
        [ ] Add weighting
        [-] Add flipped truth?
        [X] Define PID truth
        [X] Define loss
        """

        # Forward pass of model, handling whether Cell Information (ci) is included
        if "ci" in self.hparams["regime"]:
            hit_features = torch.cat([batch.cell_data, batch.x], axis=-1)
            doublet_features = torch.cat(
                [hit_features[batch.edge_index[0]], hit_features[batch.edge_index[1]]],
                axis=-1,
            )
        else:
            doublet_features = torch.cat(
                [batch.x[batch.edge_index[0]], batch.x[batch.edge_index[1]]], axis=-1
            )

        doublet_latent = self(doublet_features)

        triplet_edges = torch.empty([2, 0], dtype=torch.int64, device=self.device)

        if "pid" in self.hparams["regime"]:
            truth_graph = batch.pid_true_triplets
        else:
            truth_graph = batch.layerless_true_triplets

        # Append random edges pairs (rp) for stability
        if "rp" in self.hparams["regime"]:
            n_random = int(self.hparams["randomisation"] * truth_graph.shape[1])
            triplet_edges = torch.cat(
                [
                    triplet_edges,
                    torch.randint(
                        0, batch.edge_index.shape[1], (2, n_random), device=self.device
                    ),
                ],
                axis=-1,
            )

        # Append Hard Negative Mining (hnm) with KNN graph
        if "hnm" in self.hparams["regime"]:
            triplet_edges = torch.cat(
                [
                    triplet_edges,
                    build_edges(
                        doublet_latent, self.hparams["r_train"], self.hparams["knn"]
                    ),
                ],
                axis=-1,
            )

        # Get triplet true positives
        y_doublet = batch.pid[batch.edge_index[0]] == batch.pid[batch.edge_index[1]]
        pid_doublet = y_doublet * batch.pid[batch.edge_index[0]]
        y_triplet = y_doublet[triplet_edges[0]] * (
            pid_doublet[triplet_edges[0]] == pid_doublet[triplet_edges[1]]
        )

        # Append all truth
        if "pid" not in self.hparams["regime"]:
            y_triplet = self.adjacent_triplet_truth(batch, y_triplet, triplet_edges)

        triplet_edges = torch.cat([triplet_edges, truth_graph], axis=-1)
        y_triplet = torch.cat(
            [y_triplet, torch.ones(truth_graph.shape[1], device=self.device)]
        )

        new_weights = y_triplet.float() * self.hparams["weight"]
        new_weights[y_triplet.float() == 0] = 1

        reference = doublet_latent.index_select(0, triplet_edges[1])
        neighbors = doublet_latent.index_select(0, triplet_edges[0])
        d = torch.sum((reference - neighbors) ** 2, dim=-1)
        d = d * new_weights

        hinge = y_triplet.float()
        hinge[hinge == 0] = -1

        loss = torch.nn.functional.hinge_embedding_loss(
            d, hinge, margin=self.hparams["margin"], reduction="mean"
        )

        self.log("train_loss", loss)

        return loss

    def shared_evaluation(self, batch, batch_idx, knn_radius, knn_num, log=False):

        # Forward pass of model, handling whether Cell Information (ci) is included
        if "ci" in self.hparams["regime"]:
            hit_features = torch.cat([batch.cell_data, batch.x], axis=-1)
            doublet_features = torch.cat(
                [hit_features[batch.edge_index[0]], hit_features[batch.edge_index[1]]],
                axis=-1,
            )
        else:
            doublet_features = torch.cat(
                [batch.x[batch.edge_index[0]], batch.x[batch.edge_index[1]]], axis=-1
            )

        doublet_latent = self(doublet_features)

        # Build whole KNN graph
        triplet_edges = build_edges(doublet_latent, knn_radius, knn_num)

        y_doublet = batch.pid[batch.edge_index[0]] == batch.pid[batch.edge_index[1]]
        pid_doublet = y_doublet * batch.pid[batch.edge_index[0]]
        y_triplet = y_doublet[triplet_edges[0]] * (
            pid_doublet[triplet_edges[0]] == pid_doublet[triplet_edges[1]]
        )

        if "pid" not in self.hparams["regime"]:
            y_triplet, triplet_edges = self.adjacent_triplet_truth(
                batch, y_triplet, triplet_edges, mask=True
            )

        reference = doublet_latent.index_select(0, triplet_edges[1])
        neighbors = doublet_latent.index_select(0, triplet_edges[0])
        d = torch.sum((reference - neighbors) ** 2, dim=-1)

        hinge = y_triplet.float()
        hinge[hinge == 0] = -1

        loss = torch.nn.functional.hinge_embedding_loss(
            d, hinge, margin=self.hparams["margin"], reduction="mean"
        )

        if "pid" in self.hparams["regime"]:
            truth_graph = batch.pid_true_triplets
        else:
            truth_graph = batch.layerless_true_triplets

        cluster_true = truth_graph.shape[1]
        cluster_true_positive = y_triplet.sum()
        cluster_positive = triplet_edges.shape[1]

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
            "preds": triplet_edges.cpu().numpy(),
            "truth": y_triplet.cpu().numpy(),
            "truth_graph": truth_graph.cpu().numpy(),
            "doublet_truth": y_doublet.cpu().numpy(),
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
            batch, batch_idx, self.hparams["r_test"], 300, log=False
        )

        return outputs

    def get_true_edge_number(self, batch):
        """
        For PID Truth, return ideal number of edges
        """

        true_doublets = batch.edge_index[
            0, batch.pid[batch.edge_index[0]] == batch.pid[batch.edge_index[1]]
        ]
        true_doublet_pids = batch.pid[true_doublets].cpu().numpy()

        _, counts = np.unique(true_doublet_pids, return_counts=True)
        count_list = np.array([count * (count - 1) for count in counts])

        return count_list.sum()

    def adjacent_triplet_truth(self, batch, y_triplet, triplet_edges, mask=False):

        edge_i = batch.edge_index.cpu().numpy()[:, triplet_edges[0].cpu()]
        edge_o = batch.edge_index.cpu().numpy()[:, triplet_edges[1].cpu()]
        duplicated_hid = (edge_i == edge_o).any(axis=0) + (edge_i[::-1] == edge_o).any(
            axis=0
        )

        # Mask specifies whether to hide triplets that aren't adjacent, or whether to simply intersect the truth vector with adjacent edges
        if mask:
            y_triplet = y_triplet[torch.from_numpy(duplicated_hid).to(self.device)]
            triplet_edges = triplet_edges[
                :, torch.from_numpy(duplicated_hid).to(self.device)
            ]
            return y_triplet, triplet_edges
        else:
            y_triplet = y_triplet * torch.from_numpy(duplicated_hid).to(self.device)
            return y_triplet

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
