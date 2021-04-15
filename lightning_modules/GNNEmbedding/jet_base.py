import sys, os
import logging

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from datetime import timedelta
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.nn import Linear
import torch
from sklearn.decomposition import PCA
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import scipy.sparse as sp

from IPython import display

from ..utils import (
    load_processed_dataset,
    build_edges,
    graph_intersection,
    MultiNoiseLoss,
)


class ToyGNNEmbeddingBase(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """
        # Assign hyperparameters
        self.hparams = hparams
        self.hparams["posted_alert"] = False

    def setup(self, stage):
        if stage == "fit":
            # Handle any subset of [train, val, test] data split, assuming that ordering
            self.min_edges = (
                self.hparams["min_edges"] if "min_edges" in self.hparams else None
            )
            input_dirs = [None, None, None]
            input_dirs[: len(self.hparams["datatype_names"])] = [
                os.path.join(self.hparams["input_dir"], datatype)
                for datatype in self.hparams["datatype_names"]
            ]
            self.trainset, self.valset, self.testset = [
                load_processed_dataset(
                    input_dir,
                    self.hparams["datatype_split"][i],
                    min_edges=self.min_edges,
                )
                for i, input_dir in enumerate(input_dirs)
            ]
            self.multi_loss = MultiNoiseLoss(n_losses=2, device=self.device).to(
                self.device
            )

    #             self.fig = plt.figure()
    #             self.ax = self.fig.add_subplot(111)
    #             empty_list = np.zeros(torch.unique(self.valset[0].edge_index).shape[0])
    #             print(empty_list.shape)
    #             self.scatter1, = self.ax.plot(empty_list, empty_list, marker="o", ls="")
    #             plt.pause(0.01)

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
                [
                    {"params": self.parameters()},
                    {"params": self.multi_loss.noise_params},
                ],
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

    def get_all_pairs(self, batch):

        all_combos = combinations(torch.unique(batch.edge_index).to("cpu"), 2)
        combo_numpy = np.array(list(all_combos))
        combo_tensor = torch.from_numpy(combo_numpy).T.to(self.device)

        return combo_tensor

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
        else:
            y_triplet = y_triplet * torch.from_numpy(duplicated_hid).to(self.device)

        return y_triplet, triplet_edges

    def get_input_features(self, batch):

        # Forward pass of model, handling whether Cell Information (ci) is included
        if "ci" in self.hparams["regime"]:
            hit_features = torch.cat([batch.cell_data, batch.x], axis=-1)
        else:
            hit_features = batch.x

        return hit_features

    def get_classification_loss(self, batch, doublet_score, edge_inputs):

        y = batch.pid[edge_inputs[0]] == batch.pid[edge_inputs[1]]
        filter_loss = F.binary_cross_entropy_with_logits(
            doublet_score.squeeze(),
            y.float(),
            pos_weight=torch.tensor(self.hparams["weight"]),
        )

        edge_prediction = (
            torch.sigmoid(doublet_score) > self.hparams["edge_cut"]
        ).squeeze()

        return filter_loss, y, edge_prediction

    #     def get_embedding_loss(self, batch, latent_space, training=True, knn_num=100, knn_radius=1):
    #         pass

    def training_step(self, batch, batch_idx):

        node_features = self.get_input_features(batch)

        edge_inputs = self.get_subgraph(batch)

        latent_features, doublet_score = self(node_features, edge_inputs)

        if self.hparams["edge_loss_only"]:
            classification_loss, _, _ = self.get_classification_loss(
                batch, doublet_score, edge_inputs
            )
            loss = classification_loss
        elif self.hparams["emb_loss_only"]:
            emb_loss = self.get_embedding_training_loss(batch, latent_features)
            loss = emb_loss
        else:
            classification_loss, _, _ = self.get_classification_loss(
                batch, doublet_score, edge_inputs
            )
            emb_loss = self.get_embedding_training_loss(batch, latent_features)
            self.log_dict(
                {"emb_loss": emb_loss, "classification_loss": classification_loss}
            )

            loss = self.multi_loss([emb_loss, classification_loss])

        #         if (batch_idx % 10 == 0):
        #             self.plot_example()

        self.log("train_loss", loss)

        return loss

    def shared_evaluation(self, batch, batch_idx, knn_radius, knn_num, log=False):

        node_features = self.get_input_features(batch)
        edge_inputs = self.get_subgraph(batch)

        latent_features, doublet_score = self(node_features, edge_inputs)

        emb_loss, truth_graph, y = self.get_embedding_val_loss(
            batch, latent_features, knn_radius, knn_num
        )

        (
            classification_loss,
            filter_truth_graph,
            edge_prediction,
        ) = self.get_classification_loss(batch, doublet_score, edge_inputs)

        if self.hparams["edge_loss_only"]:
            loss = classification_loss
        elif self.hparams["emb_loss_only"]:
            loss = emb_loss
        else:
            loss = emb_loss + classification_loss

        # Edge filter performance
        cluster_true = truth_graph.shape[1]
        cluster_true_positive = y.sum()
        cluster_positive = len(y)

        eff = cluster_true_positive / max(cluster_true, 1)
        pur = cluster_true_positive / max(cluster_positive, 1)

        # Metric learning performance
        edge_true = filter_truth_graph.sum()
        edge_true_positive = (filter_truth_graph & edge_prediction).sum()
        edge_positive = edge_prediction.sum()

        edge_eff = edge_true_positive / max(edge_true, 1)
        edge_pur = edge_true_positive / max(edge_positive, 1)

        current_lr = self.optimizers().param_groups[0]["lr"]
        if log:
            self.log_dict(
                {
                    "val_loss": loss,
                    "eff": eff,
                    "pur": pur,
                    "edge_eff": edge_eff,
                    "edge_pur": edge_pur,
                    "current_lr": current_lr,
                }
            )

        #         logging.info("Efficiency: {}".format(eff))
        #         logging.info("Purity: {}".format(pur))

        return {
            "loss": loss,
            "truth": y.cpu().numpy(),
            "truth_graph": truth_graph.cpu().numpy(),
            "edge_eff": edge_eff,
            "edge_pur": edge_pur,
        }

    def validation_step(self, batch, batch_idx):

        outputs = self.shared_evaluation(
            batch, batch_idx, self.hparams["r_val"], 200, log=True
        )

        return outputs["loss"]

    def test_step(self, batch, batch_idx):

        outputs = self.shared_evaluation(
            batch, batch_idx, self.hparams["r_test"], 500, log=False
        )

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


class JetGNNNodeEmbeddingBase(ToyGNNEmbeddingBase):
    def __init__(self, hparams):
        super().__init__(hparams)

    def get_subgraph(self, batch):

        if (
            "subgraph" in self.hparams["regime"]
        ):  # and batch.sub_edge_index.sum() > 1000:

            if batch.sub_edge_index[0].dtype == "bool":
                subgraph = batch.edge_index[:, batch.sub_edge_index[0]]
            else:
                subgraph = batch.sub_edge_index

            if "random_edges" in self.hparams["regime"]:
                random_edges = self.get_random_edges(subgraph, subgraph)
                subgraph = torch.cat([subgraph, random_edges], axis=-1)

        else:
            subgraph = torch.randperm(batch.edge_index.shape[1])[
                : self.hparams["n_edges"]
            ]

        return subgraph

    def get_random_edges(self, truth_graph, subgraph):

        n_random = int(self.hparams["randomisation"] * truth_graph.shape[1])
        subgraph_hits = torch.reshape(subgraph, (-1,))
        subgraph_hits = subgraph_hits[torch.randperm(len(subgraph_hits))]
        random_edges = torch.reshape(subgraph_hits, (2, subgraph_hits.shape[0] // 2))[
            :, :n_random
        ]

        return random_edges

    def get_connected_pairs(self, subgraph, walk_length=None):

        if walk_length is None:
            walk_length = self.hparams["n_graph_iters"]

        graph_np = subgraph.cpu().numpy()
        graph_coo = sp.coo_matrix(
            (np.ones(graph_np.shape[1]), (graph_np[0], graph_np[1])),
            [graph_np.max() + 1, graph_np.max() + 1],
        )
        graph_csr = graph_coo.tocsr() + graph_coo.tocsr().T
        walked_graph = graph_csr ** (walk_length - 1)
        walked_graph = graph_csr * walked_graph + walked_graph
        connected_pairs = (
            torch.from_numpy(np.vstack(walked_graph.nonzero())).long().to(self.device)
        )
        connected_pairs = connected_pairs[:, connected_pairs[0] != connected_pairs[1]]

        return connected_pairs

    def get_embedding_training_loss(self, batch, latent_space):

        # Construct training examples or KNN validation
        subgraph = self.get_subgraph(batch)

        truth_graph = batch.pid_true_edges[
            :, np.isin(batch.pid_true_edges.cpu(), subgraph.cpu()).all(0)
        ]

        if "all_pairs" in batch.__dict__.keys() and (
            "global_train" not in self.hparams["regime"]
        ):
            doublet_edges = torch.cat(
                [batch.all_pairs, batch.all_pairs.flip(0)], axis=-1
            )

        else:
            doublet_edges = torch.empty([2, 0], dtype=torch.int64, device=self.device)

            # Append pairs within an n_graph_iter-walk radius
            doublet_edges = torch.cat(
                [doublet_edges, self.get_connected_pairs(subgraph)], axis=-1
            )

            # Append random edges pairs (rp) for stability
            random_edges = self.get_random_edges(truth_graph, subgraph)
            doublet_edges = torch.cat([doublet_edges, random_edges], axis=-1)

            # Append Hard Negative Mining (hnm) with KNN graph
            unique_hits = torch.unique(subgraph)
            knn_edges = build_edges(
                latent_space[unique_hits], self.hparams["r_train"], self.hparams["knn"]
            )
            knn_edges = unique_hits[knn_edges]
            doublet_edges = torch.cat([doublet_edges, knn_edges], axis=-1)

            # Maybe remove this??
            #             doublet_edges = torch.cat([doublet_edges, batch.all_pairs, batch.all_pairs.flip(0)], axis=-1)

            # Append truth
            doublet_edges = torch.cat([doublet_edges, truth_graph], axis=-1)

        # Get distance metric
        reference = latent_space.index_select(0, doublet_edges[1])
        neighbors = latent_space.index_select(0, doublet_edges[0])
        d = torch.sum((reference - neighbors) ** 2, dim=-1)

        y = batch.pid[doublet_edges[0]] == batch.pid[doublet_edges[1]]

        new_weights = y.float() * self.hparams["weight"]
        new_weights[y.float() == 0] = 1

        d = d * new_weights

        hinge = y.float()
        hinge[hinge == 0] = -1

        emb_loss = torch.nn.functional.hinge_embedding_loss(
            d, hinge, margin=self.hparams["margin"], reduction="mean"
        )

        return emb_loss

    def get_local_performance(
        self, subgraph, batch, latent_space, k_radius, walk_length
    ):

        walk_edges = self.get_connected_pairs(subgraph, walk_length=walk_length)
        reference = latent_space.index_select(0, walk_edges[1])
        neighbors = latent_space.index_select(0, walk_edges[0])
        d = torch.sum((reference - neighbors) ** 2, dim=-1)

        in_radius_edges = d < k_radius ** 2
        d = d[in_radius_edges]
        walk_edges = walk_edges[:, in_radius_edges]

        walk_y = batch.pid[walk_edges[0]] == batch.pid[walk_edges[1]]
        walk_truth = batch.pid_true_edges[
            :, np.isin(batch.pid_true_edges.cpu(), walk_edges.cpu()).all(0)
        ]

        walk_cluster_true = walk_truth.shape[1]
        walk_cluster_true_positive = walk_y.sum()
        walk_cluster_positive = len(walk_y)

        walk_eff = walk_cluster_true_positive / max(walk_cluster_true, 1)
        walk_pur = walk_cluster_true_positive / max(walk_cluster_positive, 1)

        return walk_eff, walk_pur

    def get_embedding_val_loss(self, batch, latent_space, k_radius, knn):

        # Construct training examples or KNN validation
        subgraph = self.get_subgraph(batch)

        # DEBUGGING: What is the performance when only considering the local neighbourhood?

        step_4_eff, step_4_pur = self.get_local_performance(
            subgraph, batch, latent_space, k_radius, walk_length=4
        )
        self.log_dict({"step_4_eff": step_4_eff, "step_4_pur": step_4_pur})

        step_8_eff, step_8_pur = self.get_local_performance(
            subgraph, batch, latent_space, k_radius, walk_length=8
        )
        self.log_dict({"step_8_eff": step_8_eff, "step_8_pur": step_8_pur})

        # REGULAR EVALUATION:

        truth_graph = batch.pid_true_edges[
            :, np.isin(batch.pid_true_edges.cpu(), subgraph.cpu()).all(0)
        ]

        if "all_pairs" in batch.__dict__.keys() and (
            "global_inference" not in self.hparams["regime"]
        ):
            doublet_edges = torch.cat(
                [batch.all_pairs, batch.all_pairs.flip(0)], axis=-1
            )

        else:
            # Get neighbourhoods within subgraph
            unique_hits = torch.unique(subgraph)
            knn_edges = build_edges(latent_space[unique_hits], k_radius, knn)
            doublet_edges = unique_hits[knn_edges]

        reference = latent_space.index_select(0, doublet_edges[1])
        neighbors = latent_space.index_select(0, doublet_edges[0])
        d = torch.sum((reference - neighbors) ** 2, dim=-1)

        in_radius_edges = d < k_radius ** 2
        d = d[in_radius_edges]
        doublet_edges = doublet_edges[:, in_radius_edges]

        y = batch.pid[doublet_edges[0]] == batch.pid[doublet_edges[1]]

        hinge = y.float()
        hinge[hinge == 0] = -1

        emb_loss = torch.nn.functional.hinge_embedding_loss(
            d, hinge, margin=self.hparams["margin"], reduction="mean"
        )

        return emb_loss, truth_graph, y


class ToyGNNEdgeEmbeddingBase(ToyGNNEmbeddingBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

    def get_embedding_loss(
        self, batch, latent_space, training=True, knn_num=100, knn_radius=1
    ):

        if "pid" in self.hparams["regime"]:
            truth_graph = batch.pid_true_triplets
        else:
            truth_graph = batch.layerless_true_triplets

        if training:
            triplet_edges = torch.empty([2, 0], dtype=torch.int64, device=self.device)

            # Append random edges pairs (rp) for stability
            if "rp" in self.hparams["regime"]:
                n_random = int(self.hparams["randomisation"] * truth_graph.shape[1])
                triplet_edges = torch.cat(
                    [
                        triplet_edges,
                        torch.randint(
                            0,
                            batch.edge_index.shape[1],
                            (2, n_random),
                            device=self.device,
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
                            latent_space, self.hparams["r_train"], self.hparams["knn"]
                        ),
                    ],
                    axis=-1,
                )

        else:
            triplet_edges = build_edges(latent_space, knn_radius, knn_num)

        # Get triplet true positives
        y_doublet = batch.pid[batch.edge_index[0]] == batch.pid[batch.edge_index[1]]
        pid_doublet = y_doublet * batch.pid[batch.edge_index[0]]
        y_triplet = y_doublet[triplet_edges[0]] * (
            pid_doublet[triplet_edges[0]] == pid_doublet[triplet_edges[1]]
        )

        # Append all truth
        if "pid" not in self.hparams["regime"]:
            y_triplet, triplet_edges = self.adjacent_triplet_truth(
                batch, y_triplet, triplet_edges, mask=(~training)
            )

        if "noisy" in self.hparams["regime"] and training:
            # Only include in loss triplets that include AT LEAST one real doublet. This may improve Edge+Filter behaviour
            either_real = y_doublet[triplet_edges].any(axis=0)
            triplet_edges = triplet_edges[:, either_real]
            y_triplet = y_triplet[either_real]

        if training:
            triplet_edges = torch.cat([triplet_edges, truth_graph], axis=-1)
            y_triplet = torch.cat(
                [y_triplet, torch.ones(truth_graph.shape[1], device=self.device)]
            )

        new_weights = y_triplet.float() * self.hparams["weight"]
        new_weights[y_triplet.float() == 0] = 1

        reference = latent_space.index_select(0, triplet_edges[1])
        neighbors = latent_space.index_select(0, triplet_edges[0])
        d = torch.sum((reference - neighbors) ** 2, dim=-1)
        d = d * new_weights

        hinge = y_triplet.float()
        hinge[hinge == 0] = -1

        emb_loss = torch.nn.functional.hinge_embedding_loss(
            d, hinge, margin=self.hparams["margin"], reduction="mean"
        )

        return emb_loss, truth_graph, y_triplet
