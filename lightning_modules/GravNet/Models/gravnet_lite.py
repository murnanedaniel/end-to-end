# System imports
import sys
import os
import logging

# 3rd party imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from ..gravnet_base import GravNetBase
from torch.nn import Linear
import torch.nn as nn
import torch
from torch_geometric.data import DataLoader
from torch_scatter import scatter_add, scatter_max
from torch.utils.checkpoint import checkpoint

# Local imports
from ..utils import graph_intersection, split_datasets, build_edges, build_knn, make_mlp


class GravNetLite(GravNetBase):
    def __init__(self, hparams):
        super().__init__(hparams)

        # Construct architecture
        # -------------------------

        # Encode input features to space co-ordinates
        self.space_encoder = make_mlp(
            hparams["hidden_channels"],
            [hparams["space_channels"]] * hparams["encoding_layers"],
            hidden_activation=hparams["hidden_activation"],
            output_activation=None,
            layer_norm=hparams["layernorm"],
        )

        # Encode input features to hidden features
        self.feature_encoder = make_mlp(
            hparams["in_channels"],
            [hparams["hidden_channels"]] * hparams["encoding_layers"],
            hidden_activation=hparams["hidden_activation"],
            output_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
        )

        # Encode hidden features to next-block hidden features
        self.feature_network = make_mlp(
            2 * hparams["hidden_channels"],
            [hparams["hidden_channels"]] * hparams["encoding_layers"],
            output_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
        )

        # Decode hidden features to output features
        self.decoder = make_mlp(
            (self.hparams["num_iterations"] + 1) * hparams["space_channels"],
            [hparams["hidden_channels"]] * hparams["decoding_layers"]
            + [hparams["output_channels"]],
            hidden_activation=hparams["hidden_activation"],
            output_activation=None,
            layer_norm=hparams["layernorm"],
        )

    #         self.decoder = make_mlp(hparams["space_channels"], [hparams["hidden_channels"]]*hparams["decoding_layers"] + [hparams["output_channels"]],
    #                                      hidden_activation=hparams["hidden_activation"], output_activation=None,
    #                                      layer_norm=hparams["layernorm"])

    def forward(self, x):

        # Encode to features

        features = self.feature_encoder(x)

        # Encode to space
        spatial = self.space_encoder(features)
        all_spatial = []
        all_spatial.append(spatial)

        for i in range(self.hparams["num_iterations"]):
            # Aggregate to messages
            edge_index = build_knn(spatial, self.hparams["hidden_knn"])
            start, end = edge_index

            reference = spatial.index_select(0, end)
            neighbors = spatial.index_select(0, start)

            d = torch.sum((reference - neighbors) ** 2, dim=-1)
            d_weight = torch.exp(-self.hparams["exp_coeff"] * d)

            weighted_sum_messages = scatter_add(
                features[start] * d_weight.unsqueeze(1), end, dim=0, dim_size=x.shape[0]
            )
            concated_messages = torch.cat((weighted_sum_messages, features), dim=-1)

            # Feature network
            new_features = self.feature_network(concated_messages)

            spatial = self.space_encoder(new_features)
            all_spatial.append(spatial)

        #         output = self.decoder(torch.cat(all_spatial, dim=-1))
        #         output = self.decoder(spatial)
        output = torch.cat(all_spatial, dim=-1)

        return output

    def get_truth(self, batch):

        truth = torch.cat(
            [batch.layerless_true_edges, batch.layerless_true_edges.flip(0)], axis=-1
        )

        return truth

    def get_true_predictions(self, batch, e_spatial, e_bidir):

        # Calculate truth from intersection between Prediction graph and Truth graph
        if "weighting" in self.hparams["regime"]:
            weights_bidir = torch.cat([batch.weights, batch.weights])
            e_spatial, y, new_weights = graph_intersection(
                e_spatial, e_bidir, using_weights=True, weights_bidir=weights_bidir
            )
            new_weights = (
                new_weights.to(self.device) * self.hparams["weight"]
            )  # Weight positive examples

            return e_spatial, y, new_weights, weights_bidir

        else:
            e_spatial, y = graph_intersection(e_spatial, e_bidir)
            new_weights = y.to(self.device) * self.hparams["weight"]

            return e_spatial, y, new_weights, None

    def get_training_edges(self, truth, spatial, batch):

        # Instantiate empty prediction edge list
        e_spatial = torch.empty([2, 0], dtype=torch.int64, device=self.device)

        # Append random edges pairs (rp) for stability
        if "rp" in self.hparams["regime"]:
            n_random = int(self.hparams["randomisation"] * truth.shape[1])
            e_spatial = torch.cat(
                [
                    e_spatial,
                    torch.randint(
                        truth.min(), truth.max(), (2, n_random), device=self.device
                    ),
                ],
                axis=-1,
            )

        # Append Hard Negative Mining (hnm) with KNN graph
        if "hnm" in self.hparams["regime"]:
            e_spatial = torch.cat(
                [
                    e_spatial,
                    build_edges(spatial, self.hparams["r_train"], self.hparams["knn"]),
                ],
                axis=-1,
            )

        e_spatial, y, new_weights, weights_bidir = self.get_true_predictions(
            batch, e_spatial, truth
        )

        # Append all positive examples and their truth and weighting
        e_spatial = torch.cat(
            [
                e_spatial.to(self.device),
                truth.transpose(0, 1).repeat(1, 1).view(-1, 2).transpose(0, 1),
            ],
            axis=-1,
        )
        y = torch.cat([y.int(), torch.ones(truth.shape[1])])

        if "weighting" in self.hparams["regime"]:
            new_weights = torch.cat(
                [new_weights, weights_bidir * self.hparams["weight"]]
            )
        else:
            new_weights = torch.cat(
                [
                    new_weights,
                    torch.ones(truth.shape[1], device=self.device)
                    * self.hparams["weight"],
                ]
            )

        return e_spatial, y, new_weights

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
            spatial = self(torch.cat([batch.cell_data, batch.x], axis=-1))
        else:
            spatial = self(batch.x)

        # Instantiate bidirectional truth (since KNN prediction will be bidirectional)
        truth = self.get_truth(batch)

        # Get training set of neighbours
        e_spatial, y, new_weights = self.get_training_edges(truth, spatial, batch)

        loss = self.hinge_loss_fn(spatial, e_spatial, y, new_weights)

        self.log("train_loss", loss)

        return loss

    def hinge_loss_fn(self, spatial, e_spatial, y, new_weights):

        hinge = y.float().to(self.device)
        hinge[hinge == 0] = -1

        reference = spatial.index_select(0, e_spatial[1])
        neighbors = spatial.index_select(0, e_spatial[0])
        d = torch.sum((reference - neighbors) ** 2, dim=-1)

        new_weights[
            y == 0
        ] = 1  # Give negative examples a weight of 1 (note that there may still be TRUE examples that are weightless)
        d = d * new_weights

        loss = torch.nn.functional.hinge_embedding_loss(
            d, hinge, margin=self.hparams["margin"], reduction="mean"
        )

        return loss

    def grav_loss_fn(self, spatial, e_spatial, y, new_weights):

        reference = spatial.index_select(0, e_spatial[1])
        neighbors = spatial.index_select(0, e_spatial[0])
        d = torch.sum((reference - neighbors) ** 2, dim=-1)

        d_weight = torch.exp(-self.hparams["exp_coeff"] * d)

        loss = torch.nn.functional.binary_cross_entropy(
            d_weight, y.float().to(self.device)
        )

        return loss


class CheckpointGravNetLite(GravNetLite):
    def __init__(self, hparams):
        super().__init__(hparams)

        # Construct architecture
        # -------------------------

        # Encode input features to space co-ordinates
        self.space_encoder = make_mlp(
            hparams["hidden_channels"],
            [hparams["space_channels"]] * hparams["encoding_layers"],
            hidden_activation=hparams["hidden_activation"],
            output_activation=None,
            layer_norm=hparams["layernorm"],
        )

        # Encode input features to hidden features
        self.feature_encoder = make_mlp(
            hparams["in_channels"],
            [hparams["hidden_channels"]] * hparams["encoding_layers"],
            hidden_activation=hparams["hidden_activation"],
            output_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
        )

        # Encode hidden features to next-block hidden features
        self.feature_network = make_mlp(
            2 * hparams["hidden_channels"],
            [hparams["hidden_channels"]] * hparams["encoding_layers"],
            output_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
        )

        # Decode hidden features to output features
        self.decoder = make_mlp(
            (self.hparams["num_iterations"] + 1) * hparams["space_channels"],
            [hparams["hidden_channels"]] * hparams["decoding_layers"]
            + [hparams["output_channels"]],
            hidden_activation=hparams["hidden_activation"],
            output_activation=None,
            layer_norm=hparams["layernorm"],
        )

    #         self.decoder = make_mlp(hparams["space_channels"], [hparams["hidden_channels"]]*hparams["decoding_layers"] + [hparams["output_channels"]],
    #                                      hidden_activation=hparams["hidden_activation"], output_activation=None,
    #                                      layer_norm=hparams["layernorm"])

    def forward(self, x):

        # Encode to features

        features = self.feature_encoder(x)

        # Encode to space
        spatial = checkpoint(self.space_encoder, features)
        all_spatial = []
        all_spatial.append(spatial)

        for i in range(self.hparams["num_iterations"]):
            # Aggregate to messages
            edge_index = build_knn(spatial, self.hparams["hidden_knn"])
            start, end = edge_index

            reference = spatial.index_select(0, end)
            neighbors = spatial.index_select(0, start)

            d = torch.sum((reference - neighbors) ** 2, dim=-1)
            d_weight = torch.exp(-self.hparams["exp_coeff"] * d)

            weighted_sum_messages = scatter_add(
                features[start] * d_weight.unsqueeze(1), end, dim=0, dim_size=x.shape[0]
            )
            concated_messages = torch.cat((weighted_sum_messages, features), dim=-1)

            # Feature network
            new_features = checkpoint(self.feature_network, concated_messages)

            spatial = checkpoint(self.space_encoder, new_features)
            all_spatial.append(spatial)

        output = checkpoint(self.decoder, torch.cat(all_spatial, dim=-1))
        #         output = torch.cat(all_spatial, dim=-1)

        return output
