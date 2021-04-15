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


class VanillaGravNet(GravNetBase):
    def __init__(self, hparams):
        super().__init__(hparams)

        # Construct architecture
        # -------------------------

        # Encode input features to space co-ordinates
        self.space_encoder = make_mlp(
            hparams["in_channels"],
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

        # Encode hidden features to next-block space features
        self.space_network = make_mlp(
            hparams["output_channels"],
            [hparams["space_channels"]] * hparams["encoding_layers"],
            output_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
        )

        # Encode hidden features to next-block hidden features
        self.feature_network = make_mlp(
            hparams["output_channels"],
            [hparams["hidden_channels"]] * hparams["encoding_layers"],
            output_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
        )

        # Decode hidden features to output features
        self.decoder = make_mlp(
            2 * hparams["hidden_channels"],
            [8 * hparams["hidden_channels"]] * hparams["decoding_layers"]
            + [hparams["output_channels"]],
            hidden_activation=hparams["hidden_activation"],
            output_activation=None,
            layer_norm=hparams["layernorm"],
        )

    def forward(self, x):

        # Encode all features
        spatial = self.space_encoder(x)
        all_output_features = []
        #         print("1:", torch.cuda.max_memory_allocated()/1024**3)

        #         edge_index = build_edges(spatial, self.hparams["hidden_r"], self.hparams["hidden_knn"])
        edge_index = build_knn(spatial, self.hparams["hidden_knn"])
        start, end = edge_index

        #         print("2:", torch.cuda.max_memory_allocated()/1024**3)

        reference = spatial.index_select(0, end)
        neighbors = spatial.index_select(0, start)

        d = torch.sum((reference - neighbors) ** 2, dim=-1)
        d_weight = torch.exp(-self.hparams["exp_coeff"] * d)

        #         print("3:", torch.cuda.max_memory_allocated()/1024**3)

        hidden_features = self.feature_encoder(x)

        weighted_sum_messages = scatter_add(
            hidden_features[start] * d_weight.unsqueeze(1),
            end,
            dim=0,
            dim_size=x.shape[0],
        )
        #         weighted_max_messages, _ = scatter_max(hidden_features[start] * d_weight.unsqueeze(1), end, dim=0, dim_size=x.shape[0])
        concated_messages = torch.cat((weighted_sum_messages, hidden_features), dim=-1)

        new_features = self.decoder(concated_messages)
        all_output_features.append(new_features)
        #         print("4:", torch.cuda.max_memory_allocated()/1024**3)

        # GRAVNET FOR LOOP HERE
        for i in range(self.hparams["num_iterations"]):
            spatial = self.space_network(new_features)
            edge_index = build_knn(spatial, self.hparams["hidden_knn"])

            start, end = edge_index

            reference = spatial.index_select(0, end)
            neighbors = spatial.index_select(0, start)

            d = torch.sum((reference - neighbors) ** 2, dim=-1)
            d_weight = torch.exp(-self.hparams["exp_coeff"] * d)

            hidden_features = self.feature_network(new_features)

            weighted_sum_messages = scatter_add(
                hidden_features[start] * d_weight.unsqueeze(1),
                end,
                dim=0,
                dim_size=x.shape[0],
            )
            #             weighted_max_messages, _ = scatter_max(hidden_features[start] * d_weight.unsqueeze(1), end, dim=0, dim_size=x.shape[0])
            concated_messages = torch.cat(
                (weighted_sum_messages, hidden_features), dim=-1
            )

            new_features = self.decoder(concated_messages)
            all_output_features.append(new_features)

        return torch.cat(all_output_features, dim=-1)


class CheckpointVanillaGravNet(GravNetBase):
    def __init__(self, hparams):
        super().__init__(hparams)

        # Construct architecture
        # -------------------------

        # Encode input features to space co-ordinates
        self.space_encoder = make_mlp(
            hparams["in_channels"],
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

        # Encode hidden features to next-block space features
        self.space_network = make_mlp(
            hparams["output_channels"],
            [hparams["space_channels"]] * hparams["encoding_layers"],
            output_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
        )

        # Encode hidden features to next-block hidden features
        self.feature_network = make_mlp(
            hparams["output_channels"],
            [hparams["hidden_channels"]] * hparams["encoding_layers"],
            output_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
        )

        # Decode hidden features to output features
        self.decoder = make_mlp(
            2 * hparams["hidden_channels"],
            [8 * hparams["hidden_channels"]] * hparams["decoding_layers"]
            + [hparams["output_channels"]],
            hidden_activation=hparams["hidden_activation"],
            output_activation=None,
            layer_norm=hparams["layernorm"],
        )

    def forward(self, x):

        # Encode all features
        spatial = self.space_encoder(x)
        all_output_features = []
        #         print("1:", torch.cuda.max_memory_allocated()/1024**3)

        #         edge_index = build_edges(spatial, self.hparams["hidden_r"], self.hparams["hidden_knn"])
        edge_index = build_knn(spatial, self.hparams["hidden_knn"])
        start, end = edge_index

        #         print("2:", torch.cuda.max_memory_allocated()/1024**3)

        reference = spatial.index_select(0, end)
        neighbors = spatial.index_select(0, start)

        d = torch.sum((reference - neighbors) ** 2, dim=-1)
        d_weight = torch.exp(-self.hparams["exp_coeff"] * d)

        #         print("3:", torch.cuda.max_memory_allocated()/1024**3)

        hidden_features = self.feature_encoder(x)

        weighted_sum_messages = scatter_add(
            hidden_features[start] * d_weight.unsqueeze(1),
            end,
            dim=0,
            dim_size=x.shape[0],
        )
        #         weighted_max_messages, _ = scatter_max(hidden_features[start] * d_weight.unsqueeze(1), end, dim=0, dim_size=x.shape[0])
        concated_messages = torch.cat((weighted_sum_messages, hidden_features), dim=-1)

        new_features = self.decoder(concated_messages)
        all_output_features.append(new_features)
        #         print("4:", torch.cuda.max_memory_allocated()/1024**3)

        # GRAVNET FOR LOOP HERE
        for i in range(self.hparams["num_iterations"]):
            spatial = self.space_network(new_features)
            edge_index = build_knn(spatial, self.hparams["hidden_knn"])

            start, end = edge_index

            reference = spatial.index_select(0, end)
            neighbors = spatial.index_select(0, start)

            d = torch.sum((reference - neighbors) ** 2, dim=-1)
            d_weight = torch.exp(-self.hparams["exp_coeff"] * d)

            hidden_features = checkpoint(self.feature_encoder, new_features)

            weighted_sum_messages = scatter_add(
                hidden_features[start] * d_weight.unsqueeze(1),
                end,
                dim=0,
                dim_size=x.shape[0],
            )
            #             weighted_max_messages, _ = scatter_max(hidden_features[start] * d_weight.unsqueeze(1), end, dim=0, dim_size=x.shape[0])
            concated_messages = torch.cat(
                (weighted_sum_messages, hidden_features), dim=-1
            )

            new_features = checkpoint(self.decoder, concated_messages)
            all_output_features.append(new_features)

        return torch.cat(all_output_features, dim=-1)
