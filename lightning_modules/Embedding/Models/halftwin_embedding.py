# System imports
import sys
import os

# 3rd party imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from ..halftwin_base import HalfTwinEmbeddingBase
from torch.nn import Linear
import torch.nn as nn
from torch_cluster import radius_graph
import torch
from torch_geometric.data import DataLoader

# Local imports
from ...utils import make_mlp


class HalfTwinEmbedding(HalfTwinEmbeddingBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """

        # Construct the MLP architecture
        self.source_network = make_mlp(
            hparams["in_channels"],
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation="Tanh",
            layer_norm=True,
        )

        self.target_network = make_mlp(
            hparams["in_channels"],
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation="Tanh",
            layer_norm=True,
        )

        self.save_hyperparameters()

    def forward(self, x):
        #
        sources = self.source_network(x)
        targets = self.target_network(x)

        return sources, targets
