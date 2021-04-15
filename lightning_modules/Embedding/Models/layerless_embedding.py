# System imports
import sys
import os

# 3rd party imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from ..embedding_base import EmbeddingBase, AugmentedEmbeddingBase, TripletEmbeddingBase
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius_graph
import torch
from torch_geometric.data import DataLoader

# Local imports
from ...utils import make_mlp


class LayerlessEmbedding(AugmentedEmbeddingBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """

        # Construct the MLP architecture
        self.emb_network = make_mlp(
            hparams["in_channels"],
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation="Tanh",
            layer_norm=True,
        )

        self.save_hyperparameters()

    def forward(self, x):
        #
        x = self.emb_network(x)

        if "norm" in self.hparams["regime"]:
            x = F.normalize(x, p=2)

        return x


class TripletEmbedding(TripletEmbeddingBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """

        # Construct the MLP architecture
        self.emb_network = make_mlp(
            hparams["in_channels"],
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation="Tanh",
            layer_norm=True,
        )

        self.save_hyperparameters()

    def forward(self, x):
        #
        x = self.emb_network(x)

        if "norm" in self.hparams["regime"]:
            x = F.normalize(x)

        return x
