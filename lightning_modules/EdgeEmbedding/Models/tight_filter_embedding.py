# System imports
import sys
import os

# 3rd party imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from ..filterEmbedding_base import TightFilterEdgeEmbeddingBase
from torch.nn import Linear
import torch.nn as nn
from torch_cluster import radius_graph
import torch
from torch_geometric.data import DataLoader

# Local imports
from ...utils import graph_intersection, make_mlp

class TightFilterEdgeEmbedding(TightFilterEdgeEmbeddingBase):

    def __init__(self, hparams):
        super().__init__(hparams)
        '''
        Initialise the Lightning Module that can scan over different embedding training regimes
        '''

        # Construct the MLP architecture
        layers = [Linear(hparams["in_channels"]*2, hparams["emb_hidden"])]
        ln = [Linear(hparams["emb_hidden"], hparams["emb_hidden"]) for _ in range(hparams["nb_layer"]-1)]
        layers.extend(ln)
        self.layers = nn.ModuleList(layers)
        self.emb_layer = nn.Linear(hparams["emb_hidden"], hparams["emb_dim"])
        self.filter_layer = nn.Linear(hparams["emb_hidden"], 1)
        self.norm = nn.LayerNorm(hparams["emb_hidden"])
        self.act = nn.Tanh()
        self.save_hyperparameters()
        
#         self.encoder = make_mlp(hparams["in_channels"]*2,
#                                 [hparams["emb_hidden"]]*nb_layers,
#                                 hidden_activation="Tanh",
#                                 output_activation=None,
#                                 layer_norm=False)
        
#         self.emb_final = make_mlp(hparams["in_channels"]*2,
#                                 [8],
#                                 hidden_activation="Tanh",
#                                 output_activation=None,
#                                 layer_norm=False)
        
#         self.filter_final = make_mlp(hparams["in_channels"]*2,
#                                 [1],
#                                 hidden_activation="Tanh",
#                                 output_activation=None,
#                                 layer_norm=False)

    def forward(self, x):
#         hits = self.normalize(hits)
        for l in self.layers:
            x = l(x)
            x = self.act(x)
#         x = self.norm(x) #Option of LayerNorm
        
        emb = self.emb_layer(x)
        score = self.filter_layer(x)
        
        return emb, score
    
class LooseFilterEdgeEmbedding(TightFilterEdgeEmbeddingBase):

    def __init__(self, hparams):
        super().__init__(hparams)
        '''
        Initialise the Lightning Module that can scan over different embedding training regimes
        '''
        self.emb_encoder = make_mlp(hparams["in_channels"]*2,
                                [hparams["emb_hidden"]]*(hparams["nb_embedding_layers"]),
                                hidden_activation="Tanh",
                                output_activation=None,
                                layer_norm=False)
        
        self.emb_network = make_mlp(hparams["emb_hidden"],
                                [hparams["emb_hidden"]]*(hparams["nb_embedding_layers"])+[hparams["emb_dim"]],
                                hidden_activation="Tanh",
                                output_activation=None,
                                layer_norm=False)
        
        self.filter_network = make_mlp(hparams["in_channels"]*2,
                                [hparams["filter_hidden"]]*hparams["nb_filter_layers"]+[1],
                                hidden_activation="Tanh",
                                output_activation=None,
                                layer_norm=True)
        
        self.save_hyperparameters()

    def forward(self, x):

        # Encode the edge features into a latent space
        emb = self.emb_encoder(x)
        
        # Score the edge with the filter
        score = self.filter_network(x)
        
        # Give "attention" to true edges, and embed in final latent space
        emb = self.emb_network(score * emb)
        
        return emb, score

