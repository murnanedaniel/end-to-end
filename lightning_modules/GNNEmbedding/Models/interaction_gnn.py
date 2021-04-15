import sys

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import torch
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch.utils.checkpoint import checkpoint

from ..gnn_base import GNNNodeEmbeddingBase, GNNEdgeEmbeddingBase
from ..toy_base import ToyGNNNodeEmbeddingBase
from ...utils import make_mlp


class InteractionEdgeEmbedding(GNNEdgeEmbeddingBase):

    """
    An interaction network class
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

        # Setup input network
        self.node_encoder = make_mlp(
            hparams["in_channels"],
            [hparams["hidden"]] * hparams["nb_node_layer"],
            output_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
        )

        # The edge network computes new edge features from connected nodes
        self.edge_encoder = make_mlp(
            2 * (hparams["hidden"]),
            [hparams["hidden"]] * hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

        # The edge network computes new edge features from connected nodes
        self.edge_network = make_mlp(
            3 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        # The node network computes new node features
        self.node_network = make_mlp(
            2 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        # Final edge output classification network
        self.output_edge_classifier = make_mlp(
            3 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_node_layer"] + [hparams["emb_dim"]],
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

    def forward(self, x, edge_index):

        start, end = edge_index

        # Encode the graph features into the hidden space
        x = self.node_encoder(x)
        e = self.edge_encoder(torch.cat([x[start], x[end]], dim=1))
        #         e = torch.sigmoid(e)

        # Save the inputs
        #         input_x = x
        #         input_e = e
        x_initial = x
        e_initial = e

        #         # Cocnatenate with initial latent space
        #         x = torch.cat([x, input_x], dim=-1)
        #         e = torch.cat([e, input_e], dim=-1)

        # Compute new node features
        edge_messages = scatter_add(e, end, dim=0, dim_size=x.shape[0]) + scatter_add(
            e, start, dim=0, dim_size=x.shape[0]
        )
        node_inputs = torch.cat([x, edge_messages], dim=-1)
        x = checkpoint(self.node_network, node_inputs)

        #         edge_outputs = []

        # Loop over iterations of edge and node networks
        for i in range(self.hparams["n_graph_iters"]):

            # Compute new edge features
            edge_inputs = torch.cat([x[start], x[end], e], dim=-1)
            e = checkpoint(self.edge_network, edge_inputs)

            # Cocnatenate with initial latent space
            #             x = torch.cat([x, input_x], dim=-1)
            #             e = torch.cat([e, input_e], dim=-1)
            x = x + x_initial
            e = e + e_initial
            #             e = torch.sigmoid(e)

            x_initial = x
            e_initial = e

            # Compute new node features
            edge_messages = scatter_add(
                e, end, dim=0, dim_size=x.shape[0]
            ) + scatter_add(e, start, dim=0, dim_size=x.shape[0])
            node_inputs = torch.cat([x, edge_messages], dim=-1)
            x = checkpoint(self.node_network, node_inputs)

        edge_inputs = torch.cat([x[start], x[end], e], dim=-1)
        e = checkpoint(self.output_edge_classifier, edge_inputs)

        return e, torch.ones(e.shape[0], device=self.device)


class InteractionNodeEmbedding(GNNNodeEmbeddingBase):

    """
    An interaction network class
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

        # Setup input network
        self.node_encoder = make_mlp(
            hparams["in_channels"],
            [hparams["hidden"]] * hparams["nb_node_layer"],
            output_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
        )

        # The edge network computes new edge features from connected nodes
        self.edge_encoder = make_mlp(
            2 * (hparams["hidden"]),
            [hparams["hidden"]] * hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

        # The edge network computes new edge features from connected nodes
        self.edge_network = make_mlp(
            3 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        # The node network computes new node features
        self.node_network = make_mlp(
            2 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        # Final edge output classification network
        self.output_edge_classifier = make_mlp(
            3 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_node_layer"] + [hparams["emb_dim"]],
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

    def forward(self, x, edge_index):

        start, end = edge_index

        # Encode the graph features into the hidden space
        x = self.node_encoder(x)
        e = self.edge_encoder(torch.cat([x[start], x[end]], dim=1))
        #         e = torch.sigmoid(e)

        # Save the inputs
        x_initial = x
        e_initial = e

        # Compute new node features
        edge_messages = scatter_add(e, end, dim=0, dim_size=x.shape[0]) + scatter_add(
            e, start, dim=0, dim_size=x.shape[0]
        )
        node_inputs = torch.cat([x, edge_messages], dim=-1)
        x = checkpoint(self.node_network, node_inputs)

        # Loop over iterations of edge and node networks
        for i in range(self.hparams["n_graph_iters"]):

            # Compute new edge features
            edge_inputs = torch.cat([x[start], x[end], e], dim=-1)
            e = checkpoint(self.edge_network, edge_inputs)

            # Cocnatenate with initial latent space
            x = x + x_initial
            e = e + e_initial
            #             e = torch.sigmoid(e)

            x_initial = x
            e_initial = e

            # Compute new node features
            edge_messages = scatter_add(
                e, end, dim=0, dim_size=x.shape[0]
            ) + scatter_add(e, start, dim=0, dim_size=x.shape[0])
            node_inputs = torch.cat([x, edge_messages], dim=-1)
            x = checkpoint(self.node_network, node_inputs)

        edge_inputs = torch.cat([x[start], x[end], e], dim=-1)
        e = checkpoint(self.output_edge_classifier, edge_inputs)

        return e, torch.ones(e.shape[0], device=self.device)


class GlobalInteractionNodeEmbedding(ToyGNNNodeEmbeddingBase):

    """
    An interaction network class
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

        # Setup input network
        self.node_encoder = make_mlp(
            hparams["in_channels"],
            [hparams["hidden"]] * hparams["nb_node_layer"],
            output_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
        )

        # The edge network computes new edge features from connected nodes
        self.edge_encoder = make_mlp(
            2 * (hparams["hidden"]),
            [hparams["hidden"]] * hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

        # The edge network computes new edge features from connected nodes
        self.edge_network = make_mlp(
            4 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        # The node network computes new node features
        self.node_network = make_mlp(
            2 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        # Final edge output classification network
        self.output_node_classifier = make_mlp(
            2 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_node_layer"] + [hparams["emb_dim"]],
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

    def forward(self, x, edge_index):

        start, end = edge_index

        # Encode the graph features into the hidden space
        x = self.node_encoder(x)
        e = self.edge_encoder(torch.cat([x[start], x[end]], dim=1))
        #         e = torch.sigmoid(e)

        # Loop over iterations of edge and node networks
        for i in range(self.hparams["n_graph_iters"]):

            # Save the inputs
            x_initial = x
            e_initial = e

            # Compute new node features
            edge_messages = scatter_add(
                e, end, dim=0, dim_size=x.shape[0]
            ) + scatter_add(e, start, dim=0, dim_size=x.shape[0])
            node_inputs = torch.cat([x, edge_messages], dim=-1)
            x = checkpoint(self.node_network, node_inputs)

            # Compute new edge features
            edge_mean = torch.mean(e, dim=0)
            edge_inputs = torch.cat(
                [x[start], x[end], e, edge_mean.repeat((e.shape[0], 1))], dim=-1
            )
            e = checkpoint(self.edge_network, edge_inputs)

            # Cocnatenate with initial latent space
            x = x + x_initial
            e = e + e_initial
        #             e = torch.sigmoid(e)

        edge_messages = scatter_add(e, end, dim=0, dim_size=x.shape[0]) + scatter_add(
            e, start, dim=0, dim_size=x.shape[0]
        )
        node_inputs = torch.cat([x, edge_messages], dim=-1)
        x = checkpoint(self.output_node_classifier, node_inputs)

        return x, torch.ones(e.shape[0], device=self.device)
