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

class EdgeNetwork(nn.Module):
    """
    A module which computes weights for edges of the graph.
    For each edge, it selects the associated nodes' features
    and applies some fully-connected network layers with a final
    sigmoid activation.
    """
    def __init__(self, input_dim, hidden_dim, nb_layers, hidden_activation='Tanh',
                 layer_norm=True):
        super(EdgeNetwork, self).__init__()
        self.network = make_mlp(input_dim*2,
                                [hidden_dim]*nb_layers+[1],
                                hidden_activation=hidden_activation,
                                output_activation=None,
                                layer_norm=layer_norm)

    def forward(self, x, edge_index):
        # Select the features of the associated nodes
        start, end = edge_index
        edge_inputs = torch.cat([x[start], x[end]], dim=1)
        return self.network(edge_inputs).squeeze(-1)

class NodeNetwork(nn.Module):
    """
    A module which computes new node features on the graph.
    For each node, it aggregates the neighbor node features
    (separately on the input and output side), and combines
    them with the node's previous features in a fully-connected
    network to compute the new features.
    """
    def __init__(self, input_dim, output_dim, nb_layers, hidden_activation='Tanh',
                 layer_norm=True):
        super(NodeNetwork, self).__init__()
        self.network = make_mlp(input_dim*2, [output_dim]*nb_layers,
                                hidden_activation=hidden_activation,
                                output_activation=None,
                                layer_norm=layer_norm)

    def forward(self, x, e, edge_index):
        start, end = edge_index
        
        mi = scatter_add(e[:, None] * x[start], end, dim=0, dim_size=x.shape[0]) + scatter_add(e[:, None] * x[end], start, dim=0, dim_size=x.shape[0])
#         node_inputs = mi + x
        node_inputs = torch.cat([mi, x], dim=1)
        return self.network(node_inputs)

class NodeMeanNetwork(nn.Module):
    """
    A module which computes new node features on the graph.
    For each node, it aggregates the neighbor node features
    (separately on the input and output side), and combines
    them with the node's previous features in a fully-connected
    network to compute the new features.
    """
    def __init__(self, input_dim, output_dim, nb_layers, hidden_activation='Tanh',
                 layer_norm=True):
        super(NodeMeanNetwork, self).__init__()
        self.network = make_mlp(input_dim*2, [output_dim]*nb_layers,
                                hidden_activation=hidden_activation,
                                output_activation=None,
                                layer_norm=layer_norm)

    def forward(self, x, e, edge_index):
        start, end = edge_index
        
        messages_in = scatter_add(e[:, None] * x[start], end, dim=0, dim_size=x.shape[0]) + scatter_add(e[:, None] * x[end], start, dim=0, dim_size=x.shape[0])
        combined_message = messages_in + x
#         combined_message = torch.cat([messages_in, x], dim=1)
        node_mean = torch.mean(combined_message, dim=0)
        node_inputs = torch.cat([combined_message, node_mean.repeat((combined_message.shape[0], 1))], dim=-1)
        return self.network(node_inputs)
    
       
        
class AttentionNodeEmbedding(ToyGNNNodeEmbeddingBase):

    def __init__(self, hparams):
        super().__init__(hparams)
        '''
        Initialise the Lightning Module that can scan over different GNN training regimes
        '''

        # Setup input network
        self.input_network = make_mlp(hparams["in_channels"], [hparams["hidden"]]*hparams["nb_edge_layer"],
                                      output_activation=hparams["hidden_activation"],
                                      layer_norm=hparams["layernorm"])
        
        # Setup the edge network
        self.edge_network = EdgeNetwork(hparams["hidden"], hparams["in_channels"] + hparams["hidden"],
                                        hparams["nb_edge_layer"], hparams["hidden_activation"], hparams["layernorm"])
        
        # Setup the node layers
        self.node_network = NodeNetwork(hparams["hidden"], hparams["hidden"],
                                        hparams["nb_node_layer"], hparams["hidden_activation"], hparams["layernorm"])

        # Setup embedding network
        self.embedding_network = make_mlp(hparams["hidden"], [hparams["hidden"]]*hparams["nb_emb_layer"]+[hparams["emb_dim"]],
                                      output_activation=None,
                                      layer_norm=False)
        
    def forward(self, x, edge_index):
                
        input_x = x

        x = self.input_network(x)

        # Shortcut connect the inputs onto the hidden representation
#         x = torch.cat([x, input_x], dim=-1)

        # Loop over iterations of edge and node networks
        for i in range(self.hparams["n_graph_iters"]):
            x_inital = x

            # Apply edge network
            e = self.edge_network(x, edge_index)

            # Apply node network
            x = self.node_network(x, torch.sigmoid(e), edge_index)

            # Shortcut connect the inputs onto the hidden representation
#             x = torch.cat([x, input_x], dim=-1)

            # Residual connection
            x = x_inital + x
            
        x = self.embedding_network(x)
            
        return x, e
    
class GlobalAttentionNodeEmbedding(GNNNodeEmbeddingBase):

    def __init__(self, hparams):
        super().__init__(hparams)
        '''
        Initialise the Lightning Module that can scan over different GNN training regimes
        '''

        # Setup input network
        self.input_network = make_mlp(hparams["in_channels"], [hparams["hidden"]]*hparams["nb_edge_layer"],
                                      output_activation=hparams["hidden_activation"],
                                      layer_norm=hparams["layernorm"])
        
        # Setup the edge network
        self.edge_network = EdgeNetwork(hparams["hidden"], hparams["in_channels"] + hparams["hidden"],
                                        hparams["nb_edge_layer"], hparams["hidden_activation"], hparams["layernorm"])
        
        # Setup the node layers
        self.node_network = NodeMeanNetwork(hparams["hidden"], hparams["hidden"],
                                        hparams["nb_node_layer"], hparams["hidden_activation"], hparams["layernorm"])

        # Setup embedding network
        self.embedding_network = make_mlp(hparams["hidden"], [hparams["hidden"]]*hparams["nb_emb_layer"]+[hparams["emb_dim"]],
                                      output_activation=None,
                                      layer_norm=False)
        
    def forward(self, x, edge_index):
                
        input_x = x

        x = self.input_network(x)

        # Shortcut connect the inputs onto the hidden representation
#         x = torch.cat([x, input_x], dim=-1)

        # Loop over iterations of edge and node networks
        for i in range(self.hparams["n_graph_iters"]):
            x_inital = x

            # Apply edge network
            e = self.edge_network(x, edge_index)

            # Apply node network
            x = self.node_network(x, torch.sigmoid(e), edge_index)

            # Shortcut connect the inputs onto the hidden representation
#             x = torch.cat([x, input_x], dim=-1)

            # Residual connection
            x = x_inital + x
            
        x = self.embedding_network(x)
            
        return x, e
    
    
class AttentionEdgeEmbedding(GNNEdgeEmbeddingBase):

    def __init__(self, hparams):
        super().__init__(hparams)
        '''
        Initialise the Lightning Module that can scan over different GNN training regimes
        '''

        # Setup input network
        self.input_network = make_mlp(hparams["in_channels"], [hparams["hidden"]]*hparams["nb_edge_layer"],
                                      output_activation=hparams["hidden_activation"],
                                      layer_norm=hparams["layernorm"])
        
        # Setup the edge network
        self.edge_network = EdgeNetwork(hparams["hidden"], hparams["in_channels"] + hparams["hidden"],
                                        hparams["nb_edge_layer"], hparams["hidden_activation"], hparams["layernorm"])
        
        # Setup the node layers
        self.node_network = NodeNetwork(hparams["hidden"], hparams["hidden"],
                                        hparams["nb_node_layer"], hparams["hidden_activation"], hparams["layernorm"])

        # Setup embedding network
        self.embedding_network = make_mlp(hparams["hidden"]*2, [hparams["hidden"]]*hparams["nb_emb_layer"]+[hparams["emb_dim"]],
                                      output_activation=None,
                                      layer_norm=False)
        
    def forward(self, x, edge_index):
                
        start, end = edge_index
        input_x = x

        x = self.input_network(x)

        # Shortcut connect the inputs onto the hidden representation
#         x = torch.cat([x, input_x], dim=-1)

        # Loop over iterations of edge and node networks
        for i in range(self.hparams["n_graph_iters"]):
            x_inital = x

            # Apply edge network
            e = self.edge_network(x, edge_index)

            # Apply node network
            x = self.node_network(x, torch.sigmoid(e), edge_index)

            # Shortcut connect the inputs onto the hidden representation
#             x = torch.cat([x, input_x], dim=-1)

            # Residual connection
            x = x_inital + x
        
    
        edge_inputs = torch.cat([x[start], x[end]], dim=1)
        
        x = self.embedding_network(edge_inputs)
            
        return x, e
