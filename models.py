import torch
import torch.nn as nn
from torch.nn import Linear
from toy_utils import *
from trainers import build_edges

from torch_scatter import scatter, segment_csr, scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_cluster import knn_graph, radius_graph
import torch_geometric

import faiss
res = faiss.StandardGpuResources()

class BatchNormEmbedding(torch.nn.Module):
    def __init__(self, in_channels, emb_hidden, nb_layer, emb_dim=3):
        super(BatchNormEmbedding, self).__init__()
        layers = [Linear(in_channels, emb_hidden)]
        ln = [Linear(emb_hidden, emb_hidden) for _ in range(nb_layer-1)]
        layers.extend(ln)
        self.layers = nn.ModuleList(layers)
        self.penultimate_layer = nn.Linear(emb_hidden, emb_hidden)
        self.emb_layer = nn.Linear(emb_hidden, emb_dim)
        self.norm = nn.LayerNorm(emb_hidden)
        self.bnorm = nn.BatchNorm1d(num_features=emb_hidden)
        self.act = nn.Tanh()
        # self.dropout = nn.Dropout(p=0.7)
#         self.mean = torch.FloatTensor(mean).to(torch.float)
#         self.std = torch.FloatTensor(std).to(torch.float)

    def forward(self, x):
#         hits = self.normalize(hits)
        for l in self.layers:
            x = l(x)
            x = self.act(x)
            x = self.bnorm(x)
            # hits = self.dropout(hits)
#         x = self.norm(x) #Option of LayerNorm
        x = self.act(self.penultimate_layer(x))
        x = self.emb_layer(x)
        return x

    def normalize(self, hits):
        try:
            hits = (hits-self.mean) / (self.std + 10**-9)
        except:
            self.mean = self.mean.to(device=hits.device)
            self.std  = self.std.to(device=hits.device)
            hits = (hits-self.mean) / (self.std + 10**-9)
        return hits
    
class Embedding(torch.nn.Module):
    def __init__(self, in_channels, emb_hidden, nb_layer, emb_dim=3):
        super(Embedding, self).__init__()
        layers = [Linear(in_channels, emb_hidden)]
        ln = [Linear(emb_hidden, emb_hidden) for _ in range(nb_layer-1)]
        layers.extend(ln)
        self.layers = nn.ModuleList(layers)
        self.emb_layer = nn.Linear(emb_hidden, emb_dim)
        self.norm = nn.LayerNorm(emb_hidden)
        self.act = nn.Tanh()

    def forward(self, x):
#         hits = self.normalize(hits)
        for l in self.layers:
            x = l(x)
            x = self.act(x)
#         x = self.norm(x) #Option of LayerNorm
        x = self.emb_layer(x)
        return x

    
class EdgeNetwork(nn.Module):
    """
    A module which computes weights for edges of the graph.
    For each edge, it selects the associated nodes' features
    and applies some fully-connected network layers with a final
    sigmoid activation.
    """
    def __init__(self, input_dim, hidden_dim=8, hidden_activation='Tanh',
                 layer_norm=True):
        super(EdgeNetwork, self).__init__()
        self.network = make_mlp(input_dim*2,
                                [hidden_dim, hidden_dim, hidden_dim, 1],
                                hidden_activation=hidden_activation,
                                output_activation=None,
                                layer_norm=layer_norm)

    def forward(self, x, edge_index):
        # Select the features of the associated nodes
        start, end = edge_index
        x1, x2 = x[start], x[end]
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
    def __init__(self, input_dim, output_dim, hidden_activation='Tanh',
                 layer_norm=True):
        super(NodeNetwork, self).__init__()
        self.network = make_mlp(input_dim*3, [output_dim]*4,
                                hidden_activation=hidden_activation,
                                output_activation=None,
                                layer_norm=layer_norm)

    def forward(self, x, e, edge_index):
        start, end = edge_index
        # Aggregate edge-weighted incoming/outgoing features
        mi = scatter_add(e[:, None] * x[start], end, dim=0, dim_size=x.shape[0])
        mo = scatter_add(e[:, None] * x[end], start, dim=0, dim_size=x.shape[0])
        node_inputs = torch.cat([mi, mo, x], dim=1)
        return self.network(node_inputs)

class ResAGNN(nn.Module):
    """
    Segment classification graph neural network model.
    Consists of an input network, an edge network, and a node network.
    """
    def __init__(self, in_channels=3, hidden_dim=8, n_graph_iters=3,
                 hidden_activation=torch.nn.Tanh, layer_norm=True):
        super(ResAGNN, self).__init__()
        self.n_graph_iters = n_graph_iters
        # Setup the input network
        self.input_network = make_mlp(in_channels, [hidden_dim],
                                      output_activation=hidden_activation,
                                      layer_norm=layer_norm)
        # Setup the edge network
        self.edge_network = EdgeNetwork(in_channels + hidden_dim, in_channels + hidden_dim,
                                        hidden_activation, layer_norm=layer_norm)
        # Setup the node layers
        self.node_network = NodeNetwork(in_channels + hidden_dim, hidden_dim,
                                        hidden_activation, layer_norm=layer_norm)
        
    def forward(self, inputs):
        """Apply forward pass of the model"""
        x = inputs.x
        edge_index = inputs.e
#         print(x.shape)
        x = self.input_network(x)
        # Shortcut connect the inputs onto the hidden representation
        x = torch.cat([x, inputs.x], dim=-1)
#         print(x.shape)
        
        # Loop over iterations of edge and node networks
        for i in range(self.n_graph_iters):
            x_inital = x
            
            # Apply edge network
            e = torch.sigmoid(self.edge_network(x, edge_index))
        
            # Apply node network
            x = self.node_network(x, e, edge_index)
            
            # Shortcut connect the inputs onto the hidden representation
            x = torch.cat([x, inputs.x], dim=-1)  
            
            x = x_inital + x
        
        return self.edge_network(x, edge_index)
    
class BiEmbeddingToAGNN(nn.Module):
    """
    Segment classification graph neural network model.
    Consists of an input network, an edge network, and a node network.
    """
    def __init__(self, in_channels=3, emb_hidden=32, nb_layer=4, emb_dim=8, hidden_dim=8, n_graph_iters=3,
                 hidden_activation=torch.nn.Tanh, layer_norm=True):
        super(BiEmbeddingToAGNN, self).__init__()
        self.n_graph_iters = n_graph_iters
        # Setup the input network
        self.input_network = make_mlp(emb_dim, [hidden_dim],
                                      output_activation=hidden_activation,
                                      layer_norm=layer_norm)
        # Setup the edge network
        self.edge_network = EdgeNetwork(in_channels + hidden_dim + emb_dim, in_channels + hidden_dim,
                                        hidden_activation, layer_norm=layer_norm)
        # Setup the node layers
        self.node_network = NodeNetwork(in_channels + hidden_dim + emb_dim, hidden_dim,
                                        hidden_activation, layer_norm=layer_norm)
        # Setup gravnet
        self.emb_network = Embedding(in_channels, emb_hidden, nb_layer, emb_dim)
        
#         self.spatial_norm = torch.nn.LayerNorm(emb_dim)
        
    def forward(self, inputs, r, small_nhood=False):
        """Apply forward pass of the model"""        
        x = inputs.x
#         print(x.shape)
        spatial = self.emb_network(x)
#         spatial = self.spatial_norm(spatial)
        
        if small_nhood: 
            edge_index = build_edges(spatial, r, 30, res)
        else:
            edge_index = build_edges(spatial, r, 100, res)
        edge_index = edge_index[:, ((inputs.layers[edge_index[1]] - inputs.layers[edge_index[0]]) == 1)]
#         print(edge_index.shape)
        x = self.input_network(spatial)
        # Shortcut connect the inputs onto the hidden representation
        x = torch.cat([x, spatial, inputs.x], dim=-1)
        
        # Loop over iterations of edge and node networks
        for i in range(self.n_graph_iters):
            x_inital = x
            
            # Apply edge network
            e = torch.sigmoid(self.edge_network(x, edge_index))
        
            # Apply node network
            x = self.node_network(x, e, edge_index)
            
            # Shortcut connect the inputs onto the hidden representation
            x = torch.cat([x, spatial, inputs.x], dim=-1)
             
            x = x_inital + x
        
        return self.edge_network(x, edge_index), spatial, edge_index, len(edge_index[0])/len(x)
    
class BiEmbeddingToBiAGNN(nn.Module):
    """
    Segment classification graph neural network model.
    Consists of an input network, an edge network, and a node network.
    """
    def __init__(self, in_channels=3, emb_hidden=32, nb_layer=4, emb_dim=8, r=0.2, hidden_dim=8, n_graph_iters=3,
                 hidden_activation=torch.nn.Tanh, layer_norm=True):
        super(BiEmbeddingToBiAGNN, self).__init__()
        self.n_graph_iters = n_graph_iters
        self.r = r
        # Setup the input network
        self.input_network = make_mlp(emb_dim, [hidden_dim],
                                      output_activation=hidden_activation,
                                      layer_norm=layer_norm)
        # Setup the edge network
        self.edge_network = EdgeNetwork(in_channels + hidden_dim + emb_dim, in_channels + hidden_dim,
                                        hidden_activation, layer_norm=layer_norm)
        # Setup the node layers
        self.node_network = NodeNetwork(in_channels + hidden_dim + emb_dim, hidden_dim,
                                        hidden_activation, layer_norm=layer_norm)
        # Setup gravnet
        self.emb_network = Embedding(in_channels, emb_hidden, nb_layer, emb_dim)
        
#         self.spatial_norm = torch.nn.LayerNorm(emb_dim)
        
    def forward(self, inputs):
        """Apply forward pass of the model"""        
        x = inputs.x
#         print(x.shape)
        spatial = self.emb_network(x)
#         spatial = self.spatial_norm(spatial)
        
        edge_index = build_edges(spatial, self.r, 100, res)
        edge_index = edge_index[:, ((inputs.layers[edge_index[1]] - inputs.layers[edge_index[0]]) == 1) | ((inputs.layers[edge_index[0]] - inputs.layers[edge_index[1]]) == 1)]
#         print(edge_index.shape)
        x = self.input_network(spatial)
        # Shortcut connect the inputs onto the hidden representation
        x = torch.cat([x, spatial, inputs.x], dim=-1)
        
        # Loop over iterations of edge and node networks
        for i in range(self.n_graph_iters):
            x_inital = x
            
            # Apply edge network
            e = torch.sigmoid(self.edge_network(x, edge_index))
        
            # Apply node network
            x = self.node_network(x, e, edge_index)
            
            # Shortcut connect the inputs onto the hidden representation
            x = torch.cat([x, spatial, inputs.x], dim=-1)
             
            x = x_inital + x
        
        return self.edge_network(x, edge_index), spatial, edge_index, len(edge_index[0])/len(x)

class EmbeddingToAGNNPretrained(nn.Module):
    """
    Segment classification graph neural network model.
    Consists of an input network, an edge network, and a node network.
    """
    def __init__(self, in_channels=3, emb_hidden=32, nb_layer=4, emb_dim=8, r=0.2, hidden_dim=8, n_graph_iters=3, pretrained_model=None, 
                 hidden_activation=torch.nn.Tanh, layer_norm=True):
        super(EmbeddingToAGNNPretrained, self).__init__()
        self.n_graph_iters = n_graph_iters
        self.r = r
        # Setup the input network
        self.input_network = make_mlp(emb_dim, [hidden_dim],
                                      output_activation=hidden_activation,
                                      layer_norm=layer_norm)
        # Setup the edge network
        self.edge_network = EdgeNetwork(in_channels + hidden_dim + emb_dim, in_channels + hidden_dim,
                                        hidden_activation, layer_norm=layer_norm)
        # Setup the node layers
        self.node_network = NodeNetwork(in_channels + hidden_dim + emb_dim, hidden_dim,
                                        hidden_activation, layer_norm=layer_norm)
        # Setup pretrained embedding
        self.emb_network = Embedding(in_channels, emb_hidden, nb_layer, emb_dim)
        self.emb_network.load_state_dict(pretrained_model.state_dict())
        
#         self.spatial_norm = torch.nn.LayerNorm(emb_dim)
        
    def forward(self, inputs):
        """Apply forward pass of the model"""        
        x = inputs.x
#         print(x.shape)
        spatial = self.emb_network(x)
#         spatial = self.spatial_norm(spatial)
        
        edge_index = build_edges(spatial, self.r, 50, res)
        edge_index = edge_index[:, (inputs.layers[edge_index[1]] - inputs.layers[edge_index[0]]) == 1]
#         print(edge_index.shape)
        x = self.input_network(spatial)
        # Shortcut connect the inputs onto the hidden representation
        x = torch.cat([x, spatial, inputs.x], dim=-1)
        
        # Loop over iterations of edge and node networks
        for i in range(self.n_graph_iters):
            x_inital = x
            
            # Apply edge network
            e = torch.sigmoid(self.edge_network(x, edge_index))
        
            # Apply node network
            x = self.node_network(x, e, edge_index)
            
            # Shortcut connect the inputs onto the hidden representation
            x = torch.cat([x, spatial, inputs.x], dim=-1)
             
            x = x_inital + x
        
        return self.edge_network(x, edge_index), spatial, edge_index, len(edge_index[0])/len(x)
    
class EmbAGNN(nn.Module):
    """
    Segment classification graph neural network model.
    Consists of an input network, an edge network, and a node network.
    """
    def __init__(self, in_channels=3, emb_hidden=32, nb_layer=4, emb_dim=8, r=0.2, k=20, hidden_dim=8, n_graph_iters=3,
                 hidden_activation=torch.nn.Tanh, layer_norm=True):
        super(EmbAGNN, self).__init__()
        self.n_graph_iters = n_graph_iters
        self.r = r
        self.k = k
        # Setup the input network
        self.input_spatial_network = make_mlp(in_channels, [emb_dim],
                                      output_activation=hidden_activation,
                                      layer_norm=layer_norm)
        
        self.input_feature_network = make_mlp(in_channels, [hidden_dim],
                                      output_activation=hidden_activation,
                                      layer_norm=layer_norm)
        
#         self.combine_network = make_mlp(emb_dim + hidden_dim, [hidden_dim],
#                                       output_activation=hidden_activation,
#                                       layer_norm=layer_norm)
        # Setup the edge network
        self.edge_network = EdgeNetwork(hidden_dim, hidden_dim,
                                        hidden_activation, layer_norm=layer_norm)
        # Setup the node layers
        self.node_network = NodeNetwork(hidden_dim, hidden_dim,
                                        hidden_activation, layer_norm=layer_norm)
        # Setup gravnet
        self.emb_network = Embedding(emb_dim + hidden_dim + in_channels, emb_hidden, nb_layer, emb_dim)
        
#         self.spatial_norm = torch.nn.LayerNorm(emb_dim)
        
    def forward(self, inputs):
        """Apply forward pass of the model"""        
        x = inputs.x
#         print(x.shape)
        spatial = self.input_spatial_network(x)
        features = self.input_feature_network(x)
        spatial = self.emb_network(torch.cat([inputs.x, features, spatial], axis=-1))

        edge_index = radius_graph(spatial, r=self.r, batch=inputs.batch, loop=False, max_num_neighbors=30)
        
        # Loop over iterations of edge and node networks
        for i in range(self.n_graph_iters):
            features_inital = features
            
            # Apply edge network
            e = torch.sigmoid(self.edge_network(features, edge_index))
        
            # Apply node network
            features = self.node_network(features, e, edge_index)
            spatial = self.emb_network(torch.cat([inputs.x, features, spatial], axis=-1))
       
            edge_index = radius_graph(spatial, r=self.r, batch=inputs.batch, loop=False, max_num_neighbors=30)
             
            features = features_inital + features
        
        return self.edge_network(features, edge_index), spatial, edge_index

class EmbAGNNRecluster(nn.Module):
    """
    Segment classification graph neural network model.
    Consists of an input network, an edge network, and a node network.
    """
    def __init__(self, in_channels=3, emb_hidden=32, nb_layer=4, emb_dim=8, r=0.2, k=20, hidden_dim=8, n_graph_iters=3, pretrained_model=None, 
                 hidden_activation=torch.nn.Tanh, layer_norm=True):
        super(EmbAGNNRecluster, self).__init__()
        self.n_graph_iters = n_graph_iters
        self.r = r
        self.k = k
        # Setup the input network
#         self.input_spatial_network = make_mlp(in_channels, [emb_dim],
#                                       output_activation=hidden_activation,
#                                       layer_norm=layer_norm)
        
        self.input_feature_network = make_mlp(emb_dim + in_channels, [hidden_dim],
                                      output_activation=hidden_activation,
                                      layer_norm=layer_norm)
        
#         self.combine_network = make_mlp(emb_dim + hidden_dim, [hidden_dim],
#                                       output_activation=hidden_activation,
#                                       layer_norm=layer_norm)
        # Setup the edge network
        self.edge_network = EdgeNetwork(hidden_dim, hidden_dim,
                                        hidden_activation, layer_norm=layer_norm)
        # Setup the node layers
        self.node_network = NodeNetwork(hidden_dim, hidden_dim,
                                        hidden_activation, layer_norm=layer_norm)
        
        # Setup pretrained embedding & new embedding        
        
        self.emb_network_1 = Embedding(in_channels, emb_hidden, nb_layer, emb_dim)
        self.emb_network_1.load_state_dict(pretrained_model.state_dict())
        
        self.emb_network_2 = Embedding(in_channels + emb_dim + hidden_dim, emb_hidden, nb_layer, emb_dim)
        
#         self.spatial_norm = torch.nn.LayerNorm(emb_dim)
        
    def forward(self, inputs):
        """Apply forward pass of the model"""   
#         print(inputs.x)
        spatial = self.emb_network_1(inputs.x)
    
#         print(spatial.shape)
        edge_index = build_edges(spatial, self.r, 50, res)
        edge_index = edge_index[:, (inputs.layers[edge_index[1]] - inputs.layers[edge_index[0]]) == 1]
        
        features = self.input_feature_network(torch.cat([spatial, inputs.x], dim=-1))
#         print(features.shape)
        # Shortcut connect the inputs onto the hidden representation
#         print(features.shape)
        # Loop over iterations of edge and node networks
        for i in range(self.n_graph_iters//2):
            features_initial = features
            
            # Apply edge network
            e = torch.sigmoid(self.edge_network(features, edge_index))
        
            # Apply node network
            features = self.node_network(features, e, edge_index)
            features = features + features_initial
            
#             print(features.shape)
            
        spatial = self.emb_network_2(torch.cat([spatial, inputs.x, features], dim=-1))
        edge_index = build_edges(spatial, self.r, 50, res)
        edge_index = edge_index[:, (inputs.layers[edge_index[1]] - inputs.layers[edge_index[0]]) == 1]
             
        for i in range(self.n_graph_iters//2):
            features_initial = features
            
            # Apply edge network
            e = torch.sigmoid(self.edge_network(features, edge_index))
        
            # Apply node network
            features = self.node_network(features, e, edge_index)
            features = features + features_initial
            
        return self.edge_network(features, edge_index), spatial, edge_index, len(edge_index[0])/len(spatial)
    
    
class MultiNoiseLoss(nn.Module):
    def __init__(self, n_losses):
        super(MultiNoiseLoss, self).__init__()
        self.noise_params = torch.rand(n_losses, requires_grad=True, device="cuda:0")
    
    def forward(self, losses):
        
        total_loss = 0
        for i, loss in enumerate(losses):
            total_loss += (1/torch.square(self.noise_params[i]))*loss + torch.log(self.noise_params[i])
        
        return total_loss