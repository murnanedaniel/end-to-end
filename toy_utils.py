"""
This file contains some common helper code for the analysis notebooks.
"""

# System
import os
import sys
import yaml
import pickle
from collections import namedtuple

# Externals
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import torch.nn as nn
from torch_geometric.data import Data
from sklearn.cluster import DBSCAN
from sklearn import metrics
import scipy as sp
# from apex import amp, optimizers

# Locals
from torch_geometric.data import Batch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sys.path.append('..')
import faiss
res = faiss.StandardGpuResources()
from trainers import build_edges

sig = torch.nn.Sigmoid()

# Metrics

def evaluate_embfilter_f1(test_loader, model, r_min, r_max, r_step, t_min, t_max, t_step):
    model.eval()

    total_true_positive, total_positive, total_true = np.zeros((int(np.ceil((r_max - r_min)/r_step)), int(np.ceil((t_max - t_min)/t_step)))), np.zeros((int(np.ceil((r_max - r_min)/r_step)), int(np.ceil((t_max - t_min)/t_step)))), np.zeros((int(np.ceil((r_max - r_min)/r_step)), int(np.ceil((t_max - t_min)/t_step))))

    for batch in test_loader:

        data = batch.to(device)
        pred, spatial, e = model(data) 
        reference = spatial.index_select(0, e[1])
        neighbors = spatial.index_select(0, e[0])
        d = torch.sqrt(torch.sum((reference - neighbors)**2, dim=-1))

        for i, r in enumerate(np.arange(r_min, r_max, r_step)):  
    #         print(i)
            r_pred = pred[d<r]
            r_e = e[:,d<r]

            y = batch.pid[r_e[0]] == batch.pid[r_e[1]]

            true = 2*len(batch.true_edges)
            edge_true, edge_false = y.float() > 0.5, y.float() < 0.5
            for j, t in enumerate(np.arange(t_min, t_max, t_step)):  
                edge_positive, edge_negative = sig(r_pred) > t, sig(r_pred) < t
                total_true_positive[i][j] += (edge_true & edge_positive).sum().item()
                total_true[i][j] += true
                total_positive[i][j] += edge_positive.sum().item()

    pur, eff = total_true_positive / total_positive, total_true_positive / total_true
    f1 = 2*pur*eff/(pur+eff) 
    return pur, eff, f1

def evaluate_biembagnn_f1(test_loader, model, r_min, r_max, r_step, t_min, t_max, t_step):
    model.eval()

    total_true_positive, total_positive, total_true = np.zeros((int(np.ceil((r_max - r_min)/r_step)), int(np.ceil((t_max - t_min)/t_step)))), np.zeros((int(np.ceil((r_max - r_min)/r_step)), int(np.ceil((t_max - t_min)/t_step)))), np.zeros((int(np.ceil((r_max - r_min)/r_step)), int(np.ceil((t_max - t_min)/t_step))))

    for batch in test_loader:

        data = batch.to(device)
        pred, spatial, e, _ = model(data) 
        reference = spatial.index_select(0, e[1])
        neighbors = spatial.index_select(0, e[0])
        d = torch.sqrt(torch.sum((reference - neighbors)**2, dim=-1))

        for i, r in enumerate(np.arange(r_min, r_max, r_step)):  
    #         print(i)
            r_pred = pred[d<r]
            r_e = e[:,d<r]

            y = batch.pid[r_e[0]] == batch.pid[r_e[1]]

            true = 2*len(batch.true_edges)
            edge_true, edge_false = y.float() > 0.5, y.float() < 0.5
            for j, t in enumerate(np.arange(t_min, t_max, t_step)):  
                edge_positive, edge_negative = sig(r_pred) > t, sig(r_pred) < t
                total_true_positive[i][j] += (edge_true & edge_positive).sum().item()
                total_true[i][j] += true
                total_positive[i][j] += edge_positive.sum().item()

    pur, eff = total_true_positive / total_positive, total_true_positive / total_true
    f1 = 2*pur*eff/(pur+eff) 
    return pur, eff, f1

def evaluate_embagnn_f1(test_loader, model, r_min, r_max, r_step, t_min, t_max, t_step):
    model.eval()

    total_true_positive, total_positive, total_true = np.zeros((int(np.ceil((r_max - r_min)/r_step)), int(np.ceil((t_max - t_min)/t_step)))), np.zeros((int(np.ceil((r_max - r_min)/r_step)), int(np.ceil((t_max - t_min)/t_step)))), np.zeros((int(np.ceil((r_max - r_min)/r_step)), int(np.ceil((t_max - t_min)/t_step))))

    for batch in test_loader:

        data = batch.to(device)
        pred, spatial, e, _ = model(data) 
        reference = spatial.index_select(0, e[1])
        neighbors = spatial.index_select(0, e[0])
        d = torch.sqrt(torch.sum((reference - neighbors)**2, dim=-1))

        for i, r in enumerate(np.arange(r_min, r_max, r_step)):  
    #         print(i)
            r_pred = pred[d<r]
            r_e = e[:,d<r]

            y = batch.pid[r_e[0]] == batch.pid[r_e[1]]

            true = len(batch.true_edges)
            edge_true, edge_false = y.float() > 0.5, y.float() < 0.5
            for j, t in enumerate(np.arange(t_min, t_max, t_step)):  
                edge_positive, edge_negative = sig(r_pred) > t, sig(r_pred) < t
                total_true_positive[i][j] += (edge_true & edge_positive).sum().item()
                total_true[i][j] += true
                total_positive[i][j] += edge_positive.sum().item()

    pur, eff = total_true_positive / total_positive, total_true_positive / total_true
    f1 = 2*pur*eff/(pur+eff) 
    return pur, eff, f1

def evaluate_embedding_f1(test_loader, model, r_min, r_max, r_step):
    
    model.eval()
    
    total_true_positive, total_positive, total_true = np.zeros(int(np.ceil((r_max - r_min)/r_step)), dtype=np.float), np.zeros(int(np.ceil((r_max - r_min)/r_step)), dtype=np.float), np.zeros(int(np.ceil((r_max - r_min)/r_step)), dtype=np.float)

    for batch in test_loader:

        data = batch.to(device)
        spatial = model(data.x)

        for i, r in enumerate(np.arange(r_min, r_max, r_step)):    
    #         e = radius_graph(emb_feats, r=r, batch=batch.batch, loop=False, max_num_neighbors=5000)

            e_spatial = build_edges(spatial, r, 100, res)
            e_adjacent = e_spatial[:, ((batch.layers[e_spatial[1]] - batch.layers[e_spatial[0]]) == 1) | ((batch.layers[e_spatial[0]] - batch.layers[e_spatial[1]]) == 1)]

            reference = spatial.index_select(0, e_adjacent[1])
            neighbors = spatial.index_select(0, e_adjacent[0])

            d = torch.sum((reference - neighbors)**2, dim=-1)

            y = (batch.pid[e_spatial[0]] == batch.pid[e_spatial[1]]) & ((batch.layers[e_spatial[1]] - batch.layers[e_spatial[0]] == 1) | (batch.layers[e_spatial[0]] - batch.layers[e_spatial[1]] == 1))

            true = 2*len(batch.true_edges)
            true_positive = (y.float()).sum().item()
            positive = len(e_adjacent[0])

            total_positive[i] += positive
            total_true[i] += true
            total_true_positive[i] += true_positive

    pur, eff = total_true_positive / total_positive, total_true_positive / total_true
    f1 = 2*pur*eff/(pur+eff)
    return pur, eff, f1

def evaluate_embedding_vmeasure(test_loader, model, e_min, e_max, e_step):
    
    model.eval()

    homogeneity, completeness = np.zeros(int(np.ceil((e_max - e_min)/e_step)), dtype=np.float), np.zeros(int(np.ceil((e_max - e_min)/e_step)), dtype=np.float)

    for batch in test_loader:

        data = batch.to(device)
        spatial = model(data.x)

        for i, e in enumerate(np.arange(e_min, e_max, e_step)):    

            embedded = spatial.cpu().detach().numpy()
            db = DBSCAN(eps=e, min_samples=1).fit(embedded)
            labels = db.labels_
            labels_true = batch.pid.cpu().numpy()

            homogeneity[i] += metrics.homogeneity_score(labels_true, labels)
            completeness[i] += metrics.completeness_score(labels_true, labels)
    #         print("Hom:", homogeneity[i], "Comp:", completeness[i])
    homogeneity = homogeneity/len(test_loader.dataset)
    completeness = completeness/len(test_loader.dataset)
    
    return homogeneity, completeness

@torch.no_grad()
def classify_event(model, batch, r, adjacent=False):
    model.eval()
    data = batch.to(device)
    spatial = model(data.x)
        
    e_spatial = build_edges(spatial, r, 100, res)
    
    if adjacent:
        e_spatial = e_spatial[:, ((batch.layers[e_spatial[1]] - batch.layers[e_spatial[0]]) == 1) | ((batch.layers[e_spatial[0]] - batch.layers[e_spatial[1]]) == 1)]
        e_spatial = remove_duplicate_edges(data.x.cpu().numpy(), e_adjacent.cpu().numpy()).astype(int)
    
    data = Data(x = data.x, emb = spatial, pid = data.pid, e = torch.from_numpy(e_spatial))
    return data

@torch.no_grad()
def classify_gnn_event(model, batch, r):
    model.eval()
    data = batch.to(device)
    
    spatial = model(data.x)
        
    e_spatial = build_edges(spatial, r, 100, res)
    e_adjacent = e_spatial[:, ((batch.layers[e_spatial[1]] - batch.layers[e_spatial[0]]) == 1)]
    
    e_adjacent = remove_duplicate_edges(data.x.cpu().numpy(), e_adjacent.cpu().numpy()).astype(int)
    y = batch.pid[e_adjacent[0]] == batch.pid[e_adjacent[1]]
    
    data = Data(x = data.x.cpu(), y = y.float().cpu(), emb = spatial, pid = data.pid.cpu(), e = torch.from_numpy(e_adjacent), layers = data.layers.cpu(), true_edges = data.true_edges.cpu())
    return data

# Model handling

def save_model(epoch, model, optimizer, scheduler, running_loss, config, PATH):
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': running_loss,
                'config': config
                }, os.path.join('/global/cscratch1/sd/danieltm/ExaTrkX/model_comparisons/', PATH))

def save_model_from_script(epoch, model, optimizer, scheduler, running_loss, config, PATH):
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': running_loss,
                'config': config
                }, os.path.join('model_comparisons/', PATH))
    
def save_mixed_model(epoch, model, optimizer, scheduler, running_loss, config, PATH):
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'amp': amp.state_dict(),
                'loss': running_loss,
                'config': config
                }, os.path.join('../model_comparisons/', PATH))
    
def remove_duplicate_edges(X, e):
    
    # Re-introduce layer/directionality information using the r-direction
    r_mask = X[e[0,:],0] > X[e[1,:],0]
    e[0,r_mask], e[1,r_mask] = e[1,r_mask], e[0,r_mask]
    
    # Use cupy sparse matrices to remove duplicates
    e_sparse = sp.sparse.coo_matrix(([1]*e.shape[1], e))
    e_sparse.sum_duplicates()
    
    # There are some self-edges, for some reason. This step may become unnecessary once bug is solved
    e_sparse.setdiag(0)
    e_sparse.eliminate_zeros()
    
    # Reshape as numpy array
    e = np.vstack([e_sparse.row, e_sparse.col])
    
    return e

def graph_intersection(pred_graph, truth_graph):
    array_size = max(pred_graph.max().item(), truth_graph.max().item()) + 1  
    
    l1 = pred_graph.cpu().numpy()
    l2 = truth_graph.cpu().numpy()
    e_1 = sp.sparse.coo_matrix((np.ones(l1.shape[1]), l1), shape=(array_size, array_size)).tocsr()
    e_2 = sp.sparse.coo_matrix((np.ones(l2.shape[1]), l2), shape=(array_size, array_size)).tocsr()
    e_intersection = (e_1.multiply(e_2) - ((e_1 - e_2)>0)).tocoo()

    new_pred_graph = torch.from_numpy(np.vstack([e_intersection.row, e_intersection.col])).long().to(device)
    y = e_intersection.data > 0
    
    return new_pred_graph, y
    
def make_mlp(input_size, sizes,
             hidden_activation=nn.ReLU,
             output_activation=nn.ReLU,
             layer_norm=False):
    """Construct an MLP with specified fully-connected layers."""
    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes
    # Hidden layers
    for i in range(n_layers-1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i+1]))
        layers.append(hidden_activation())
    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1]))
        layers.append(output_activation())
    return nn.Sequential(*layers)

#__________________________ Vanilla Edge Classifaction Network _____________


class EdgeNetwork(nn.Module):
    """
    A module which computes weights for edges of the graph.
    For each edge, it selects the associated nodes' features
    and applies some fully-connected network layers with a final
    sigmoid activation.
    """
    def __init__(self, input_dim, hidden_dim=8, hidden_activation=nn.Tanh,
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
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_activation=nn.Tanh,
                 layer_norm=True):
        super(NodeNetwork, self).__init__()
        self.network = make_mlp(input_dim*3, [hidden_dim, hidden_dim, hidden_dim, output_dim],
                                hidden_activation=hidden_activation,
                                output_activation=hidden_activation,
                                layer_norm=layer_norm)

    def forward(self, x, e, edge_index):
        start, end = edge_index
        # Aggregate edge-weighted incoming/outgoing features
        mi = scatter_add(e[:, None] * x[start], end, dim=0, dim_size=x.shape[0])
        mo = scatter_add(e[:, None] * x[end], start, dim=0, dim_size=x.shape[0])
        node_inputs = torch.cat([mi, mo, x], dim=1)
        return self.network(node_inputs)


class Edge_Class_Net(nn.Module):
    """
    Segment classification graph neural network model.
    Consists of an input network, an edge network, and a node network.
    """
    def __init__(self, input_dim=3, hidden_dim=8, n_graph_iters=3,
                 hidden_activation=nn.Tanh, layer_norm=True):
        super(Edge_Class_Net, self).__init__()
        self.n_graph_iters = n_graph_iters
        # Setup the input network
        self.input_network = make_mlp(input_dim, [hidden_dim],
                                      output_activation=hidden_activation,
                                      layer_norm=layer_norm)
        # Setup the edge network
        self.edge_network = EdgeNetwork(input_dim+hidden_dim, hidden_dim,
                                        hidden_activation, layer_norm=layer_norm)
        # Setup the node layers
        self.node_network = NodeNetwork(input_dim+hidden_dim, hidden_dim, hidden_dim,
                                        hidden_activation, layer_norm=layer_norm)

    def forward(self, inputs):
        """Apply forward pass of the model"""
        # Apply input network to get hidden representation
        x = self.input_network(inputs.x)
        # Shortcut connect the inputs onto the hidden representation
        x = torch.cat([x, inputs.x], dim=-1)
        # Loop over iterations of edge and node networks
        for i in range(self.n_graph_iters):
            # Apply edge network
            e = torch.sigmoid(self.edge_network(x, inputs.edge_index))
            # Apply node network
            x = self.node_network(x, e, inputs.edge_index)
            # Shortcut connect the inputs onto the hidden representation
            x = torch.cat([x, inputs.x], dim=-1)
        # Apply final edge network
        return self.edge_network(x, inputs.edge_index)
    
#_______________ Vanilla Track Counter ________________________
    
class Out_Net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Out_Net, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, out_channels)
        self.lin2 = torch.nn.Linear(in_channels, in_channels)

    def forward(self, x, batch):
        x = tnn.global_mean_pool(x, batch)
        x = F.relu(x)
        x = self.lin1(x.float())
        x = F.relu(x)
        
        return x


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.lin = nn.Sequential(torch.nn.Linear(64, 64),  nn.ReLU(), torch.nn.Linear(64, 64),  nn.ReLU(), torch.nn.Linear(64, 64),  nn.ReLU(), torch.nn.Linear(64, 64),  nn.ReLU())
        self.linout = nn.Sequential(torch.nn.Linear(8, 8),  nn.ReLU(), torch.nn.Linear(8, 8),  nn.ReLU(), torch.nn.Linear(8, 8),  nn.ReLU(), torch.nn.Linear(8, 8),  nn.ReLU())
#         self.conv1 = GCNConv(2, 16)
#         self.conv2 = GCNConv(16, 64)
#         self.conv3 = GCNConv(64, 64)
        self.input = nn.Sequential(torch.nn.Linear(2, 64),nn.ReLU())
        self.conv1 = tnn.HypergraphConv(64, 64, use_attention=True)
        self.conv2 = tnn.HypergraphConv(64, 64, use_attention=True)
#         self.conv2 = tnn.nn.Linear(64, 64)
        self.out = Out_Net(64, 12)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.input(x.float())
        x = self.conv1(x.float(), edge_index)
        x = F.relu(x)
        x = self.lin(x.float())
        x = self.conv2(x.float(), edge_index)
        x = F.relu(x)
        x = self.lin(x.float())
        x = self.conv2(x.float(), edge_index)
        x = F.relu(x)
        x = self.lin(x.float())
        x = self.conv2(x.float(), edge_index)
        x = F.relu(x)
        x = self.lin(x.float())
        x = self.conv2(x.float(), edge_index)
        x = F.relu(x)
        x = self.lin(x.float())
        x = self.conv2(x.float(), edge_index)
        x = F.relu(x)
        x = self.out(x, batch)
#         x = F.relu(x)
#         x = self.linout(x.float())
#         return torch.sigmoid(x)
        return x
    

#__________________________ Combined Edge + Counter Classifaction Network _____________


# class EdgeNetwork(nn.Module):
#     """
#     A module which computes weights for edges of the graph.
#     For each edge, it selects the associated nodes' features
#     and applies some fully-connected network layers with a final
#     sigmoid activation.
#     """
#     def __init__(self, input_dim, hidden_dim=8, hidden_activation=nn.Tanh,
#                  layer_norm=True):
#         super(EdgeNetwork, self).__init__()
#         self.network = make_mlp(input_dim*2,
#                                 [hidden_dim, hidden_dim, hidden_dim, 1],
#                                 hidden_activation=hidden_activation,
#                                 output_activation=None,
#                                 layer_norm=layer_norm)

#     def forward(self, x, edge_index):
#         # Select the features of the associated nodes
#         start, end = edge_index
#         x1, x2 = x[start], x[end]
#         edge_inputs = torch.cat([x[start], x[end]], dim=1)
#         return self.network(edge_inputs).squeeze(-1)

# class NodeNetwork(nn.Module):
#     """
#     A module which computes new node features on the graph.
#     For each node, it aggregates the neighbor node features
#     (separately on the input and output side), and combines
#     them with the node's previous features in a fully-connected
#     network to compute the new features.
#     """
#     def __init__(self, input_dim, output_dim, hidden_activation=nn.Tanh,
#                  layer_norm=True):
#         super(NodeNetwork, self).__init__()
#         self.network = make_mlp(input_dim*3, [output_dim]*4,
#                                 hidden_activation=hidden_activation,
#                                 output_activation=hidden_activation,
#                                 layer_norm=layer_norm)

#     def forward(self, x, e, edge_index):
#         start, end = edge_index
#         # Aggregate edge-weighted incoming/outgoing features
#         mi = scatter_add(e[:, None] * x[start], end, dim=0, dim_size=x.shape[0])
#         mo = scatter_add(e[:, None] * x[end], start, dim=0, dim_size=x.shape[0])
#         node_inputs = torch.cat([mi, mo, x], dim=1)
#         return self.network(node_inputs)

# class Out_Net(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Out_Net, self).__init__()
#         self.lin1 = torch.nn.Linear(in_channels, out_channels)
#         self.lin2 = torch.nn.Linear(in_channels, in_channels)

#     def forward(self, x, batch):
#         x = tnn.global_mean_pool(x, batch)
#         x = self.lin2(x.float())
#         x = F.relu(x)
#         x = self.lin1(x.float())
        
#         return x

class Edge_Graph_Class_Net(nn.Module):
    """
    Segment classification graph neural network model.
    Consists of an input network, an edge network, and a node network.
    """
    def __init__(self, input_dim=3, hidden_dim=8, n_graph_iters=3,
                 output_dim = 8, hidden_activation=nn.Tanh, layer_norm=True):
        super(Edge_Graph_Class_Net, self).__init__()
        self.n_graph_iters = n_graph_iters
        # Setup the input network
        self.input_network = make_mlp(input_dim, [hidden_dim],
                                      output_activation=hidden_activation,
                                      layer_norm=layer_norm)
        # Setup the edge network
        self.edge_network = EdgeNetwork(input_dim+hidden_dim, hidden_dim,
                                        hidden_activation, layer_norm=layer_norm)
        # Setup the node layers
        self.node_network = NodeNetwork(input_dim+hidden_dim, hidden_dim,
                                        hidden_activation, layer_norm=layer_norm)
        self.out_network = Out_Net(input_dim+hidden_dim, output_dim)

    def forward(self, inputs):
        """Apply forward pass of the model"""
        # Apply input network to get hidden representation
        x = self.input_network(inputs.x)
        # Shortcut connect the inputs onto the hidden representation
        x = torch.cat([x, inputs.x], dim=-1)
        # Loop over iterations of edge and node networks
        for i in range(self.n_graph_iters):
            # Apply edge network
            e = torch.sigmoid(self.edge_network(x, inputs.edge_index))
            # Apply node network
            x = self.node_network(x, e, inputs.edge_index)
            # Shortcut connect the inputs onto the hidden representation
            x = torch.cat([x, inputs.x], dim=-1)
        # Apply final edge network
        o = self.out_network(x, inputs.batch)
        return self.edge_network(x, inputs.edge_index), o
    
    
#__________________ Combined Edge & Track Param Classifier ___________


class Edge_Track_Net(nn.Module):
    """
    Segment classification graph neural network model.
    Consists of an input network, an edge network, and a node network.
    """
    def __init__(self, input_dim=3, hidden_dim=8, n_graph_iters=3,
                 output_dim=3, hidden_activation=nn.Tanh, layer_norm=True):
        super(Edge_Track_Net, self).__init__()
        self.n_graph_iters = n_graph_iters
        # Setup the input network
        self.input_network = make_mlp(input_dim, [hidden_dim],
                                      output_activation=hidden_activation,
                                      layer_norm=layer_norm)
        # Setup the edge network
        self.edge_network = EdgeNetwork(input_dim+hidden_dim, hidden_dim,
                                        hidden_activation, layer_norm=layer_norm)
        # Setup the node layers
        self.node_network = NodeNetwork(input_dim+hidden_dim, hidden_dim,
                                        hidden_activation, layer_norm=False)
        
        self.output_network = make_mlp(input_dim+hidden_dim, [hidden_dim, output_dim],
                                      output_activation=hidden_activation,
                                      layer_norm=False)

    def forward(self, inputs):
        """Apply forward pass of the model"""
        # Apply input network to get hidden representation
        x = self.input_network(inputs.x)
        # Shortcut connect the inputs onto the hidden representation
        x = torch.cat([x, inputs.x], dim=-1)
        # Loop over iterations of edge and node networks
        for i in range(self.n_graph_iters):
            # Apply edge network
            e = torch.sigmoid(self.edge_network(x, inputs.edge_index))
            # Apply node network
            x = self.node_network(x, e, inputs.edge_index)
            # Shortcut connect the inputs onto the hidden representation
            x = torch.cat([x, inputs.x], dim=-1)
        # Apply final edge network
        return self.edge_network(x, inputs.edge_index), self.output_network(x)
    


#___________________________________________________________________
    
# class GCNConv(MessagePassing):
#     def __init__(self, in_channels, out_channels):
#         super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
#         self.lin = torch.nn.Linear(in_channels, out_channels)

#     def forward(self, x, edge_index):
#         # x has shape [N, in_channels]
#         # edge_index has shape [2, E]

#         # Step 1: Add self-loops to the adjacency matrix.
#         edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

#         # Step 2: Linearly transform node feature matrix.
#         x = self.lin(x.float())

#         # Step 3-5: Start propagating messages.
#         return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

#     def message(self, x_j, edge_index, size):
#         # x_j has shape [E, out_channels]

#         # Step 3: Normalize node features.
#         row, col = edge_index
#         deg = degree(row, size[0], dtype=x_j.dtype)
#         deg_inv_sqrt = deg.pow(-0.5)
#         norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

#         return norm.view(-1, 1) * x_j

#     def update(self, aggr_out):
#         # aggr_out has shape [N, out_channels]

#         # Step 5: Return new node embeddings.
#         return aggr_out

# class Out_Net(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Out_Net, self).__init__()
#         self.lin = torch.nn.Linear(in_channels, out_channels)

#     def forward(self, x, batch):
#         x = scatter_mean(x, batch, dim=0)
#         x = self.lin(x.float())
        


# class Net(torch.nn.Module):
#     def __init__(self, dataset):
#         super(Net, self).__init__()
#         self.conv1 = GCNConv(2, 16)
#         self.out = Out_Net(16, 2)

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch

#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.out(x, batch)
#         print(x)
# #         return F.log_softmax(x, dim=1)
#         return F.sigmoid(x)
    
