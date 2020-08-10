import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy as sp
import pandas as pd
from torch_cluster import knn_graph, radius_graph
import faiss

sig = torch.nn.Sigmoid()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
res = faiss.StandardGpuResources()

def train_connected_emb(model, train_loader, optimizer, m_configs):
    edge_correct, edge_total_positive, edge_total_true, edge_true_positive, total = 0, 1, 0, 0, 0
    cluster_correct, cluster_total_positive, cluster_total_true, cluster_total_true_positive, cluster_total = 0, 1, 0, 0, 0
    correct = 0
    total = 0
    total_loss = 0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        data = batch.to(device)
        spatial = model(data.x)
        e_bidir = torch.cat([batch.true_edges.to(device), 
                               torch.stack([batch.true_edges[1], batch.true_edges[0]], axis=1).T.to(device)], axis=-1) 
        
        # Get clustered edge list
        e_spatial = build_edges(spatial, m_configs['r_train'], 100, res)       
        array_size = max(e_spatial.max().item(), e_bidir.max().item()) + 1  
            
        l1 = e_spatial.cpu().numpy()
        l2 = e_bidir.cpu().numpy()
        e_1 = sp.sparse.coo_matrix((np.ones(l1.shape[1]), l1), shape=(array_size, array_size)).tocsr()
        e_2 = sp.sparse.coo_matrix((np.ones(l2.shape[1]), l2), shape=(array_size, array_size)).tocsr()
        e_final = (e_1.multiply(e_2) - ((e_1 - e_2)>0)).tocoo()
                
        e_spatial = torch.from_numpy(np.vstack([e_final.row, e_final.col])).long().to(device)
        y_cluster = e_final.data > 0
        
        e_spatial = torch.cat([e_spatial, np.tile(e_bidir, m_configs['weight'])], axis=-1) 
        y_cluster = np.concatenate([y_cluster.astype(int), np.ones(e_bidir.shape[1]*m_configs['weight'])])
        
        hinge = torch.from_numpy(y_cluster).float().to(device)
        hinge[hinge == 0] = -1

        reference = spatial.index_select(0, e_spatial[1])
        neighbors = spatial.index_select(0, e_spatial[0])
        d = torch.sum((reference - neighbors)**2, dim=-1)

        loss = torch.nn.functional.hinge_embedding_loss(d, hinge, margin=m_configs["margin"], reduction=m_configs["reduction"])
        total_loss += loss.item()
#         print(i, loss)
        loss.backward()
        optimizer.step()
#         print("Trained:", i)
    
    return total_loss

def evaluate_connected_emb(model, test_loader, m_configs):
    edge_correct, edge_total_positive, edge_total_true, edge_true_positive, total = 0, 1, 0, 0, 0
    cluster_correct, cluster_total_positive, cluster_total_true, cluster_total_true_positive, cluster_total = 0, 1, 0, 0, 0
    total_loss = 0
    total_av_nhood_size, total_av_adjacent_nhood_size = 0, 0
    for i, batch in enumerate(test_loader):
        data = batch.to(device)
        spatial = model(data.x)
        e_spatial = build_edges(spatial, m_configs['r_val'], 100, res)  
        e_bidir = torch.cat([batch.true_edges.to(device), 
                               torch.stack([batch.true_edges[1], batch.true_edges[0]], axis=1).T.to(device)], axis=-1) 
        array_size = max(e_spatial.max().item(), e_bidir.max().item()) + 1
        
        l1 = e_spatial.cpu().numpy()
        l2 = e_bidir.cpu().numpy()
        e_1 = sp.sparse.coo_matrix((np.ones(l1.shape[1]), l1), shape=(array_size, array_size)).tocsr()
        e_2 = sp.sparse.coo_matrix((np.ones(l2.shape[1]), l2), shape=(array_size, array_size)).tocsr()
        e_final = (e_1.multiply(e_2) - ((e_1 - e_2)>0)).tocoo()
                
        e_spatial = torch.from_numpy(np.vstack([e_final.row, e_final.col])).long().to(device)
        y_cluster = e_final.data > 0
        
        hinge = torch.from_numpy(y_cluster).float().to(device)
        hinge[hinge == 0] = -1

        reference = spatial.index_select(0, e_spatial[1])
        neighbors = spatial.index_select(0, e_spatial[0])
        d = torch.sum((reference - neighbors)**2, dim=-1)

        loss = torch.nn.functional.hinge_embedding_loss(d, hinge, margin=m_configs["margin"], reduction=m_configs["reduction"])
#         print("Loss:", loss.item())
        total_loss += loss.item()
        
        #Cluster performance
        cluster_true = 2*len(batch.true_edges[0])
        
        cluster_true_positive = len(df0.merge(df1))
        cluster_positive = len(e_spatial[0])
        
        cluster_total_true_positive += cluster_true_positive
        cluster_total_positive += max(cluster_positive, 1)
        cluster_total_true += cluster_true
#         total_av_adjacent_nhood_size += len(e_adjacent[0]) / len(spatial)
        
#         print("CLUSTER:", "True positive:", cluster_true_positive, "True:", cluster_true, "Positive:", cluster_positive, "Av nhood size:", len(e_spatial[0])/len(spatial))

    cluster_eff = (cluster_total_true_positive / max(cluster_total_true, 1))
    cluster_pur = (cluster_total_true_positive / max(cluster_total_positive, 1))

#     print('CLUSTER Purity: {:.4f}, Efficiency: {:.4f}'.format(cluster_pur, cluster_eff))
    
    return cluster_pur, cluster_eff, total_loss


def train_embgnn(model, train_loader, optimizer, loss_fn, m_configs, epoch):
    edge_correct, edge_total_positive, edge_total_true, edge_true_positive, total = 0, 1, 0, 0, 0
    cluster_correct, cluster_total_positive, cluster_total_true, cluster_total_true_positive, cluster_total = 0, 1, 0, 0, 0
    total_loss = 0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        data = batch.to(device)
        if epoch < 3:
            pred, spatial, e, _ = model(data, m_configs["r"], small_nhood=True)
        else:
            pred, spatial, e, _ = model(data, m_configs["r"], small_nhood=False)
                
        # Get fake edge list
        candidates = build_edges(spatial, m_configs['r_train'], 100, res)
        fake_list = candidates[:, (batch.pid[candidates[0]] != batch.pid[candidates[1]])
                              & ~((batch.layers[candidates[1]] - batch.layers[candidates[0]] == 1) 
                  | (batch.layers[candidates[0]] - batch.layers[candidates[1]] == 1))]
         
        # Concatenate all candidates
        e_spatial = torch.cat([fake_list, batch.true_edges.T.to(device), 
                               torch.stack([batch.true_edges[:,1], batch.true_edges[:,0]], axis=0).to(device)], axis=-1)        

        reference = spatial.index_select(0, e_spatial[1])
        neighbors = spatial.index_select(0, e_spatial[0])

        d = torch.sum((reference - neighbors)**2, dim=-1)

        y_edge = (batch.pid[e[0]] == batch.pid[e[1]])
        y_cluster = ((batch.pid[e_spatial[0]] == batch.pid[e_spatial[1]]) 
                  & ((batch.layers[e_spatial[1]] - batch.layers[e_spatial[0]] == 1) 
                  | (batch.layers[e_spatial[0]] - batch.layers[e_spatial[1]] == 1)))
        
        hinge = y_cluster.float()
        hinge[hinge == 0] = -1 

        loss_1 = F.binary_cross_entropy_with_logits(pred.float(), y_edge.float(), pos_weight=torch.tensor(m_configs["weight"]))
        loss_2 = torch.nn.functional.hinge_embedding_loss(d, hinge, margin=m_configs["margin"], reduction=m_configs["reduction"])
#         print("Loss 1:", loss_1.item(), "Loss 2:", loss_2.item())
        loss = loss_fn([loss_1.to(device), loss_2.to(device)])
#         print("Loss:", loss, "Noise params:", loss_fn.noise_params)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    return total_loss

def evaluate_embgnn(model, test_loader, loss_fn, m_configs):
    edge_correct, edge_total_positive, edge_total_true, edge_true_positive, total = 0, 1, 0, 0, 0
    cluster_correct, cluster_total_positive, cluster_total_true, cluster_total_true_positive, cluster_total = 0, 1, 0, 0, 0
    total_loss, total_loss_1, total_loss_2 = 0, 0, 0
    total_av_nhood_size, total_av_adjacent_nhood_size = 0, 0
#     print('Beginning evaluation')
    for i, batch in enumerate(test_loader):
        data = batch.to(device)
        pred, spatial, e, _ = model(data,  m_configs["r"], small_nhood=False)
        e_spatial = build_edges(spatial, m_configs['r_val'], 100, res)
        e_adjacent = e_spatial[:, (batch.layers[e_spatial[1]] - batch.layers[e_spatial[0]] == 1)
                  | (batch.layers[e_spatial[0]] - batch.layers[e_spatial[1]] == 1)]
        
        reference = spatial.index_select(0, e_spatial[1])
        neighbors = spatial.index_select(0, e_spatial[0])
        d = torch.sum((reference - neighbors)**2, dim=-1)
        y_edge = (batch.pid[e[0]] == batch.pid[e[1]])
        y_cluster_all = ((batch.pid[e_spatial[0]] == batch.pid[e_spatial[1]]) 
                  & ((batch.layers[e_spatial[1]] - batch.layers[e_spatial[0]] == 1)
                  | (batch.layers[e_spatial[0]] - batch.layers[e_spatial[1]] == 1)))
        y_cluster = (batch.pid[e_adjacent[0]] == batch.pid[e_adjacent[1]])
        hinge = y_cluster_all.float()
        hinge[hinge == 0] = -1

        loss_1 = F.binary_cross_entropy_with_logits(pred.float(), y_edge.float(), pos_weight=torch.tensor(m_configs["weight"]))
        loss_2 = torch.nn.functional.hinge_embedding_loss(d, hinge, margin=m_configs["margin"], reduction=m_configs["reduction"])
#         print("Loss 1:", loss_1.item(), "Loss 2:", loss_2.item())
        loss = loss_fn([loss_1.to(device), loss_2.to(device)])
#         print('Losses combined')
#         print("Combined loss:", loss, "Noise params:", loss_fn.noise_params)
        total_loss += loss.item()
        total_loss_1 += loss_1.item()
        total_loss_2 += loss_2.item()
        
        #Cluster performance
        cluster_true = 2*len(batch.true_edges)

        cluster_true_positive = (y_cluster.float()).sum().item()
        cluster_total_true_positive += cluster_true_positive

        cluster_positive = len(e_adjacent[0])
        cluster_total_positive += max(cluster_positive, 1)

        cluster_total_true += cluster_true
        
        #Edge performance
        edge_true, edge_false = y_edge.float() > 0.5, y_edge.float() < 0.5
        edge_positive, edge_negative = sig(pred) > 0.5, sig(pred) < 0.5

        edge_correct += ((sig(pred) > 0.5) == (y_edge.float() > 0.5)).sum().item()

        edge_true_positive += (edge_true & edge_positive).sum().item()
        edge_total_true += edge_true.sum().item()
        edge_total_positive += edge_positive.sum().item()
        
        total_av_adjacent_nhood_size += len(e_adjacent[0]) / len(spatial)
        total += len(pred)

    edge_acc = edge_correct/ max(total, 1)
    edge_eff = (edge_true_positive / max(edge_total_true, 1))
    edge_pur = (edge_true_positive / max(edge_total_positive, 1))

    cluster_eff = (cluster_total_true_positive / max(cluster_total_true, 1))
    cluster_pur = (cluster_total_true_positive / max(cluster_total_positive, 1))
    
    return edge_acc, edge_pur, edge_eff, cluster_pur, cluster_eff, total_loss, total_loss_1, total_loss_2

def balanced_adjacent_train(model, train_loader, optimizer, loss_fn, m_configs):
    edge_correct, edge_total_positive, edge_total_true, edge_true_positive, total = 0, 1, 0, 0, 0
    cluster_correct, cluster_total_positive, cluster_total_true, cluster_total_true_positive, cluster_total = 0, 1, 0, 0, 0
    total_loss = 0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        data = batch.to(device)
        pred, spatial, e, _ = model(data)
        
        # Get fake edge list
        candidates = build_edges(spatial, m_configs['r_train'], 100, res)
        fake_list = candidates[:,batch.pid[candidates[0]] != batch.pid[candidates[1]]]
        
#         print(batch.pid[fake_list[0]] == batch.pid[fake_list[1]])
        
        # Concatenate all candidates
        e_spatial = torch.cat([fake_list, batch.true_edges.T.to(device)], axis=-1)

        reference = spatial.index_select(0, e_spatial[1])
        neighbors = spatial.index_select(0, e_spatial[0])

        d = torch.sum((reference - neighbors)**2, dim=-1)

        y_edge = ((batch.pid[e[0]] == batch.pid[e[1]]) & (batch.layers[e[1]] - batch.layers[e[0]] == 1))
        y_cluster = ((batch.pid[e_spatial[0]] == batch.pid[e_spatial[1]]) & (batch.layers[e_spatial[1]] - batch.layers[e_spatial[0]] == 1))
        
        hinge = y_cluster.float()
        hinge[hinge == 0] = -1 

        loss_1 = F.binary_cross_entropy_with_logits(pred.float(), y_edge.float(), pos_weight=torch.tensor(m_configs["weight"]))
        loss_2 = torch.nn.functional.hinge_embedding_loss(d, hinge, margin=m_configs["margin"], reduction=m_configs["reduction"])
#         print("Loss 1:", loss_1.item(), "Loss 2:", loss_2.item())
        loss = loss_fn([loss_1.to(device), loss_2.to(device)])
#         print("Loss:", loss, "Noise params:", loss_fn.noise_params)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        #Cluster performance
        batch_cpu = batch.pid.cpu()
        pids, counts = np.unique(batch_cpu, return_counts=True)
        
        cluster_true_positive = (y_cluster.float()).sum().item()
        cluster_total_true_positive += cluster_true_positive 
        
        cluster_positive = len(e_spatial[0])
        cluster_total_positive += max(cluster_positive, 1)

        edge_correct += ((sig(pred) > 0.5) == (y_edge.float() > 0.5)).sum().item()
        total += len(pred)

    edge_acc = edge_correct/max(total, 1)
    cluster_pur = (cluster_total_true_positive / cluster_total_positive)
    
    return edge_acc, cluster_pur, total_loss

def evaluate_adjacent(model, test_loader, loss_fn, m_configs):
    edge_correct, edge_total_positive, edge_total_true, edge_true_positive, total = 0, 1, 0, 0, 0
    cluster_correct, cluster_total_positive, cluster_total_true, cluster_total_true_positive, cluster_total = 0, 1, 0, 0, 0
    total_loss = 0
    total_av_nhood_size, total_av_adjacent_nhood_size = 0, 0
#     print('Beginning evaluation')
    for i, batch in enumerate(test_loader):
        data = batch.to(device)
        pred, spatial, e, _ = model(data)
        e_spatial = build_edges(spatial, m_configs['r_val'], 100, res)
        e_adjacent = e_spatial[:, (batch.layers[e_spatial[1]] - batch.layers[e_spatial[0]]) == 1]
        
        reference = spatial.index_select(0, e_spatial[1])
        neighbors = spatial.index_select(0, e_spatial[0])
        d = torch.sum((reference - neighbors)**2, dim=-1)
        y_edge = ((batch.pid[e[0]] == batch.pid[e[1]]) & (batch.layers[e[1]] - batch.layers[e[0]] == 1))
        y_cluster_all = ((batch.pid[e_spatial[0]] == batch.pid[e_spatial[1]]) & (batch.layers[e_spatial[1]] - batch.layers[e_spatial[0]] == 1))
        y_cluster = (batch.pid[e_adjacent[0]] == batch.pid[e_adjacent[1]])
        hinge = y_cluster_all.float()
        hinge[hinge == 0] = -1

        loss_1 = F.binary_cross_entropy_with_logits(pred.float(), y_edge.float(), pos_weight=torch.tensor(m_configs["weight"]))
        loss_2 = torch.nn.functional.hinge_embedding_loss(d, hinge, margin=m_configs["margin"], reduction=m_configs["reduction"])
#         print("Loss 1:", loss_1.item(), "Loss 2:", loss_2.item())
        loss = loss_fn([loss_1.to(device), loss_2.to(device)])
#         print('Losses combined')
#         print("Combined loss:", loss, "Noise params:", loss_fn.noise_params)
        total_loss += loss.item()
        
        #Cluster performance
        cluster_true = len(batch.true_edges)

        cluster_true_positive = (y_cluster.float()).sum().item()
        cluster_total_true_positive += cluster_true_positive

        cluster_positive = len(e_adjacent[0])
        cluster_total_positive += max(cluster_positive, 1)

        cluster_total_true += cluster_true
#         print('Cluster performance calculated')
        #Edge performance
        edge_true, edge_false = y_edge.float() > 0.5, y_edge.float() < 0.5
        edge_positive, edge_negative = sig(pred) > 0.5, sig(pred) < 0.5

        edge_correct += ((sig(pred) > 0.5) == (y_edge.float() > 0.5)).sum().item()

        edge_true_positive += (edge_true & edge_positive).sum().item()
        edge_total_true += edge_true.sum().item()
        edge_total_positive += edge_positive.sum().item()

#         print("EDGES:", "True positive:", (edge_true & edge_positive).sum().item(), "True:", edge_true.sum().item(), "Positive", edge_positive.sum().item())
#         print("CLUSTER:", "True positive:", cluster_true_positive, "True:", cluster_true, "Positive:", cluster_positive)
#         print('Edge performance calculated')
        total_av_adjacent_nhood_size += len(e_adjacent[0]) / len(spatial)
        total += len(pred)

    edge_acc = edge_correct/ max(total, 1)
    edge_eff = (edge_true_positive / max(edge_total_true, 1))
    edge_pur = (edge_true_positive / max(edge_total_positive, 1))

    cluster_eff = (cluster_total_true_positive / max(cluster_total_true, 1))
    cluster_pur = (cluster_total_true_positive / max(cluster_total_positive, 1))

#     print('EDGE Accuracy: {:.4f}, Purity: {:.4f}, Efficiency: {:.4f}'.format(edge_acc, edge_pur, edge_eff))
#     print('CLUSTER Purity: {:.4f}, Efficiency: {:.4f}'.format(cluster_pur, cluster_eff))
    
    return edge_acc, edge_pur, edge_eff, cluster_pur, cluster_eff, total_loss, total_av_adjacent_nhood_size/(i+1)

def train_bi_emb(model, train_loader, optimizer, m_configs):
    edge_correct, edge_total_positive, edge_total_true, edge_true_positive, total = 0, 1, 0, 0, 0
    cluster_correct, cluster_total_positive, cluster_total_true, cluster_total_true_positive, cluster_total = 0, 1, 0, 0, 0
    correct = 0
    total = 0
    total_loss = 0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        data = batch.to(device)
        spatial = model(data.x)
        
        # Get fake edge list
#         candidates = radius_graph(spatial, r=m_configs['r_train'], batch=batch.batch, loop=False, max_num_neighbors=200)
        candidates = build_edges(spatial, m_configs['r_train'], 100, res)
        fake_list = candidates[:,batch.pid[candidates[0]] != batch.pid[candidates[1]]]
                
        # Concatenate all candidates
        e_spatial = torch.cat([fake_list, batch.true_edges.T.to(device), 
                               torch.stack([batch.true_edges[:,1], batch.true_edges[:,0]], axis=0).to(device)], axis=-1)        
#         e_spatial = torch.cat([fake_list, batch.true_edges.T.to(device)], axis=-1)        
        reference = spatial.index_select(0, e_spatial[1])
        neighbors = spatial.index_select(0, e_spatial[0])

        d = torch.sum((reference - neighbors)**2, dim=-1)

        y_cluster = (batch.pid[e_spatial[0]] == batch.pid[e_spatial[1]]) & ((batch.layers[e_spatial[1]] - batch.layers[e_spatial[0]] == 1) | (batch.layers[e_spatial[0]] - batch.layers[e_spatial[1]] == 1))
        
        hinge = y_cluster.float()
        hinge[hinge == 0] = -1

        loss = torch.nn.functional.hinge_embedding_loss(d, hinge, margin=m_configs["margin"], reduction=m_configs["reduction"])
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        cluster_true_positive = (y_cluster.float()).sum().item()
        cluster_total_true_positive += cluster_true_positive 
        
        cluster_positive = len(e_spatial[0])
        cluster_total_positive += max(cluster_positive, 1)
        
    cluster_pur = (cluster_total_true_positive / cluster_total_positive)
    
    return cluster_pur, total_loss

def evaluate_bi_emb(model, test_loader, m_configs):
    edge_correct, edge_total_positive, edge_total_true, edge_true_positive, total = 0, 1, 0, 0, 0
    cluster_correct, cluster_total_positive, cluster_total_true, cluster_total_true_positive, cluster_total = 0, 1, 0, 0, 0
    total_loss = 0
    total_av_nhood_size, total_av_adjacent_nhood_size = 0, 0
    for i, batch in enumerate(test_loader):
        data = batch.to(device)
        spatial = model(data.x)
        e_spatial = build_edges(spatial, m_configs['r_val'], 100, res)
        e_adjacent = e_spatial[:, ((batch.layers[e_spatial[1]] - batch.layers[e_spatial[0]]) == 1) | ((batch.layers[e_spatial[0]] - batch.layers[e_spatial[1]]) == 1)]
        
        reference = spatial.index_select(0, e_spatial[1])
        neighbors = spatial.index_select(0, e_spatial[0])
        d = torch.sum((reference - neighbors)**2, dim=-1)
        y_cluster_all = (batch.pid[e_spatial[0]] == batch.pid[e_spatial[1]]) & ((batch.layers[e_spatial[1]] - batch.layers[e_spatial[0]] == 1) | (batch.layers[e_spatial[0]] - batch.layers[e_spatial[1]] == 1))
        y_cluster = (batch.pid[e_adjacent[0]] == batch.pid[e_adjacent[1]])
        hinge = y_cluster_all.float()
        hinge[hinge == 0] = -1

        loss = torch.nn.functional.hinge_embedding_loss(d, hinge, margin=m_configs["margin"], reduction=m_configs["reduction"])
#         print("Loss:", loss.item())
        total_loss += loss.item()
        #Cluster performance
        cluster_true = 2*len(batch.true_edges)

        cluster_true_positive = (y_cluster.float()).sum().item()
        cluster_total_true_positive += cluster_true_positive
        cluster_positive = len(e_adjacent[0])
        cluster_total_positive += max(cluster_positive, 1)

        cluster_total_true += cluster_true
        total_av_nhood_size += len(e_spatial[0]) / len(spatial)
        total_av_adjacent_nhood_size += len(e_adjacent[0]) / len(spatial)
        
#         print("CLUSTER:", "True positive:", cluster_true_positive, "True:", cluster_true, "Positive:", cluster_positive, "Av nhood size:", len(e_spatial[0])/len(spatial))

    cluster_eff = (cluster_total_true_positive / max(cluster_total_true, 1))
    cluster_pur = (cluster_total_true_positive / max(cluster_total_positive, 1))

#     print('CLUSTER Purity: {:.4f}, Efficiency: {:.4f}'.format(cluster_pur, cluster_eff))
    
    return cluster_pur, cluster_eff, total_loss, total_av_adjacent_nhood_size/(i+1)

def train_emb(model, train_loader, optimizer, m_configs):
    edge_correct, edge_total_positive, edge_total_true, edge_true_positive, total = 0, 1, 0, 0, 0
    cluster_correct, cluster_total_positive, cluster_total_true, cluster_total_true_positive, cluster_total = 0, 1, 0, 0, 0
    correct = 0
    total = 0
    total_loss = 0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        data = batch.to(device)
        spatial = model(data.x)
        
        # Get fake edge list
#         candidates = radius_graph(spatial, r=m_configs['r_train'], batch=batch.batch, loop=False, max_num_neighbors=200)
        candidates = build_edges(spatial, m_configs['r_train'], 100, res)

        fake_list = candidates[:,batch.pid[candidates[0]] != batch.pid[candidates[1]]]
                
        # Concatenate all candidates
        e_spatial = torch.cat([fake_list, batch.true_edges.T.to(device)], axis=-1)        
        reference = spatial.index_select(0, e_spatial[1])
        neighbors = spatial.index_select(0, e_spatial[0])

        d = torch.sum((reference - neighbors)**2, dim=-1)

        y_cluster = ((batch.pid[e_spatial[0]] == batch.pid[e_spatial[1]]) & (batch.layers[e_spatial[1]] - batch.layers[e_spatial[0]] == 1))
        
        hinge = y_cluster.float()
        hinge[hinge == 0] = -1

        loss = torch.nn.functional.hinge_embedding_loss(d, hinge, margin=m_configs["margin"], reduction=m_configs["reduction"])
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        #Cluster performance
        batch_cpu = batch.pid.cpu()
        pids, counts = np.unique(batch_cpu, return_counts=True)
        
        cluster_true_positive = (y_cluster.float()).sum().item()
        cluster_total_true_positive += cluster_true_positive 
        
        cluster_positive = len(e_spatial[0])
        cluster_total_positive += max(cluster_positive, 1)
        
    cluster_pur = (cluster_total_true_positive / cluster_total_positive)
    
    return cluster_pur, total_loss

def evaluate_emb(model, test_loader, m_configs):
    edge_correct, edge_total_positive, edge_total_true, edge_true_positive, total = 0, 1, 0, 0, 0
    cluster_correct, cluster_total_positive, cluster_total_true, cluster_total_true_positive, cluster_total = 0, 1, 0, 0, 0
    total_loss = 0
    total_av_nhood_size, total_av_adjacent_nhood_size = 0, 0
    for i, batch in enumerate(test_loader):
        data = batch.to(device)
        spatial = model(data.x)
        e_spatial = build_edges(spatial, m_configs['r_val'], 100, res)
        e_adjacent = e_spatial[:, (batch.layers[e_spatial[1]] - batch.layers[e_spatial[0]]) == 1]
        
        reference = spatial.index_select(0, e_spatial[1])
        neighbors = spatial.index_select(0, e_spatial[0])
        d = torch.sum((reference - neighbors)**2, dim=-1)
        y_cluster_all = ((batch.pid[e_spatial[0]] == batch.pid[e_spatial[1]]) & (batch.layers[e_spatial[1]] - batch.layers[e_spatial[0]] == 1))
        y_cluster = (batch.pid[e_adjacent[0]] == batch.pid[e_adjacent[1]])
        hinge = y_cluster_all.float()
        hinge[hinge == 0] = -1

        loss = torch.nn.functional.hinge_embedding_loss(d, hinge, margin=m_configs["margin"], reduction=m_configs["reduction"])
#         print("Loss:", loss.item())
        total_loss += loss.item()
        #Cluster performance
        cluster_true = len(batch.true_edges)

        cluster_true_positive = (y_cluster.float()).sum().item()
        cluster_total_true_positive += cluster_true_positive
        cluster_positive = len(e_adjacent[0])
        cluster_total_positive += max(cluster_positive, 1)

        cluster_total_true += cluster_true
        total_av_nhood_size += len(e_spatial[0]) / len(spatial)
        total_av_adjacent_nhood_size += len(e_adjacent[0]) / len(spatial)
        
#         print("CLUSTER:", "True positive:", cluster_true_positive, "True:", cluster_true, "Positive:", cluster_positive, "Av nhood size:", len(e_spatial[0])/len(spatial))

    cluster_eff = (cluster_total_true_positive / max(cluster_total_true, 1))
    cluster_pur = (cluster_total_true_positive / max(cluster_total_positive, 1))

#     print('CLUSTER Purity: {:.4f}, Efficiency: {:.4f}'.format(cluster_pur, cluster_eff))
    
    return cluster_pur, cluster_eff, total_loss, total_av_adjacent_nhood_size/(i+1)

def train_gnn(model, gnn_train_loader, optimizer, m_configs):
    correct = 0
    total = 0
    total_loss = 0
    for i, batch in enumerate(gnn_train_loader):
        optimizer.zero_grad()
        data = batch.to(device)
        pred = model(data)
              
        y = batch.pid[batch.e[0]] == batch.pid[batch.e[1]]
                
        loss = F.binary_cross_entropy_with_logits(pred.float(), y.float(), pos_weight=torch.tensor(m_configs["weight"]))
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        true, false = y.float() > 0.5, y.float() < 0.5
        positive, negative = sig(pred) > 0.5, sig(pred) < 0.5
        
        correct += ((sig(pred) > 0.5) == (y.float() > 0.5)).sum().item()
        total += len(pred)
    acc = correct/total
    return acc, total_loss

def evaluate_gnn(model, gnn_test_loader, m_configs):
    correct, total_positive, total_true, true_positive, total, total_loss = 0, 1, 0, 0, 0, 0
    
    for batch in gnn_test_loader:
        data = batch.to(device)
        pred = model(data)
              
        y = batch.pid[batch.e[0]] == batch.pid[batch.e[1]]
                
        loss = F.binary_cross_entropy_with_logits(pred.float(), y.float(), pos_weight=torch.tensor(m_configs["weight"]))
        total_loss += loss.item()
        
        true, false = y.float() > 0.5, y.float() < 0.5
        positive, negative = sig(pred) > 0.5, sig(pred) < 0.5
        true_positive += (true & positive).sum().item()
        total_positive += max(positive.sum().item(), 1)
        
        correct += ((sig(pred) > 0.5) == (y.float() > 0.5)).sum().item()
        total_true += max(true.sum().item(), 1)
        
#         print("True positive:", (true & positive).sum().item(), "True:", true.sum().item(), "Positive", positive.sum().item())
        
        total += len(pred)

    acc = correct/total
    eff = (true_positive / total_true)
    pur = (true_positive / total_positive)
        
    return acc, eff, pur, total_loss

def balanced_train(model, train_loader, optimizer, loss_fn, m_configs):
    edge_correct, edge_total_positive, edge_total_true, edge_true_positive, total = 0, 1, 0, 0, 0
    cluster_correct, cluster_total_positive, cluster_total_true, cluster_total_true_positive, cluster_total = 0, 1, 0, 0, 0
    correct = 0
    total = 0
    total_loss = 0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        data = batch.to(device)
        pred, spatial, e = model(data)
        
        # Get fake edge list
        candidates = radius_graph(spatial, r=m_configs['r_train'], batch=batch.batch, loop=False, max_num_neighbors=200)
        fake_list = candidates[:,batch.pid[candidates[0]] != batch.pid[candidates[1]]]
        
#         print(batch.pid[fake_list[0]] == batch.pid[fake_list[1]])
        
        # Concatenate all candidates
        e_spatial = torch.cat([fake_list, batch.true_edges.T.to(device)], axis=-1)

        reference = spatial.index_select(0, e_spatial[1])
        neighbors = spatial.index_select(0, e_spatial[0])

        d = torch.sum((reference - neighbors)**2, dim=-1)

        y_edge = batch.pid[e[0]] == batch.pid[e[1]]    
        y_cluster = batch.pid[e_spatial[0]] == batch.pid[e_spatial[1]]    

        hinge = y_cluster.float()
        hinge[batch.pid[e_spatial[0]] != batch.pid[e_spatial[1]]] = -1  

        loss_1 = F.binary_cross_entropy_with_logits(pred.float(), y_edge.float(), pos_weight=torch.tensor(m_configs["weight"]))
        loss_2 = torch.nn.functional.hinge_embedding_loss(d, hinge, margin=m_configs["margin"], reduction=m_configs["reduction"])
#         print("Loss 1:", loss_1.item(), "Loss 2:", loss_2.item())
        loss = loss_fn([loss_1.to(device), loss_2.to(device)])
#         print("Loss:", loss, "Noise params:", loss_fn.noise_params)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        #Cluster performance
        batch_cpu = batch.pid.cpu()
        pids, counts = np.unique(batch_cpu, return_counts=True)
        
        cluster_true_positive = (y_cluster.float()).sum().item()
        cluster_total_true_positive += cluster_true_positive 
        
        cluster_positive = len(e_spatial[0])
        cluster_total_positive += max(cluster_positive, 1)

        edge_correct += ((sig(pred) > 0.5) == (y_edge.float() > 0.5)).sum().item()
        total += len(pred)

    edge_acc = edge_correct/total
    cluster_pur = (cluster_total_true_positive / cluster_total_positive)
    
    return edge_acc, cluster_pur, total_loss

def evaluate(model, test_loader, loss_fn, m_configs):
    edge_correct, edge_total_positive, edge_total_true, edge_true_positive, total = 0, 1, 0, 0, 0
    cluster_correct, cluster_total_positive, cluster_total_true, cluster_total_true_positive, cluster_total = 0, 1, 0, 0, 0
    total_loss = 0
    for batch in test_loader:
        data = batch.to(device)
        pred, spatial, e = model(data)
        
        e_spatial = radius_graph(spatial, r=m_configs["r_val"], batch=batch.batch, loop=False, max_num_neighbors=200)
#         e_spatial = knn_graph(spatial, k=m_configs["k"], batch=batch.batch, loop=False)

        reference = spatial.index_select(0, e_spatial[1])
        neighbors = spatial.index_select(0, e_spatial[0])

        d = torch.sum((reference - neighbors)**2, dim=-1)

        y_edge = ((batch.pid[e[0]] == batch.pid[e[1]]) & (batch.layers[e[1]] - batch.layers[e[0]] == 1))
        y_cluster = ((batch.pid[e_spatial[0]] == batch.pid[e_spatial[1]]) & (batch.layers[e_spatial[1]] - batch.layers[e_spatial[0]] == 1))
        
        hinge = y_cluster.float()
        hinge[hinge == 0] = -1

        loss_1 = F.binary_cross_entropy_with_logits(pred.float(), y_edge.float(), pos_weight=torch.tensor(m_configs["weight"]))
        loss_2 = torch.nn.functional.hinge_embedding_loss(d, hinge, margin=m_configs["margin"], reduction=m_configs["reduction"])
#         print("Loss 1:", loss_1.item(), "Loss 2:", loss_2.item())

        loss = loss_fn([loss_1, loss_2]).item()
#         print("Combined loss:", loss, "Noise params:", loss_fn.noise_params)
        total_loss += loss
        
        #Cluster performance
        cluster_true = len(batch.true_edges)

        cluster_true_positive = (y_cluster.float()).sum().item()
        cluster_total_true_positive += cluster_true_positive

        cluster_positive = len(e_spatial[0])
        cluster_total_positive += max(cluster_positive, 1)

        cluster_total_true += cluster_true

        #Edge performance
        edge_true, edge_false = y_edge.float() > 0.5, y_edge.float() < 0.5
        edge_positive, edge_negative = sig(pred) > 0.5, sig(pred) < 0.5

        edge_correct += ((sig(pred) > 0.5) == (y_edge.float() > 0.5)).sum().item()

        edge_true_positive += (edge_true & edge_positive).sum().item()
        edge_total_true += edge_true.sum().item()
        edge_total_positive += edge_positive.sum().item()

#         print("EDGES:", "True positive:", (edge_true & edge_positive).sum().item(), "True:", edge_true.sum().item(), "Positive", edge_positive.sum().item())
#         print("CLUSTER:", "True positive:", cluster_true_positive, "True:", cluster_true, "Positive:", cluster_positive)

        total += len(pred)

    edge_acc = edge_correct/total
    edge_eff = (edge_true_positive / max(edge_total_true, 1))
    edge_pur = (edge_true_positive / max(edge_total_positive, 1))

    cluster_eff = (cluster_total_true_positive / max(cluster_total_true, 1))
    cluster_pur = (cluster_total_true_positive / max(cluster_total_positive, 1))

#     print('EDGE Accuracy: {:.4f}, Purity: {:.4f}, Efficiency: {:.4f}'.format(edge_acc, edge_pur, edge_eff))
#     print('CLUSTER Purity: {:.4f}, Efficiency: {:.4f}'.format(cluster_pur, cluster_eff))
    
    return edge_acc, edge_pur, edge_eff, cluster_pur, cluster_eff, total_loss

def train_all_emb(model, train_loader, optimizer, m_configs):
    edge_correct, edge_total_positive, edge_total_true, edge_true_positive, total = 0, 1, 0, 0, 0
    cluster_correct, cluster_total_positive, cluster_total_true, cluster_total_true_positive, cluster_total = 0, 1, 0, 0, 0
    correct = 0
    total = 0
    total_loss = 0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        data = batch.to(device)
        spatial = model(data.x)
        
        records_array = batch.pid.cpu()
        idx_sort = np.argsort(records_array)
        sorted_records_array = records_array[idx_sort]
        _, idx_start, _ = np.unique(sorted_records_array, return_counts=True,
                                return_index=True)
        # sets of indices
        indices = np.split(idx_sort, idx_start[1:])
        true_edges = np.concatenate([list(permutations(i, r=2)) for i in indices if len(list(permutations(i, r=2))) > 0])
        
        # Get fake edge list
#         candidates = radius_graph(spatial, r=m_configs['r_train'], batch=batch.batch, loop=False, max_num_neighbors=200)
        candidates = build_edges(spatial, m_configs['r_train'], 100, res)

        fake_list = candidates[:,batch.pid[candidates[0]] != batch.pid[candidates[1]]]
                
        # Concatenate all candidates
        e_spatial = torch.cat([fake_list, torch.from_numpy(true_edges.T).to(device)], axis=-1)        
#         e_spatial = torch.cat([candidates, batch.true_edges.T.to(device), torch.stack([batch.true_edges[:,1], batch.true_edges[:,0]], axis=0).to(device)], axis=-1)        
#         e_spatial = torch.cat([fake_list, batch.true_edges.T.to(device)], axis=-1)        
        reference = spatial.index_select(0, e_spatial[1])
        neighbors = spatial.index_select(0, e_spatial[0])

        d = torch.sum((reference - neighbors)**2, dim=-1)

        y_cluster = (batch.pid[e_spatial[0]] == batch.pid[e_spatial[1]])
        
        hinge = y_cluster.float()
        hinge[hinge == 0] = -1

        loss = torch.nn.functional.hinge_embedding_loss(d, hinge, margin=m_configs["margin"], reduction=m_configs["reduction"])
        total_loss += loss.item()
        print(loss)
        loss.backward()
        optimizer.step()
                
        cluster_true_positive = (y_cluster.float()).sum().item()
        cluster_total_true_positive += cluster_true_positive 
        
        cluster_positive = len(e_spatial[0])
        cluster_total_positive += max(cluster_positive, 1)
        
    cluster_pur = (cluster_total_true_positive / cluster_total_positive)
    
    return cluster_pur, total_loss

def evaluate_all_emb(model, test_loader, m_configs):
    edge_correct, edge_total_positive, edge_total_true, edge_true_positive, total = 0, 1, 0, 0, 0
    cluster_correct, cluster_total_positive, cluster_total_true, cluster_total_true_positive, cluster_total = 0, 1, 0, 0, 0
    total_loss = 0
    total_av_nhood_size, total_av_adjacent_nhood_size = 0, 0
    for i, batch in enumerate(test_loader):
        data = batch.to(device)
        spatial = model(data.x)
        e_spatial = build_edges(spatial, m_configs['r_val'], 100, res)  
        e_adjacent = e_spatial[:, ((batch.layers[e_spatial[1]] - batch.layers[e_spatial[0]]) == 1) | ((batch.layers[e_spatial[0]] - batch.layers[e_spatial[1]]) == 1)]
        
        reference = spatial.index_select(0, e_spatial[1])
        neighbors = spatial.index_select(0, e_spatial[0])
        d = torch.sum((reference - neighbors)**2, dim=-1)
        y_cluster_all = (batch.pid[e_spatial[0]] == batch.pid[e_spatial[1]])
        y_cluster = (batch.pid[e_adjacent[0]] == batch.pid[e_adjacent[1]])
        hinge = y_cluster_all.float()
        hinge[hinge == 0] = -1

        loss = torch.nn.functional.hinge_embedding_loss(d, hinge, margin=m_configs["margin"], reduction=m_configs["reduction"])
#         print("Loss:", loss.item())
        total_loss += loss.item()
        #Cluster performance
        batch_cpu = batch.pid.cpu()
        pids, counts = np.unique(batch_cpu, return_counts=True)
        
        cluster_true = 2*len(batch.true_edges)
        cluster_true_positive = (y_cluster.float()).sum().item()
        cluster_positive = len(e_adjacent[0])
        
        cluster_total_true_positive += cluster_true_positive
        cluster_total_positive += max(cluster_positive, 1)
        cluster_total_true += cluster_true
        total_av_nhood_size += len(e_spatial[0]) / len(spatial)
#         total_av_adjacent_nhood_size += len(e_adjacent[0]) / len(spatial)
        
#         print("CLUSTER:", "True positive:", cluster_true_positive, "True:", cluster_true, "Positive:", cluster_positive, "Av nhood size:", len(e_spatial[0])/len(spatial))

    cluster_eff = (cluster_total_true_positive / max(cluster_total_true, 1))
    cluster_pur = (cluster_total_true_positive / max(cluster_total_positive, 1))

#     print('CLUSTER Purity: {:.4f}, Efficiency: {:.4f}'.format(cluster_pur, cluster_eff))
    
    return cluster_pur, cluster_eff, total_loss, total_av_nhood_size/(i+1)

def swig_ptr_from_FloatTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.float32
    return faiss.cast_integer_to_float_ptr(
        x.storage().data_ptr() + x.storage_offset() * 4)

def swig_ptr_from_LongTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.int64, 'dtype=%s' % x.dtype
    return faiss.cast_integer_to_long_ptr(
        x.storage().data_ptr() + x.storage_offset() * 8)

def search_index_pytorch(index, x, k, D=None, I=None):
    """call the search function of an index with pytorch tensor I/O (CPU
    and GPU supported)"""
    assert x.is_contiguous()
    n, d = x.size()
    assert d == index.d

    if D is None:
        D = torch.empty((n, k), dtype=torch.float32, device=x.device)
    else:
        assert D.size() == (n, k)

    if I is None:
        I = torch.empty((n, k), dtype=torch.int64, device=x.device)
    else:
        assert I.size() == (n, k)
    torch.cuda.synchronize()
    xptr = swig_ptr_from_FloatTensor(x)
    Iptr = swig_ptr_from_LongTensor(I)
    Dptr = swig_ptr_from_FloatTensor(D)
    index.search_c(n, xptr,
                   k, Dptr, Iptr)
    torch.cuda.synchronize()
    return D, I

def build_edges(spatial, r_max, k_max, res, return_indices=False):
    
    index_flat = faiss.IndexFlatL2(spatial.shape[1])
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    spatial_np = spatial.cpu().detach().numpy()
    gpu_index_flat.add(spatial_np)
    
    D, I = search_index_pytorch(gpu_index_flat, spatial, k_max)
    
    D, I = D[:,1:], I[:,1:]
    ind = torch.Tensor.repeat(torch.arange(I.shape[0]), (I.shape[1], 1), 1).T.to(device)
    edge_list = torch.stack([ind[D <= r_max**2], I[D <= r_max**2]])
    
    if return_indices:
        return edge_list, D, I, ind
    else:
        return edge_list
    
    