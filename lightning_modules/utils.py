import os
import logging
from memory_profiler import profile

import faiss
import faiss.contrib.torch_utils
from pytorch3d import ops
import torch
import torch.nn as nn
from torch.utils.data import random_split
import scipy as sp
import numpy as np
import pandas as pd
import trackml.dataset
import random

from scipy.optimize import root_scalar as root

device = "cuda" if torch.cuda.is_available() else "cpu"


class MultiNoiseLoss(nn.Module):
    def __init__(self, n_losses, device):
        super(MultiNoiseLoss, self).__init__()
        self.noise_params = torch.rand(n_losses, requires_grad=True, device=device)

    def forward(self, losses):

        total_loss = 0
        for i, loss in enumerate(losses):
            total_loss += (1 / torch.square(self.noise_params[i])) * loss + torch.log(
                self.noise_params[i]
            )

        return total_loss


def load_dataset(input_dir, num, pt_cut):
    if input_dir is not None:
        all_events = os.listdir(input_dir)
        all_events = sorted([os.path.join(input_dir, event) for event in all_events])
        random.shuffle(all_events)
        loaded_events = []
        for event in all_events[:num]:
            try:
                loaded_event = torch.load(event, map_location=torch.device("cpu"))
                loaded_events.append(loaded_event)
            #                 logging.info('Loaded event: {}'.format(loaded_event.event_file))
            except:
                logging.info("Corrupted event file: {}".format(event))
        loaded_events = filter_hit_pt(loaded_events, pt_cut)
        return loaded_events
    else:
        return None


def load_processed_dataset(input_dir, num, min_edges=None):
    if input_dir is not None:
        all_events = os.listdir(input_dir)
        all_events = sorted([os.path.join(input_dir, event) for event in all_events])
        random.shuffle(all_events)
        loaded_events = []
        for event in all_events[:num]:
            try:
                loaded_event = torch.load(event, map_location=torch.device("cpu"))
                if (min_edges is None) or (
                    loaded_event.sub_edge_index.sum() > min_edges
                ):
                    loaded_events.append(loaded_event)
            #                 logging.info('Loaded event: {}'.format(loaded_event.event_file))
            except:
                logging.info("Corrupted event file: {}".format(event))
        return loaded_events
    else:
        return None


def split_datasets(input_dir, train_split, pt_cut=0, seed=1):
    """
    Prepare the random Train, Val, Test split, using a seed for reproducibility. Seed should be
    changed across final varied runs, but can be left as default for experimentation.
    """
    torch.manual_seed(seed)
    loaded_events = load_dataset(input_dir, sum(train_split), pt_cut)
    train_events, val_events, test_events = random_split(loaded_events, train_split)

    return train_events, val_events, test_events


def fetch_pt(event):
    # Handle event in batched form
    event_file = (
        event.event_file[0] if type(event.event_file) is list else event.event_file
    )
    # Load the truth data from the event directory
    truth = trackml.dataset.load_event(event_file, parts=["truth"])[0]
    hid = event.hid[0] if type(event.hid) is list else event.hid
    merged_truth = pd.DataFrame(hid.cpu().numpy(), columns=["hit_id"]).merge(
        truth, on="hit_id"
    )
    pt = np.sqrt(merged_truth.tpx ** 2 + merged_truth.tpy ** 2)

    return pt.to_numpy()


def fetch_type(event):
    # Handle event in batched form
    event_file = (
        event.event_file[0] if type(event.event_file) is list else event.event_file
    )
    # Load the truth data from the event directory
    truth, particles = trackml.dataset.load_event(
        event_file, parts=["truth", "particles"]
    )
    hid = event.hid[0] if type(event.hid) is list else event.hid
    merged_truth = truth.merge(particles, on="particle_id")
    p_type = pd.DataFrame(hid.cpu().numpy(), columns=["hit_id"]).merge(
        merged_truth, on="hit_id"
    )
    p_type = p_type.particle_type.values

    return p_type


def filter_edge_pt(events, pt_cut=0):
    # Handle event in batched form
    if type(events) is not list:
        events = [events]

    if pt_cut > 0:
        for event in events:
            pt = fetch_pt(event)
            edge_subset = pt[event.edge_index] > pt_cut
            combined_subset = edge_subset[0] & edge_subset[1]
            event.edge_index = event.edge_index[:, combined_subset]
            event.y = event.y[combined_subset]
            event.y_pid = event.y_pid[combined_subset]

    return events


def filter_hit_pt(events, pt_cut=0):
    # Handle event in batched form
    if type(events) is not list:
        events = [events]

    if pt_cut > 0:
        for event in events:
            pt = fetch_pt(event)
            hit_subset = pt > pt_cut
            event.cell_data = event.cell_data[hit_subset]
            event.hid = event.hid[hit_subset]
            event.x = event.x[hit_subset]
            event.pid = event.pid[hit_subset]
            event.layers = event.layers[hit_subset]
            if "pt" in event.__dict__.keys():
                event.pt = event.pt[hit_subset]
            if "layerless_true_edges" in event.__dict__.keys():
                event.layerless_true_edges, remaining_edges = reset_edge_id(
                    hit_subset, event.layerless_true_edges
                )

            if "layerwise_true_edges" in event.__dict__.keys():
                event.layerwise_true_edges, remaining_edges = reset_edge_id(
                    hit_subset, event.layerwise_true_edges
                )

            if "weights" in event.__dict__.keys():
                event.weights = event.weights[remaining_edges]

    return events


def reset_edge_id(subset, graph):
    subset_ind = np.where(subset)[0]
    filler = -np.ones((graph.max() + 1,))
    filler[subset_ind] = np.arange(len(subset_ind))
    graph = torch.from_numpy(filler[graph]).long()
    exist_edges = (graph[0] >= 0) & (graph[1] >= 0)
    graph = graph[:, exist_edges]

    return graph, exist_edges


def push_all_negs_back(a):
    # Based on http://stackoverflow.com/a/42859463/3293881
    valid_mask = a != -1
    flipped_mask = valid_mask.sum(1, keepdims=1) > np.arange(a.shape[1] - 1, -1, -1)
    flipped_mask = flipped_mask[:, ::-1]
    a[flipped_mask] = a[valid_mask]
    a[~flipped_mask] = -1
    return a


def graph_intersection(
    pred_graph, truth_graph, using_weights=False, weights_bidir=None
):

    array_size = max(pred_graph.max().item(), truth_graph.max().item()) + 1

    l1 = pred_graph.cpu().numpy()
    l2 = truth_graph.cpu().numpy()
    e_1 = sp.sparse.coo_matrix(
        (np.ones(l1.shape[1]), l1), shape=(array_size, array_size)
    ).tocsr()
    e_2 = sp.sparse.coo_matrix(
        (np.ones(l2.shape[1]), l2), shape=(array_size, array_size)
    ).tocsr()
    del l1

    e_intersection = e_1.multiply(e_2) - ((e_1 - e_2) > 0)
    del e_1
    del e_2

    if using_weights:
        weights_list = weights_bidir.cpu().numpy()
        weights_sparse = sp.sparse.coo_matrix(
            (weights_list, l2), shape=(array_size, array_size)
        ).tocsr()
        del weights_list
        del l2
        new_weights = weights_sparse[e_intersection.astype("bool")]
        del weights_sparse
        new_weights = torch.from_numpy(np.array(new_weights)[0])

    e_intersection = e_intersection.tocoo()
    new_pred_graph = torch.from_numpy(
        np.vstack([e_intersection.row, e_intersection.col])
    ).long()  # .to(device)
    y = torch.from_numpy(e_intersection.data > 0)  # .to(device)
    del e_intersection

    if using_weights:
        return new_pred_graph, y, new_weights
    else:
        return new_pred_graph, y


def build_edges(spatial, r_max, k_max, return_indices=False, target_spatial=None):

    if k_max > 200:
        if device == "cuda":
            res = faiss.StandardGpuResources()
            if target_spatial is None:
                D, I = faiss.knn_gpu(res, spatial, spatial, k_max)
            else:
                D, I = faiss.knn_gpu(res, spatial, target_spatial, k_max)
        elif device == "cpu":
            index = faiss.IndexFlatL2(spatial.shape[1])
            index.add(spatial)
            if target_spatial is None:
                D, I = index.search(spatial, k_max)
            else:
                D, I = index.search(target_spatial, k_max)

    else:
        if target_spatial is None:
            knn_object = ops.knn_points(
                spatial.unsqueeze(0), spatial.unsqueeze(0), K=k_max, return_sorted=False
            )
        else:
            knn_object = ops.knn_points(
                spatial.unsqueeze(0),
                target_spatial.unsqueeze(0),
                K=k_max,
                return_sorted=False,
            )
        I = knn_object.idx[0]
        D = knn_object.dists[0]

    # Overlay the "source" hit ID onto each neighbour ID (this is necessary as the FAISS algo does some shortcuts)
    ind = torch.Tensor.repeat(
        torch.arange(I.shape[0], device=device), (I.shape[1], 1), 1
    ).T
    edge_list = torch.stack([ind[D <= r_max ** 2], I[D <= r_max ** 2]])

    # Remove self-loops
    edge_list = edge_list[:, edge_list[0] != edge_list[1]]

    if return_indices:
        return edge_list, D, I, ind
    else:
        return edge_list


def build_knn(spatial, k):

    if device == "cuda":
        res = faiss.StandardGpuResources()
        _, I = faiss.knn_gpu(res, spatial, spatial, k)
    elif device == "cpu":
        index = faiss.IndexFlatL2(spatial.shape[1])
        index.add(spatial)
        _, I = index.search(spatial, k)

    print(spatial)
    print(I)
    ind = torch.Tensor.repeat(
        torch.arange(I.shape[0], device=device), (I.shape[1], 1), 1
    ).T
    edge_list = torch.stack([ind, I])

    # Remove self-loops
    edge_list = edge_list[:, edge_list[0] != edge_list[1]]

    return edge_list


def get_best_run(run_label, wandb_save_dir):
    for (root_dir, dirs, files) in os.walk(wandb_save_dir + "/wandb"):
        if run_label in dirs:
            run_root = root_dir

    best_run_base = os.path.join(run_root, run_label, "checkpoints")
    best_run = os.listdir(best_run_base)
    best_run_path = os.path.join(best_run_base, best_run[0])

    return best_run_path


# --------------------------- Model Building --------------------------


def make_mlp(
    input_size,
    sizes,
    hidden_activation="ReLU",
    output_activation="ReLU",
    layer_norm=False,
):
    """Construct an MLP with specified fully-connected layers."""
    hidden_activation = getattr(nn, hidden_activation)
    if output_activation is not None:
        output_activation = getattr(nn, output_activation)
    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes
    # Hidden layers
    for i in range(n_layers - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i + 1]))
        layers.append(hidden_activation())
    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1]))
        layers.append(output_activation())
    return nn.Sequential(*layers)


# -------------------------- Performance Evaluation -------------------


def embedding_model_evaluation(model, trainer, fom="eff", fixed_value=0.96):

    # Seed solver with one batch, then run on full test dataset
    sol = root(
        evaluate_set_root,
        args=(model, trainer, fixed_value, fom),
        x0=0.6,
        x1=1.0,
        xtol=0.001,
    )
    print("Seed solver complete, radius:", sol.root)

    # Return ( (efficiency, purity), radius_size)
    return evaluate_set_metrics(sol.root, model, trainer), sol.root


def evaluate_set_root(r, model, trainer, goal=0.96, fom="eff"):
    eff, pur = evaluate_set_metrics(r, model, trainer)

    if fom == "eff":
        return eff - goal

    elif fom == "pur":
        return pur - goal


def get_metrics(test_results, model):

    ps = [len(result["truth"]) for result in test_results]
    ts = [result["truth_graph"].shape[1] for result in test_results]
    tps = [result["truth"].sum() for result in test_results]

    efficiencies = [tp / t for (t, tp) in zip(ts, tps)]
    purities = [tp / p for (p, tp) in zip(ps, tps)]

    mean_efficiency = np.mean(efficiencies)
    mean_purity = np.mean(purities)

    return mean_efficiency, mean_purity


def evaluate_set_metrics(r_test, model, trainer):

    model.hparams.r_test = r_test
    test_results = trainer.test(ckpt_path=None)

    mean_efficiency, mean_purity = get_metrics(test_results, model)

    print(mean_purity, mean_efficiency)

    return mean_efficiency, mean_purity
