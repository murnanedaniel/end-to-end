# System imports
import os
import sys
from pprint import pprint as pp
from time import time as tt
import inspect

# External imports
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.decomposition import PCA
from sklearn.metrics import auc
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from mpl_toolkits.mplot3d import Axes3D
import argparse
from itertools import permutations

from itertools import chain
import trackml.dataset

import ipywidgets as widgets
from ipywidgets import interact, interact_manual

# Pick up local packages
sys.path.append("..")

# Local imports
from prepare import select_hits
from toy_utils import *
from models import *
from trainers import *

# Get rid of RuntimeWarnings, gross
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

import wandb


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("train.py")
    add_arg = parser.add_argument

    add_arg("--hidden-dim", type=int, default=None, help="Hidden layer dimension size")
    add_arg(
        "--n-graph-iters", type=int, default=None, help="Number of graph iterations"
    )
    add_arg(
        "--emb-dim",
        type=int,
        default=None,
        help="Number of spatial embedding dimensions",
    )
    add_arg(
        "--emb-hidden",
        type=int,
        default=None,
        help="Number of embedding hidden dimensions",
    )
    add_arg("--nb-layer", type=int, default=None, help="Number of embedding layers")
    add_arg("--r-val", type=float, default=None, help="Radius of graph construction")
    add_arg("--r-train", type=float, default=None, help="Radius of embedding training")
    add_arg("--margin", type=float, default=None, help="Radius of hinge loss")
    add_arg("--lr-1", type=float, default=None, help="Embedding loss learning rate")
    add_arg("--lr-2", type=float, default=None, help="AGNN loss learning rate")
    add_arg("--lr-3", type=float, default=None, help="Weight balance learning rate")
    add_arg("--weight", type=float, default=None, help="Positive weight in AGNN")
    add_arg("--train-size", type=int, default=None, help="Number of train population")
    add_arg("--val-size", type=int, default=None, help="Number of validate population")
    add_arg("--pt-cut", type=float, default=None, help="Cutoff for momentum")
    add_arg("--adjacent", type=bool, default=False, help="Enforce adjacent layers?")
    add_arg("--pretrain-epochs", type=int, default=5)

    return parser.parse_args()


def build_event(event_file, pt_min, feature_scale, adjacent):
    hits, particles, truth = trackml.dataset.load_event(
        event_file, parts=["hits", "particles", "truth"]
    )
    hits = select_hits(hits, truth, particles, pt_min=pt_min).assign(
        evtid=int(event_file[-9:])
    )
    layers = hits.layer.to_numpy()

    # Get true edge list
    records_array = hits.particle_id.to_numpy()
    idx_sort = np.argsort(records_array)
    sorted_records_array = records_array[idx_sort]
    _, idx_start, _ = np.unique(
        sorted_records_array, return_counts=True, return_index=True
    )
    # sets of indices
    res = np.split(idx_sort, idx_start[1:])
    true_edges = np.concatenate(
        [list(permutations(i, r=2)) for i in res if len(list(permutations(i, r=2))) > 0]
    )
    if adjacent:
        true_edges = true_edges[
            (layers[true_edges.T[1]] - layers[true_edges.T[0]] == 1)
        ]

    return (
        hits[["r", "phi", "z"]].to_numpy() / feature_scale,
        hits.particle_id.to_numpy(),
        layers,
        true_edges,
    )


def prepare_event(event_file, pt_min, feature_scale, adjacent=True):
    #     print("Preparing",event_file)
    X, pid, layers, true_edges = build_event(
        event_file, pt_min, feature_scale, adjacent
    )
    data = Data(
        x=torch.from_numpy(X).float(),
        pid=torch.from_numpy(pid),
        layers=torch.from_numpy(layers),
        true_edges=torch.from_numpy(true_edges),
    )
    return data


def save_model(epoch, model, optimizer, scheduler, running_loss, config, PATH):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": running_loss,
            "config": config,
        },
        os.path.join("model_comparisons/", PATH),
    )


def main(args):
    #     print(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset processing

    input_dir = "/global/cscratch1/sd/danieltm/ExaTrkX/trackml/train_all/"
    all_events = os.listdir(input_dir)
    all_events = [input_dir + event[:14] for event in all_events]
    np.random.shuffle(all_events)

    train_dataset = [
        prepare_event(event_file, args.pt_cut, [1000, np.pi, 1000], args.adjacent)
        for event_file in all_events[:4000]
    ]
    test_dataset = [
        prepare_event(event_file, args.pt_cut, [1000, np.pi, 1000], args.adjacent)
        for event_file in all_events[-args.val_size :]
    ]
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Model config
    e_configs = {
        "in_channels": 3,
        "emb_hidden": args.emb_hidden,
        "nb_layer": args.nb_layer,
        "emb_dim": args.emb_dim,
    }
    m_configs = {
        "in_channels": 3,
        "emb_hidden": args.emb_hidden,
        "nb_layer": args.nb_layer,
        "emb_dim": args.emb_dim,
        "r": args.r_val,
        "hidden_dim": args.hidden_dim,
        "n_graph_iters": args.n_graph_iters,
    }
    other_configs = {
        "weight": args.weight,
        "r_train": args.r_train,
        "r_val": args.r_val,
        "margin": args.margin,
        "reduction": "mean",
    }

    # Create and pretrain embedding
    embedding_model = Embedding(**e_configs).to(device)
    wandb.init(group="EmbeddingToAGNN_PurTimesEff", config=m_configs)
    embedding_optimizer = torch.optim.Adam(
        embedding_model.parameters(), lr=0.0005, weight_decay=1e-3, amsgrad=True
    )

    for epoch in range(args.pretrain_epochs):
        tic = tt()
        embedding_model.train()
        cluster_pur, train_loss = train_emb(
            embedding_model, train_loader, embedding_optimizer, other_configs
        )

        embedding_model.eval()
        with torch.no_grad():
            cluster_pur, cluster_eff, val_loss, av_nhood_size = evaluate_emb(
                embedding_model, test_loader, other_configs
            )
        wandb.log(
            {
                "val_loss": val_loss,
                "train_loss": train_loss,
                "cluster_pur": cluster_pur,
                "cluster_eff": cluster_eff,
                "av_nhood_size": av_nhood_size,
            }
        )

    # Create and train main model
    model = EmbeddingToAGNNPretrained(**m_configs, pretrained_model=embedding_model).to(
        device
    )
    multi_loss = MultiNoiseLoss(n_losses=2).to(device)
    m_configs.update(other_configs)
    wandb.run.save()
    print(wandb.run.name)
    model_name = wandb.run.name
    wandb.watch(model, log="all")

    # Optimizer config

    optimizer = torch.optim.AdamW(
        [
            {"params": model.emb_network.parameters()},
            {
                "params": chain(
                    model.node_network.parameters(),
                    model.edge_network.parameters(),
                    model.input_network.parameters(),
                )
            },
            {"params": multi_loss.noise_params},
        ],
        lr=0.001,
        weight_decay=1e-3,
        amsgrad=True,
    )

    # Scheduler config

    lambda1 = lambda ep: 1 / (args.lr_1 ** (ep // 10))
    lambda2 = lambda ep: 1 / (args.lr_2 ** (ep // 30))
    lambda3 = lambda ep: 1 / (args.lr_3 ** (ep // 10))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[lambda1, lambda2, lambda3]
    )

    # Training loop

    for epoch in range(50):
        tic = tt()
        model.train()
        if args.adjacent:
            edge_acc, cluster_pur, train_loss = balanced_adjacent_train(
                model, train_loader, optimizer, multi_loss, m_configs
            )
        else:
            edge_acc, cluster_pur, train_loss = balanced_train(
                model, train_loader, optimizer, multi_loss, m_configs
            )
        #         print("Training loss:", train_loss)

        model.eval()
        if args.adjacent:
            with torch.no_grad():
                (
                    edge_acc,
                    edge_pur,
                    edge_eff,
                    cluster_pur,
                    cluster_eff,
                    val_loss,
                    av_nhood_size,
                ) = evaluate_adjacent(model, test_loader, multi_loss, m_configs)
        else:
            with torch.no_grad():
                (
                    edge_acc,
                    edge_pur,
                    edge_eff,
                    cluster_pur,
                    cluster_eff,
                    val_loss,
                    av_nhood_size,
                ) = evaluate(model, test_loader, multi_loss, m_configs)
        scheduler.step()
        wandb.log(
            {
                "val_loss": val_loss,
                "train_loss": train_loss,
                "edge_acc": edge_acc,
                "edge_pur": edge_pur,
                "edge_eff": edge_eff,
                "cluster_pur": cluster_pur,
                "cluster_eff": cluster_eff,
                "lr": scheduler._last_lr[0],
                "combined_performance": edge_eff * cluster_eff * edge_pur + cluster_pur,
                "combined_efficiency": edge_eff * cluster_eff * edge_pur,
                "noise_1": multi_loss.noise_params[0].item(),
                "noise_2": multi_loss.noise_params[1].item(),
                "av_nhood_size": av_nhood_size,
            }
        )

        save_model(
            epoch,
            model,
            optimizer,
            scheduler,
            cluster_eff,
            m_configs,
            "EmbeddingToAGNN/" + model_name + ".tar",
        )


#         print('Epoch: {}, Edge Accuracy: {:.4f}, Edge Purity: {:.4f}, Edge Efficiency: {:.4f}, Cluster Purity: {:.4f}, Cluster Efficiency: {:.4f}, Loss: {:.4f}, LR: {} in time {}'.format(epoch, edge_acc, edge_pur, edge_eff, cluster_pur, cluster_eff, val_loss, scheduler._last_lr, tt()-tic))

if __name__ == "__main__":

    # Parse the command line
    args = parse_args()
    #     print(args)

    main(args)
