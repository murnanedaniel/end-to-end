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

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("train.py")
    add_arg = parser.add_argument

    add_arg("config", nargs="?", default="configs/hello.yaml")
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

    return parser.parse_args()


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

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Dataset processing
    pt_cut = config["selection"]["pt_min"]
    train_number = config["selection"]["train_number"]
    test_number = config["selection"]["test_number"]
    load_dir = config["input_dir"]
    model_dir = config["model_dir"]

    # Construct experiment name
    group = str(pt_cut) + "pt_cut"
    if endcaps:
        group += "_endcaps"
    print("Running experiment group:", group)

    train_path = os.path.join(load_dir, group, str(train_number) + "_events_train")
    test_path = os.path.join(load_dir, group, str(test_number) + "_events_test")

    train_dataset = torch.load(train_path)
    test_dataset = torch.load(test_path)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Model config

    m_configs = config["model"]
    model = EmbeddingToAGNN(**m_configs).to(device)
    #     multi_loss = MultiNoiseLoss(n_losses=2).to(device)
    m_configs.update(config["training"])
    m_configs.update(config["selection"])
    wandb.init(group=group, config=m_configs)
    wandb.run.save()
    print(wandb.run.name)
    model_name = wandb.run.name
    wandb.watch(model, log="all")

    # Optimizer config

    #     optimizer = torch.optim.AdamW([
    #     {'params': model.emb_network.parameters()},
    #     {'params': chain(model.node_network.parameters(), model.edge_network.parameters(), model.input_network.parameters())},
    #     {'params': multi_loss.noise_params}],
    #             lr = 0.001, weight_decay=1e-3, amsgrad=True)

    # Scheduler config

    #     lambda1 = lambda ep: 1 / (args.lr_1**(ep//10))
    #     lambda2 = lambda ep: 1 / (args.lr_2**(ep//30))
    #     lambda3 = lambda ep: 1 / (args.lr_3**(ep//10))
    #     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2, lambda3])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=m_configs["lr"],
        weight_decay=m_configs["weight_decay"],
        amsgrad=True,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=m_configs["factor"], patience=m_configs["patience"]
    )

    # Training loop

    for epoch in range(30):
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

        print(
            "Epoch: {}, Edge Accuracy: {:.4f}, Edge Purity: {:.4f}, Edge Efficiency: {:.4f}, Cluster Purity: {:.4f}, Cluster Efficiency: {:.4f}, Loss: {:.4f}, LR: {} in time {}".format(
                epoch,
                edge_acc,
                edge_pur,
                edge_eff,
                cluster_pur,
                cluster_eff,
                val_loss,
                scheduler._last_lr,
                tt() - tic,
            )
        )


if __name__ == "__main__":

    # Parse the command line
    args = parse_args()
    #     print(args)

    main(args)
