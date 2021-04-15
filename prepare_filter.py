# System imports
import os
import sys
from pprint import pprint as pp
from time import time as tt
import inspect
import importlib

# External imports
import matplotlib.pyplot as plt
import matplotlib.colors
import scipy as sp
from sklearn.decomposition import PCA
from sklearn.metrics import auc
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch_geometric.data import DataLoader
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import shuffle, choice

from itertools import chain

from torch.nn import Linear
import torch.nn.functional as F
from torch_scatter import scatter, segment_csr, scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_cluster import knn_graph, radius_graph
import trackml.dataset
import torch_geometric
from itertools import permutations
import itertools
import plotly.express as px

import ipywidgets as widgets
from ipywidgets import interact, interact_manual

from sklearn.cluster import DBSCAN
from sklearn import metrics

# Local imports
from prepare_utils import *
from performance_utils import *
from toy_utils import *
from models import *
from trainers import *
from lightning_modules.embedding_scanner import Embedding_Model

# Get rid of RuntimeWarnings, gross
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
device = "cuda" if torch.cuda.is_available() else "cpu"
import wandb
import faiss

props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
torch_seed = 0


def process_dataset(dataset, number, save_dir, model, ratio, train=True):
    for i, batch in enumerate(dataset[:number]):
        tic = tt()

        if not os.path.exists(os.path.join(save_dir, batch.event_file[-4:])):
            data = batch.to(device)
            spatial = model(torch.cat([data.cell_data, data.x], axis=-1))

            e_spatial = build_edges(spatial, 1.4, 1024, res)

            # Get the truth graphs
            e_bidir = torch.cat(
                [
                    batch.layerless_true_edges,
                    torch.stack(
                        [batch.layerless_true_edges[1], batch.layerless_true_edges[0]],
                        axis=1,
                    ).T,
                ],
                axis=-1,
            )

            # Get random edge list
            e_spatial = build_edges(spatial, 1.4, 1024, res)

            # Remove duplicate edges by distance from vertex
            R_dist = torch.sqrt(batch.x[:, 0] ** 2 + batch.x[:, 2] ** 2)
            e_spatial = e_spatial[:, (R_dist[e_spatial[0]] < R_dist[e_spatial[1]])]

            e_spatial, y = graph_intersection(e_spatial, e_bidir)

            # Re-introduce random direction, to avoid training bias
            random_flip = torch.randint(2, (e_spatial.shape[1],)).bool()
            e_spatial[0, random_flip], e_spatial[1, random_flip] = (
                e_spatial[1, random_flip],
                e_spatial[0, random_flip],
            )

            batch.embedding = spatial.cpu().detach()

            if train and (
                ratio != 0
            ):  # Sample only ratio:1 fake:true edges, to keep trainset manageable

                num_true = y.sum()
                fake_indices = choice(
                    np.where(~y)[0], int(num_true * ratio), replace=True
                )
                true_indices = np.where(y)[0]
                combined_indices = np.concatenate([true_indices, fake_indices])
                shuffle(combined_indices)

                batch.e_radius = e_spatial[:, combined_indices].cpu()
                batch.y = torch.from_numpy(y[combined_indices]).float()

            else:
                batch.e_radius = e_spatial.cpu()
                batch.y = torch.from_numpy(y).float()

            with open(
                os.path.join(save_dir, batch.event_file[-4:]), "wb"
            ) as pickle_file:
                torch.save(batch, pickle_file)

            print(
                i,
                "saved in time",
                tt() - tic,
                "with efficiency",
                (batch.y.sum() / batch.layerless_true_edges.shape[1]).item(),
                "and purity",
                (batch.y.sum() / batch.e_radius.shape[1]).item(),
            )

        else:
            print(i, "already built")


def main():
    """Main function"""

    # Load raw events
    pt_cut = 0
    embedding_train_number = 1000
    embedding_test_number = 100
    load_dir = (
        "/global/cscratch1/sd/danieltm/ExaTrkX/trackml_processed/truth_processed/"
    )
    train_path = os.path.join(
        load_dir, str(pt_cut) + "_pt_cut", str(embedding_train_number) + "_events_train"
    )
    test_path = os.path.join(
        load_dir, str(pt_cut) + "_pt_cut", str(embedding_test_number) + "_events_test"
    )

    train_dataset = torch.load(train_path)
    test_dataset = torch.load(test_path)

    print("Raw events loaded")

    # Load embedding model
    run_label = "qy6e0oi9"

    best_run_path = get_best_run(
        run_label, wandb_save_dir="/global/cscratch1/sd/danieltm/ExaTrkX/wandb_data"
    )

    model = Embedding_Model.load_from_checkpoint(best_run_path)
    model = model.to(device)

    print("Embedding model loaded")

    # Process filter candidates
    ratio = 0
    filter_train_number = 1000
    filter_test_number = 100

    save_dir = (
        "/global/cscratch1/sd/danieltm/ExaTrkX/trackml_processed/filter_processed/"
    )
    train_dir = os.path.join(
        save_dir,
        str(pt_cut) + "_pt_cut",
        str(filter_train_number) + "_events_train_cell_info",
    )
    test_dir = os.path.join(
        save_dir,
        str(pt_cut) + "_pt_cut",
        str(filter_test_number) + "_events_test_cell_info",
    )
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print("Directory structure set up")

    # Process testset
    model.eval()
    with torch.no_grad():
        process_dataset(
            test_dataset, filter_test_number, test_dir, model, ratio, train=False
        )
    print("Testset processed")

    # Process trainset (n.b. train is set to true, to only sample the neighbourhoods)
    model.eval()
    with torch.no_grad():
        process_dataset(
            train_dataset, filter_train_number, train_dir, model, ratio, train=True
        )
    print("Trainset processed")


if __name__ == "__main__":
    main()
