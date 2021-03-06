"""
Data preparation script for GNN tracking.

This script processes the TrackML dataset and produces graph data on disk.
"""

# System
import os
import argparse
import logging
import multiprocessing as mp
from functools import partial

# Externals
import yaml
import numpy as np
import pandas as pd
import trackml.dataset

import torch
from torch_geometric.data import Data

from itertools import permutations
import itertools

# Locals
# from datasets.graph import Graph, save_graphs


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("prepare.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="configs/prepare_trackml.yaml")
    add_arg("--n-workers", type=int, default=1)
    add_arg("--task", type=int, default=0)
    add_arg("--n-tasks", type=int, default=1)
    add_arg("-v", "--verbose", action="store_true")
    add_arg("--show-config", action="store_true")
    add_arg("--interactive", action="store_true")
    return parser.parse_args()


def select_hits(hits, truth, particles, pt_min=0, endcaps=False, noise=False):
    # Barrel volume and layer ids
    if endcaps:
        vlids = [
            (7, 2),
            (7, 4),
            (7, 6),
            (7, 8),
            (7, 10),
            (7, 12),
            (7, 14),
            (8, 2),
            (8, 4),
            (8, 6),
            (8, 8),
            (9, 2),
            (9, 4),
            (9, 6),
            (9, 8),
            (9, 10),
            (9, 12),
            (9, 14),
            (12, 2),
            (12, 4),
            (12, 6),
            (12, 8),
            (12, 10),
            (12, 12),
            (13, 2),
            (13, 4),
            (13, 6),
            (13, 8),
            (14, 2),
            (14, 4),
            (14, 6),
            (14, 8),
            (14, 10),
            (14, 12),
            (16, 2),
            (16, 4),
            (16, 6),
            (16, 8),
            (16, 10),
            (16, 12),
            (17, 2),
            (17, 4),
            (18, 2),
            (18, 4),
            (18, 6),
            (18, 8),
            (18, 10),
            (18, 12),
        ]
    else:
        vlids = [
            (8, 2),
            (8, 4),
            (8, 6),
            (8, 8),
            (13, 2),
            (13, 4),
            (13, 6),
            (13, 8),
            (17, 2),
            (17, 4),
        ]
    n_det_layers = len(vlids)
    # Select barrel layers and assign convenient layer number [0-9]
    vlid_groups = hits.groupby(["volume_id", "layer_id"])
    hits = pd.concat(
        [vlid_groups.get_group(vlids[i]).assign(layer=i) for i in range(n_det_layers)]
    )
    if noise is False:
        # Calculate particle transverse momentum
        pt = np.sqrt(particles.px ** 2 + particles.py ** 2)
        # Applies pt cut, removes noise hits
        particles = particles[pt > pt_min]
        truth = truth[["hit_id", "particle_id"]].merge(
            particles[["particle_id", "vx", "vy", "vz"]], on="particle_id"
        )
    else:
        # Calculate particle transverse momentum
        pt = np.sqrt(truth.tpx ** 2 + truth.tpy ** 2)
        # Applies pt cut
        truth = truth[pt > pt_min]
        truth.loc[truth["particle_id"] == 0, "particle_id"] = float("NaN")
    # Calculate derived hits variables
    r = np.sqrt(hits.x ** 2 + hits.y ** 2)
    phi = np.arctan2(hits.y, hits.x)
    # Select the data columns we need
    hits = (
        hits[["hit_id", "x", "y", "z", "layer"]]
        .assign(r=r, phi=phi)
        .merge(truth[["hit_id", "particle_id", "vx", "vy", "vz"]], on="hit_id")
    )
    # (DON'T) Remove duplicate hits
    #     hits = hits.loc[
    #         hits.groupby(['particle_id', 'layer'], as_index=False).r.idxmin()
    #     ]
    return hits


def build_event(
    event_file,
    pt_min,
    feature_scale,
    adjacent=True,
    endcaps=False,
    layerless=True,
    layerwise=True,
    noise=False,
):
    # Get true edge list using the ordering by R' = distance from production vertex of each particle
    hits, particles, truth = trackml.dataset.load_event(
        event_file, parts=["hits", "particles", "truth"]
    )
    hits = select_hits(
        hits, truth, particles, pt_min=pt_min, endcaps=endcaps, noise=noise
    ).assign(evtid=int(event_file[-9:]))
    layers = hits.layer.to_numpy()

    # Handle which truth graph(s) are being produced
    layerless_true_edges, layerwise_true_edges = None, None

    if layerless:
        hits = hits.assign(
            R=np.sqrt(
                (hits.x - hits.vx) ** 2
                + (hits.y - hits.vy) ** 2
                + (hits.z - hits.vz) ** 2
            )
        )
        hits = hits.sort_values("R").reset_index(drop=True).reset_index(drop=False)
        hit_list = (
            hits.groupby(["particle_id", "layer"], sort=False)["index"]
            .agg(lambda x: list(x))
            .groupby(level=0)
            .agg(lambda x: list(x))
        )

        e = []
        for row in hit_list.values:
            for i, j in zip(row[0:-1], row[1:]):
                e.extend(list(itertools.product(i, j)))

        layerless_true_edges = np.array(e).T
        print(
            "Layerless truth graph built for",
            event_file,
            "with size",
            layerless_true_edges.shape,
        )

    if layerwise:
        # Get true edge list using the ordering of layers
        records_array = hits.particle_id.to_numpy()
        idx_sort = np.argsort(records_array)
        sorted_records_array = records_array[idx_sort]
        _, idx_start, _ = np.unique(
            sorted_records_array, return_counts=True, return_index=True
        )
        # sets of indices
        res = np.split(idx_sort, idx_start[1:])
        layerwise_true_edges = np.concatenate(
            [
                list(permutations(i, r=2))
                for i in res
                if len(list(permutations(i, r=2))) > 0
            ]
        ).T
        if adjacent:
            layerwise_true_edges = layerwise_true_edges[
                :,
                (
                    layers[layerwise_true_edges[1]] - layers[layerwise_true_edges[0]]
                    == 1
                ),
            ]
        print(
            "Layerwise truth graph built for",
            event_file,
            "with size",
            layerwise_true_edges.shape,
        )

    return (
        hits[["r", "phi", "z"]].to_numpy() / feature_scale,
        hits.particle_id.to_numpy(),
        layers,
        layerless_true_edges,
        layerwise_true_edges,
        hits["hit_id"].to_numpy(),
    )


def prepare_event(
    event_file,
    output_dir,
    pt_min=0,
    adjacent=True,
    endcaps=False,
    layerless=True,
    layerwise=True,
    noise=False,
):

    evtid = int(event_file[-9:])

    print("Preparing", evtid)

    feature_scale = [1000, np.pi, 1000]

    X, pid, layers, layerless_true_edges, layerwise_true_edges, hid = build_event(
        event_file,
        pt_min,
        feature_scale,
        adjacent=adjacent,
        endcaps=endcaps,
        layerless=layerless,
        layerwise=layerwise,
        noise=noise,
    )

    print("Building Data for", evtid)

    data = Data(
        x=torch.from_numpy(X).float(),
        pid=torch.from_numpy(pid),
        layers=torch.from_numpy(layers),
        event_file=event_file,
        hid=torch.from_numpy(hid),
    )
    if layerless_true_edges is not None:
        data.layerless_true_edges = torch.from_numpy(layerless_true_edges)
    if layerwise_true_edges is not None:
        data.layerwise_true_edges = torch.from_numpy(layerwise_true_edges)

    filename = os.path.join(output_dir, str(evtid))
    print("Event", evtid, "writing graphs to", filename)
    with open(filename, "wb") as pickle_file:
        torch.save(data, pickle_file)


def main():
    """Main function"""

    # Parse the command line
    args = parse_args()

    # Setup logging
    log_format = "%(asctime)s %(levelname)s %(message)s"
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=log_format)
    logging.info("Initializing")
    if args.show_config:
        logging.info("Command line config: %s" % args)

    # Load configuration
    with open(args.config) as f:
        config = yaml.load(f)
    if args.task == 0:
        logging.info("Configuration: %s" % config)

    # Find the input files
    input_dir = config["input_dir"]
    all_files = os.listdir(input_dir)
    all_events = sorted(
        np.unique([os.path.join(input_dir, event[:14]) for event in all_files])
    )[: config["n_files"]]

    # Split the input files by number of tasks and select my chunk only
    all_events = np.array_split(all_events, args.n_tasks)[args.task]

    # Prepare output
    output_dir = os.path.expandvars(config["output_dir"])
    os.makedirs(output_dir, exist_ok=True)
    logging.info("Writing outputs to " + output_dir)

    # Process input files with a worker pool
    with mp.Pool(processes=args.n_workers) as pool:
        process_func = partial(
            prepare_event, output_dir=output_dir, **config["selection"]
        )
        pool.map(process_func, all_events)

    # Drop to IPython interactive shell
    if args.interactive:
        logging.info("Starting IPython interactive session")
        import IPython

        IPython.embed()

    logging.info("All done!")


if __name__ == "__main__":
    main()
