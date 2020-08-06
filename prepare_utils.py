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
from itertools import permutations

from torch_geometric.data import Data
import torch

# Externals
import yaml
import numpy as np
import pandas as pd
import trackml.dataset

# Locals
# from datasets.graph import Graph, save_graphs

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('prepare.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/prepare_trackml.yaml')
    add_arg('--n-workers', type=int, default=1)
    add_arg('--task', type=int, default=0)
    add_arg('--n-tasks', type=int, default=1)
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--show-config', action='store_true')
    add_arg('--interactive', action='store_true')
    return parser.parse_args()

def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2 - phi1
    dphi[dphi > np.pi] -= 2*np.pi
    dphi[dphi < -np.pi] += 2*np.pi
    return dphi

def calc_eta(r, z):
    theta = np.arctan2(r, z)
    return -1. * np.log(np.tan(theta / 2.))

def build_event(event_file, pt_min, feature_scale, adjacent=True, endcaps=False, noise=False):
    hits, particles, truth = trackml.dataset.load_event(
        event_file, parts=['hits', 'particles', 'truth'])
    hits = select_hits(hits, truth, particles, pt_min=pt_min, endcaps=endcaps, noise=noise).assign(evtid=int(event_file[-9:]))
    
    if endcaps:
        hits = hits.assign(R=np.sqrt(hits.r**2 + hits.z**2))
        hits = hits.sort_values('R').reset_index(drop=True).reset_index(drop=False)
        layers = hits.layer.to_numpy()
        hit_list = hits.groupby(['particle_id', 'layer'], sort=False)['index'].agg(lambda x: list(x)).groupby(level=0).agg(lambda x: list(x))

        e = []
        for row in hit_list.values:
            for i, j in zip(row[0:-1], row[1:]):
                e.extend(list(itertools.product(i, j)))

        true_edges = np.array(e).T
        
    else:
        layers = hits.layer.to_numpy()
        # Get true edge list
        records_array = hits.particle_id.to_numpy()
        idx_sort = np.argsort(records_array)
        sorted_records_array = records_array[idx_sort]
        _, idx_start, _ = np.unique(sorted_records_array, return_counts=True,
                                return_index=True)
        # sets of indices
        res = np.split(idx_sort, idx_start[1:])
        true_edges = np.concatenate([list(permutations(i, r=2)) for i in res if len(list(permutations(i, r=2))) > 0])
        if adjacent: true_edges = true_edges[(layers[true_edges.T[1]] - layers[true_edges.T[0]] == 1)]
    
    return hits[['r', 'phi', 'z']].to_numpy() / feature_scale, hits.particle_id.to_numpy(), layers, true_edges

def prepare_event(event_file, pt_min, feature_scale, adjacent=True, endcaps=False, noise=False):
    print("Preparing",event_file)
    X, pid, layers, true_edges = build_event(event_file, pt_min, feature_scale, adjacent=adjacent, endcaps=endcaps, noise=noise)
    data = Data(x = torch.from_numpy(X).float(), pid = torch.from_numpy(pid), layers=torch.from_numpy(layers), true_edges= torch.from_numpy(true_edges))
    return data

def select_hits(hits, truth, particles, pt_min=0, endcaps=False, noise=False):
    # Barrel volume and layer ids
    if endcaps:
        vlids = [(7, 2), (7, 4), (7, 6), (7, 8), (7, 10), (7, 12), (7, 14), (8, 2), (8, 4), (8, 6), (8, 8), (9, 2), (9, 4), (9, 6), (9, 8), (9, 10), (9, 12), (9, 14), (12, 2), (12, 4), (12, 6), (12, 8), (12, 10), (12, 12), (13, 2), (13, 4), (13, 6), (13, 8), (14, 2), (14, 4), (14, 6), (14, 8), (14, 10), (14, 12), (16, 2), (16, 4), (16, 6), (16, 8), (16, 10), (16, 12), (17, 2), (17, 4), (18, 2), (18, 4), (18, 6), (18, 8), (18, 10), (18, 12)]
    else:
        vlids = [(8,2), (8,4), (8,6), (8,8), (13,2), (13,4), (13,6), (13,8), (17,2), (17,4)]
    n_det_layers = len(vlids)
    # Select barrel layers and assign convenient layer number [0-9]
    vlid_groups = hits.groupby(['volume_id', 'layer_id'])
    hits = pd.concat([vlid_groups.get_group(vlids[i]).assign(layer=i)
                      for i in range(n_det_layers)])
    if noise is False:
        # Calculate particle transverse momentum
        pt = np.sqrt(particles.px**2 + particles.py**2)
        # Applies pt cut, removes noise hits
        particles = particles[pt > pt_min]
        truth = (truth[['hit_id', 'particle_id']]
                 .merge(particles[['particle_id']], on='particle_id'))
    else:
        # Calculate particle transverse momentum
        pt = np.sqrt(truth.tpx**2 + truth.tpy**2)
        # Applies pt cut
        truth = truth[pt > pt_min]
        truth.loc[truth['particle_id'] == 0,'particle_id'] = float('NaN')
    # Calculate derived hits variables
    r = np.sqrt(hits.x**2 + hits.y**2)
    phi = np.arctan2(hits.y, hits.x)
    # Select the data columns we need
    hits = (hits[['hit_id', 'z', 'layer']]
            .assign(r=r, phi=phi)
            .merge(truth[['hit_id', 'particle_id']], on='hit_id'))
    # (DON'T) Remove duplicate hits
#     hits = hits.loc[
#         hits.groupby(['particle_id', 'layer'], as_index=False).r.idxmin()
#     ]
    return hits

def split_detector_sections(hits, phi_edges, eta_edges):
    """Split hits according to provided phi and eta boundaries."""
    hits_sections = []
    # Loop over sections
    for i in range(len(phi_edges) - 1):
        phi_min, phi_max = phi_edges[i], phi_edges[i+1]
        # Select hits in this phi section
        phi_hits = hits[(hits.phi > phi_min) & (hits.phi < phi_max)]
        # Center these hits on phi=0
        centered_phi = phi_hits.phi - (phi_min + phi_max) / 2
        phi_hits = phi_hits.assign(phi=centered_phi, phi_section=i)
        for j in range(len(eta_edges) - 1):
            eta_min, eta_max = eta_edges[j], eta_edges[j+1]
            # Select hits in this eta section
            eta = calc_eta(phi_hits.r, phi_hits.z)
            sec_hits = phi_hits[(eta > eta_min) & (eta < eta_max)]
            hits_sections.append(sec_hits.assign(eta_section=j))
    return hits_sections

