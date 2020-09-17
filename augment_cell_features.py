"""
Data preparation script for GNN tracking.

This script processes the TrackML dataset and produces graph data on disk.
"""

# System
import os
import sys
import argparse
import logging
import multiprocessing as mp
from functools import partial

# Externals
import yaml
import numpy as np
import pandas as pd
import trackml.dataset

sys.path.append('../exatrkx-work/graph_building/src')
from preprocess_with_dir.preprocess import *

import torch
from torch_geometric.data import Data

from itertools import permutations
import itertools

def prepare_event(event, cell_features, output_dir, detector_orig, detector_proc):
    
    data = torch.load(event)
    
    event_file = data.event_file
    evtid = event_file[-4:]
    print("Augmenting", evtid)
            
    hits, truth = get_one_event(event_file,
                  detector_orig,
                  detector_proc,
                  remove_endcaps=True,
                  remove_noise=True,
                  pt_cut=0)
    
    hid = pd.DataFrame(data.hid.numpy(), columns = ["hit_id"])
    cell_data = torch.from_numpy((hid.merge(hits, on="hit_id")[cell_features]).to_numpy()).float()
    data.cell_data = cell_data
    
    filename = os.path.join(output_dir, str(evtid))
    print('Event', evtid, 'writing graphs to', filename)
    with open(filename, 'wb') as pickle_file:
        torch.save(data, pickle_file)

def main():
    """Main function"""

    pt_cut = 0
    save_dir = "/global/cscratch1/sd/danieltm/ExaTrkX/trackml_processed/truth_processed"
    basename = os.path.join(save_dir, str(pt_cut) + "_pt_cut")
    load_path = os.path.join(basename, "all_events")
    all_events = os.listdir(load_path)
    all_events = sorted([os.path.join(load_path, event) for event in all_events])
    
    cell_features = ['cell_count', 'cell_val',
                     'leta', 'lphi', 'lx', 'ly', 'lz', 'geta', 'gphi']
    detector_path = "/global/cscratch1/sd/danieltm/ExaTrkX/trackml/detectors.csv"
    detector_orig, detector_proc = load_detector(detector_path)
    
    logging.info('Writing outputs to ' + load_path)

    # Process input files with a worker pool
    with mp.Pool(processes=32) as pool:
        process_func = partial(prepare_event, cell_features=cell_features, output_dir=load_path, detector_orig=detector_orig, detector_proc=detector_proc)
        pool.map(process_func, all_events)

    logging.info('All done!')

if __name__ == '__main__':
    main()
