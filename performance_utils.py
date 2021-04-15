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

# --------------------- Processing for performance analysis ---------------------------


def fetch_truths(events):
    loaded_truth = [
        trackml.dataset.load_event(event.event_file, parts=["truth"])[0]
        for event in events
    ]
    loaded_hits = [event.hid for event in events]
    merged_truth = [
        pd.DataFrame(hid.cpu().numpy(), columns=["hit_id"]).merge(truth, on="hit_id")
        for (hid, truth) in zip(loaded_hits, loaded_truth)
    ]
    pt = [np.sqrt(truth.tpx ** 2 + truth.tpy ** 2) for truth in merged_truth]

    return pt, merged_truth


# -------------------------Visualisation routines--------------------------------------
