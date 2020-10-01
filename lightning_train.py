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

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger

from itertools import chain
import trackml.dataset

import ipywidgets as widgets
from ipywidgets import interact, interact_manual

# Pick up local packages
sys.path.append('..')

# Local imports
from prepare import select_hits
from toy_utils import *
from lightning_modules.embedding_scanner import Embedding_Model
# from models import *
# from trainers import *

# Get rid of RuntimeWarnings, gross
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import wandb

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('train.py')
    add_arg = parser.add_argument
    
    add_arg('config', nargs='?', default='configs/hello.yaml')
    add_arg('-r','--regime', nargs='+', help='Choose regime of [rp, hnm, ci]', default=None)
    add_arg('--hidden-dim', type=int, default=None,
            help='Hidden layer dimension size')
    add_arg('--n-graph-iters', type=int, default=None,
            help='Number of graph iterations')
    add_arg('--emb-dim', type=int, default=None,
            help='Number of spatial embedding dimensions')
    add_arg('--emb-hidden', type=int, default=None,
            help='Number of embedding hidden dimensions')
    add_arg('--nb-layer', type=int, default=None,
            help='Number of embedding layers')
    add_arg('--r-val', type=float, default=None,
            help='Radius of graph construction')
    add_arg('--r-train', type=float, default=None,
            help='Radius of embedding training')
    add_arg('--margin', type=float, default=None,
            help='Radius of hinge loss')
    add_arg('--weight', type=float, default=None,
            help='Positive weight in AGNN')
    add_arg('--randomisation', type=float, default=None,
            help='Ratio of RP to truth')
    add_arg('--warmup', type=float, default=None,
            help='Number of BATCHES to warmup upon')
    add_arg('--train-size', type=int, default=None,
            help='Number of train population')
    add_arg('--val-size', type=int, default=None,
            help='Number of validate population')
    add_arg('--pt-cut', type=float, default=None,
            help='Cutoff for momentum')
    add_arg('--adjacent', type=bool, default=False,
           help='Enforce adjacent layers?')
    
    return parser.parse_args()

def setup_wandb(config):
    
    hparams = config['hparams']
    wandb_logger = WandbLogger(project=config["project"], group='_'.join(hparams['regime']), log_model=True, save_dir = hparams['wandb_save_dir'])
    wandb_logger.log_hyperparams(hparams)
    wandb_logger.log_hyperparams(config['selection'])
    
    return wandb_logger

def get_loaders(config):
    
    # Dataset processing
    pt_cut = config['selection']['pt_min']
    train_number = config['selection']['train_number']
    test_number = config['selection']['test_number']
    load_dir = config['input_dir']
    
    # Construct experiment name
    group = str(pt_cut) + "_pt_cut"
    if config['selection']['endcaps']: group += "_endcaps"
    print("Running with dataset", group, "on device", device)
    
    train_path = os.path.join(load_dir, group, str(train_number) + "_events_train")
    test_path = os.path.join(load_dir, group, str(test_number) + "_events_test")

    train_dataset = torch.load(train_path)
    test_dataset = torch.load(test_path)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    return train_loader, test_loader
    
def main(args):

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    train_loader, test_loader = get_loaders(config)
    
    hparams = config['hparams']
    model = Embedding_Model(hparams)
    
    wandb_logger = setup_wandb(config)
    
    trainer = Trainer(gpus=1, limit_val_batches=0.1, max_epochs=hparams['max_epochs'], logger=wandb_logger, progress_bar_refresh_rate=100)
    
    trainer.fit(model, train_loader, test_loader)

        
if __name__=="__main__":
    
    # Parse the command line
    args = parse_args()
#     print(args)
    
    main(args)