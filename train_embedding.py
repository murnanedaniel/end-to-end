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
sys.path.append('..')

# Local imports
from prepare import select_hits
from toy_utils import *
from models import *
from trainers import *

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
    add_arg('--lr-1', type=float, default=None,
            help='Embedding loss learning rate')
    add_arg('--lr-2', type=float, default=None,
            help='AGNN loss learning rate')
    add_arg('--lr-3', type=float, default=None,
            help='Weight balance learning rate')
    add_arg('--weight', type=float, default=None,
            help='Positive weight in AGNN')
    add_arg('--train-size', type=int, default=None,
            help='Number of train population')
    add_arg('--val-size', type=int, default=None,
            help='Number of validate population')
    add_arg('--pt-cut', type=float, default=None,
            help='Cutoff for momentum')
    add_arg('--adjacent', type=bool, default=False,
           help='Enforce adjacent layers?')
    
    return parser.parse_args()
    

def save_model(epoch, model, optimizer, scheduler, running_loss, config, PATH):
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': running_loss,
                'config': config
                }, os.path.join('model_comparisons/', PATH))

def main(args):

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    # Dataset processing
    pt_cut = config['selection']['pt_min']
    train_number = config['selection']['train_number']
    test_number = config['selection']['test_number']
    load_dir = config['input_dir']
    
    # Construct experiment name
    group = str(pt_cut) + "_pt_cut"
    if config['selection']['endcaps']: group += "_endcaps"
    print("Running experiment group", group, "on device", device)
    
    train_path = os.path.join(load_dir, group, str(train_number) + "_events_train")
    test_path = os.path.join(load_dir, group, str(test_number) + "_events_test")

    train_dataset = torch.load(train_path)
    test_dataset = torch.load(test_path)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    # Model config
    
    m_configs = config['model']
    model = Embedding(**m_configs).to(device)
#     multi_loss = MultiNoiseLoss(n_losses=2).to(device)
    m_configs.update(config['training'])
    m_configs.update(config['selection'])
    wandb.init(project=config['project'], group=group, config=m_configs)
    wandb.run.save()
    print(wandb.run.name)
    model_name = wandb.run.name
    wandb.watch(model, log='all')
    
    # Optimizer & Scheduler config
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=m_configs['lr'], weight_decay=m_configs['weight_decay'], amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=m_configs['factor'], patience=m_configs['patience'])
    
    print("Training on configs:", m_configs)
    # Training loop
    for epoch in range(m_configs['epochs']):
        tic = tt() 
        model.train()
        train_loss = train_connected_emb(model, train_loader, optimizer, m_configs)
        print('Training loss: {:.4f}'.format(train_loss))

        model.eval()
        with torch.no_grad():
            cluster_pur, cluster_eff, val_loss = evaluate_connected_emb(model, test_loader, m_configs)
        wandb.log({"val_loss": val_loss, "train_loss": train_loss, "cluster_pur": cluster_pur, "cluster_eff": cluster_eff, "lr": optimizer.param_groups[0]['lr']})
        scheduler.step(val_loss)

        save_model_from_script(epoch, model, optimizer, scheduler, val_loss, m_configs, 'Embedding/'+model_name+'.tar')

        print('Epoch: {}, Eff: {:.4f}, Pur: {:.4f}, Loss: {:.4f}, LR: {} in time {}'.format(epoch, cluster_eff, cluster_pur, val_loss, optimizer.param_groups[0]['lr'], tt()-tic))

        
if __name__=="__main__":
    
    # Parse the command line
    args = parse_args()
#     print(args)
    
    main(args)