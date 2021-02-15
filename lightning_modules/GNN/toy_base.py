import sys, os
import logging

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from datetime import timedelta
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.nn import Linear
import torch

from ..utils import load_processed_dataset

def calc_eta(r, z):
    theta = torch.atan(r/z)
    return -1. * torch.log(torch.tan(theta / 2.))

class GNNBase(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        '''
        Initialise the Lightning Module that can scan over different GNN training regimes
        '''
        # Assign hyperparameters
        self.hparams = hparams
        self.hparams["posted_alert"] = False
        
    def setup(self, stage):
        if stage == "fit":
            # Handle any subset of [train, val, test] data split, assuming that ordering
            input_dirs = [None, None, None]
            input_dirs[:len(self.hparams["datatype_names"])] = [os.path.join(self.hparams["input_dir"], datatype) for datatype in self.hparams["datatype_names"]]
            self.trainset, self.valset, self.testset = [load_processed_dataset(input_dir, self.hparams["datatype_split"][i]) for i, input_dir in enumerate(input_dirs)]
        
    def train_dataloader(self):
        if self.trainset is not None:
            return DataLoader(self.trainset, batch_size=1, num_workers=1)
        else:
            return None

    def val_dataloader(self):
        if self.valset is not None:
            return DataLoader(self.valset, batch_size=1, num_workers=1)
        else:
            return None

    def test_dataloader(self):
        if self.testset is not None:
            return DataLoader(self.testset, batch_size=1, num_workers=1)
        else:
            return None
        
    def configure_optimizers(self):
        optimizer = [torch.optim.AdamW(self.parameters(), lr=(self.hparams["lr"]), betas=(0.9, 0.999), eps=1e-08, amsgrad=True)]
#         scheduler = [
#             {
#                 'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer[0], factor=self.hparams["factor"], patience=self.hparams["patience"]),
#                 'monitor': 'val_loss',
#                 'interval': 'epoch',
#                 'frequency': 1
#             }
#         ]
        scheduler = [
            {
                'scheduler': torch.optim.lr_scheduler.StepLR(optimizer[0], step_size=self.hparams["patience"], gamma=self.hparams["factor"]),
                'interval': 'epoch',
                'frequency': 1
            }
        ]
        return optimizer, scheduler

    def random_sample(self, batch):
        
        if "bidirectional" in self.hparams:
            bidir_edges = torch.cat([batch.edge_index, batch.edge_index.flip(0)], axis=-1)
        
        else:
            bidir_edges = batch.edge_index
                        
        if "hnm" in self.hparams["regime"]:
            
            bidir_edges = self.find_hard_negatives(bidir_edges, batch)
        
        if "subgraph" in self.hparams["regime"] and batch.sub_edge_index.sum() > 1000:
            
            subgraph_indices = batch.sub_edge_index
            
        elif "eta_slice" in self.hparams["regime"]:
            
            eta = calc_eta(batch.x[:, 0], batch.x[:, 2])
            eta_av = (eta[bidir_edges[0]] + eta[bidir_edges[1]]) / 2
            bidir_edges = bidir_edges[:, eta_av.argsort()]
            
            rand_index = torch.randint(bidir_edges.shape[1] - self.hparams["n_edges"], (1,)).item()
            subgraph_indices = torch.arange(rand_index, (rand_index + self.hparams["n_edges"]))
#             bidir_edges = bidir_edges[:, rand_index:(rand_index + self.hparams["n_edges"])]
            
        elif 'n_edges' in self.hparams:
            
            subgraph_indices = torch.randperm(bidir_edges.shape[1])[:self.hparams["n_edges"]]
        
        if "balanced" in self.hparams["regime"]:
            
            y = batch.pid[bidir_edges[0]] == batch.pid[bidir_edges[1]]
            
            num_true, num_false = y.bool().sum(), (~y.bool()).sum()
            fake_indices = torch.where(~y.bool())[0][torch.randperm(num_false)[:num_true]]
            true_indices = torch.where(y.bool())[0]
            combined_indices = torch.cat([true_indices, fake_indices])

            # Shuffle indices:
            bidir_edges = bidir_edges[:, combined_indices[torch.randperm(len(combined_indices))]]
    
        return bidir_edges, subgraph_indices
    
    def training_step(self, batch, batch_idx):
       
        input_edges, loss_indices = self.random_sample(batch)
            
        output = self(batch.x, input_edges).squeeze()
#         print(output.shape)
        
        y_pid = (batch.pid[input_edges[0, loss_indices]] == batch.pid[input_edges[1, loss_indices]]).float()
#         print(y_pid.shape)
        loss = F.binary_cross_entropy_with_logits(output[loss_indices], y_pid.float(), pos_weight = torch.tensor(self.hparams["weight"]))
            
        self.log('train_loss', loss)

        return loss

    def shared_evaluation(self, batch, batch_idx):

        weight = (torch.tensor(self.hparams["weight"]) if ("weight" in self.hparams)
                      else torch.tensor((~batch.y_pid.bool()).sum() / batch.y_pid.sum()))

        output = (self(torch.cat([batch.cell_data, batch.x], axis=-1), batch.edge_index).squeeze()
                  if ('ci' in self.hparams["regime"])
                  else self(batch.x, batch.edge_index).squeeze())
        
        truth = (batch.pid[batch.edge_index[0]] == batch.pid[batch.edge_index[1]]).float() if 'pid' in self.hparams["regime"] else batch.y

        if 'weighting' in self.hparams['regime']:
            manual_weights = batch.weights
        else:
            manual_weights = None
            
#         loss = F.binary_cross_entropy_with_logits(output, truth.float(), weight = manual_weights, pos_weight = weight)
        loss = F.binary_cross_entropy_with_logits(output, truth.float())

        #Edge filter performance
        preds = F.sigmoid(output) > self.hparams["edge_cut"]
        edge_positive = preds.sum().float()

        edge_true = truth.sum().float()
        edge_true_positive = (truth.bool() & preds).sum().float()
    
        eff = edge_true_positive/edge_true
        pur = edge_true_positive/max(edge_positive,1)
        
        if (eff > 0.99) and (pur > 0.99) and not self.hparams["posted_alert"] and self.hparams["slack_alert"]:
            self.logger.experiment.alert(title="High Performance", 
                        text="Efficiency and purity have both cracked 99%. Great job, Dan! You're having a great Thursday, and I think you've earned a celebratory beer.",
                        wait_duration=timedelta(minutes=60))
            self.hparams["posted_alert"] = True
        
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log_dict({'val_loss': loss, 'edge_eff': eff, 'edge_pur': pur, "current_lr": current_lr})
    
        return {"loss": loss, "true_positive": (truth.bool() & preds).float().cpu().numpy(), "true": truth.float().cpu().numpy(), "positive": preds.float().cpu().numpy()}

    def validation_step(self, batch, batch_idx):
        
        outputs = self.shared_evaluation(batch, batch_idx)
            
        return outputs["loss"]
    
    def test_step(self, batch, batch_idx):

        outputs = self.shared_evaluation(batch, batch_idx)
        
        return outputs
    
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # warm up lr
        if (self.hparams["warmup"] is not None) and (self.trainer.global_step < self.hparams["warmup"]):
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.hparams["warmup"])
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams["lr"]
        
        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
