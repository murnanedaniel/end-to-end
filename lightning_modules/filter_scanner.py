import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch.nn import Linear
import sys

# Local imports
sys.path.append('..')
from prepare_utils import *
from performance_utils import *
from toy_utils import *
from models import *
from trainers import *

class Filter(torch.nn.Module):
    def __init__(self, in_channels, emb_channels, hidden, nb_layer):
        super(Filter, self).__init__()
        self.input_layer = Linear(in_channels*2 + emb_channels*2, hidden)
        layers = [Linear(hidden, hidden) for _ in range(nb_layer-1)]
        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden, 1)
        self.norm = nn.LayerNorm(hidden)
        self.act = nn.Tanh()

    def forward(self, x, e, emb=None):
        if emb is not None:
            x = self.input_layer(torch.cat([x[e[0]], emb[e[0]], x[e[1]], emb[e[1]]], dim=-1))
        else:
            x = self.input_layer(torch.cat([x[e[0]], x[e[1]]], dim=-1))
        for l in self.layers:
            x = l(x)
            x = self.act(x)
#         x = self.norm(x) #Option of LayerNorm
        x = self.output_layer(x)
        return x
    
class Filter_Model(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        '''
        Initialise the Lightning Module that can scan over different embedding training regimes
        '''
        # Assign hyperparameters
        self.hparams = hparams

        # Construct the MLP architecture      
        self.input_layer = Linear(hparams["in_channels"]*2 + hparams["emb_channels"]*2, hparams["hidden"])
        layers = [Linear(hparams["hidden"], hparams["hidden"]) for _ in range(hparams["nb_layer"]-1)]
        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hparams["hidden"], 1)
        self.layernorm = nn.LayerNorm(hparams["hidden"])
        self.batchnorm = nn.BatchNorm1d(num_features=hparams["hidden"], track_running_stats=False)
        self.act = nn.Tanh()

    def forward(self, x, e, emb=None):
        if emb is not None:
            x = self.input_layer(torch.cat([x[e[0]], emb[e[0]], x[e[1]], emb[e[1]]], dim=-1))
        else:
            x = self.input_layer(torch.cat([x[e[0]], x[e[1]]], dim=-1))
        for l in self.layers:
            x = l(x)
            x = self.act(x)
            if self.hparams["layernorm"]: x = self.layernorm(x) #Option of LayerNorm
            if self.hparams["batchnorm"]: x = self.batchnorm(x) #Option of Batch
        x = self.output_layer(x)
        return x
    
    def configure_optimizers(self):
        optimizer = [torch.optim.AdamW(self.parameters(), lr=(self.hparams["lr"]), betas=(0.9, 0.999), eps=1e-08, amsgrad=True)]
        scheduler = [
            {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer[0], factor=self.hparams["factor"], patience=self.hparams["patience"]),
                'monitor': 'checkpoint_on',
                'interval': 'epoch',
                'frequency': 1
            }
        ]
#         scheduler = [torch.optim.lr_scheduler.StepLR(optimizer[0], step_size=1, gamma=0.3)]
        return optimizer, scheduler
        
    def training_step(self, batch, batch_idx):
        
        emb = (None if (self.hparams["emb_channels"] == 0) 
               else batch.embedding)  # Does this work??      
        
        if self.hparams['ratio'] != 0:
            num_true, num_false = batch.y.bool().sum(), (~batch.y.bool()).sum()
            fake_indices = torch.where(~batch.y.bool())[0][torch.randint(num_false, (num_true.item()*hparams['ratio'],))]
            true_indices = torch.where(batch.y.bool())[0]
            combined_indices = torch.cat([true_indices, fake_indices])
            # Shuffle indices:
            combined_indices[torch.randperm(len(combined_indices))]
            weight = torch.tensor(self.hparams['ratio'])
        
        else:
            combined_indices = torch.range(batch.e_radius.shape[1])
            weight = (torch.tensor((~batch.y.bool()).sum() / batch.y.sum()) if (self.hparams["weight"]==None) 
                      else torch.tensor(self.hparams["weight"]))       
        
        output = (self(torch.cat([batch.cell_data, batch.x], axis=-1), batch.e_radius[:,combined_indices], emb).squeeze() 
                  if ('ci' in self.hparams["regime"]) 
                  else self(batch.x, batch.e_radius[:,combined_indices], emb).squeeze())
        
        if ('pid' in self.hparams["regime"]):
            y_pid = batch.pid[batch.e_radius[0,combined_indices]] == batch.pid[batch.e_radius[1,combined_indices]]
            loss = F.binary_cross_entropy_with_logits(output, y_pid.float(), pos_weight = weight) 
        else:
            loss = F.binary_cross_entropy_with_logits(output, batch.y[combined_indices], pos_weight = weight)
        
        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss, prog_bar=True)
        
        return result
        
    def validation_step(self, batch, batch_idx):
        
        emb = (None if (self.hparams["emb_channels"] == 0) 
               else batch.embedding)  # Does this work??   
        
        subset_ind = torch.randint(batch.e_radius.shape[1], (int(batch.e_radius.shape[1]*self.hparams['val_subset']),))
        
        output = self(torch.cat([batch.cell_data, batch.x], axis=-1), batch.e_radius[:, subset_ind], emb).squeeze() if ('ci' in self.hparams["regime"]) else self(batch.x, batch.e_radius[:, subset_ind], emb).squeeze()
        
        val_loss = F.binary_cross_entropy_with_logits(output, batch.y[subset_ind])
        
        result = pl.EvalResult(checkpoint_on=val_loss)
        result.log('val_loss', val_loss)
        
        #Edge filter performance
        preds = F.sigmoid(output) > 0.5 #Maybe send to CPU??
        edge_positive = preds.sum().float()
        if ('pid' in self.hparams["regime"]):
            y_pid = batch.pid[batch.e_radius[0,subset_ind]] == batch.pid[batch.e_radius[1,subset_ind]]
            edge_true = y_pid.sum()
            edge_true_positive = (y_pid & preds).sum().float()
        else:
            edge_true = batch.y[subset_ind].sum()
            edge_true_positive = (batch.y[subset_ind].bool() & preds).sum().float()
        
        
        result.log_dict({'eff': torch.tensor(edge_true_positive/edge_true), 'pur': torch.tensor(edge_true_positive/edge_positive)})        
        return result
    
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # warm up lr
        if (self.hparams["warmup"] is not None) and (self.trainer.global_step < self.hparams["warmup"]):
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.hparams["warmup"])
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams["lr"]

        # update params
        optimizer.step()
        optimizer.zero_grad()