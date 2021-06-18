# System imports
import os
import sys
import logging

# External imports
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.checkpoint import checkpoint

# Pick up local packages
sys.path.append("..")

# Local imports
from lightning_modules.GNNEmbedding.Models.interaction_gnn import (
    InteractionEdgeEmbedding,
    GlobalInteractionNodeEmbedding,
)
from lightning_modules.GNNEmbedding.Models.agnn import (
    GlobalAttentionNodeEmbedding,
    ConcatAttentionNodeEmbedding,
    ConcatPlusAttentionNodeEmbedding,
)
from lightning_modules.GNNEmbedding.Models.agnn import AttentionNodeEmbedding
from lightning_modules.GNNEmbedding.Models.agnn import LocalAttentionNodeEmbedding
from lightning_modules.GNN.Models.agnn import ResAGNN
from lightning_modules.Filter.Models.vanilla_filter import VanillaFilter
from pytorch_lightning.loggers import WandbLogger

logging.basicConfig(level=logging.INFO)


def main():

    with open("../lightning_modules/GNNEmbedding/train_jet_gnn.yaml") as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    model = LocalAttentionNodeEmbedding(hparams)
    wandb_logger = WandbLogger(project="End2End-ConnectedJetNodeEmbedding")
    wandb_logger.watch(model)
    wandb_logger.log_hyperparams({"model": type(model)})
    trainer = Trainer(
        gpus=1,
        max_epochs=hparams["max_epochs"],
        logger=wandb_logger,
        num_sanity_val_steps=0,
        accumulate_grad_batches=1,
    )

    trainer.fit(model)


if __name__ == "__main__":

    main()
