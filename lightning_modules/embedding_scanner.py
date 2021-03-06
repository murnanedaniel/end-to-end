import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch.nn import Linear
import sys

# Local imports
sys.path.append("..")
from prepare_utils import *
from performance_utils import *
from toy_utils import *
from models import *
from trainers import *


class Embedding_Model(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        # Assign hyperparameters
        self.hparams = hparams

        # Construct the MLP architecture
        layers = [Linear(hparams["in_channels"], hparams["emb_hidden"])]
        ln = [
            Linear(hparams["emb_hidden"], hparams["emb_hidden"])
            for _ in range(hparams["nb_layer"] - 1)
        ]
        layers.extend(ln)
        self.layers = nn.ModuleList(layers)
        self.emb_layer = nn.Linear(hparams["emb_hidden"], hparams["emb_dim"])
        self.norm = nn.LayerNorm(hparams["emb_hidden"])
        self.act = nn.Tanh()

    def forward(self, x):
        #         hits = self.normalize(hits)
        for l in self.layers:
            x = l(x)
            x = self.act(x)
        #         x = self.norm(x) #Option of LayerNorm
        x = self.emb_layer(x)
        return x

    def configure_optimizers(self):
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=(self.hparams["lr"]),
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True,
            )
        ]
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer[0],
                    factor=self.hparams["factor"],
                    patience=self.hparams["patience"],
                ),
                "monitor": "checkpoint_on",
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        #         scheduler = [torch.optim.lr_scheduler.StepLR(optimizer[0], step_size=1, gamma=0.3)]
        return optimizer, scheduler

    def training_step(self, batch, batch_idx):

        if "ci" in self.hparams["regime"]:
            spatial = self(torch.cat([batch.cell_data, batch.x], axis=-1))
        else:
            spatial = self(batch.x)

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

        e_spatial = torch.empty([2, 0], dtype=torch.int64, device=self.device)

        if "rp" in self.hparams["regime"]:
            # Get random edge list
            n_random = int(self.hparams["randomisation"] * e_bidir.shape[1])
            e_spatial = torch.cat(
                [
                    e_spatial,
                    torch.randint(
                        e_bidir.min(), e_bidir.max(), (2, n_random), device=self.device
                    ),
                ],
                axis=-1,
            )

        if "hnm" in self.hparams["regime"]:
            e_spatial = torch.cat(
                [
                    e_spatial,
                    build_edges(
                        spatial, self.hparams["r_train"], self.hparams["knn"], res
                    ),
                ],
                axis=-1,
            )

        e_spatial, y_cluster = graph_intersection(e_spatial, e_bidir)

        e_spatial = torch.cat(
            [
                e_spatial,
                e_bidir.transpose(0, 1)
                .repeat(1, self.hparams["weight"])
                .view(-1, 2)
                .transpose(0, 1),
            ],
            axis=-1,
        )
        y_cluster = np.concatenate(
            [y_cluster.astype(int), np.ones(e_bidir.shape[1] * self.hparams["weight"])]
        )

        hinge = torch.from_numpy(y_cluster).float().to(device)
        hinge[hinge == 0] = -1

        reference = spatial.index_select(0, e_spatial[1])
        neighbors = spatial.index_select(0, e_spatial[0])
        d = torch.sum((reference - neighbors) ** 2, dim=-1)

        loss = torch.nn.functional.hinge_embedding_loss(
            d, hinge, margin=self.hparams["margin"], reduction="mean"
        )

        result = pl.TrainResult(minimize=loss)
        result.log("train_loss", loss, prog_bar=True)

        return result

    def validation_step(self, batch, batch_idx):

        if "ci" in self.hparams["regime"]:
            spatial = self(torch.cat([batch.cell_data, batch.x], axis=-1))
        else:
            spatial = self(batch.x)

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
        e_spatial = build_edges(spatial, self.hparams["r_val"], 1000, res)

        e_spatial, y_cluster = graph_intersection(e_spatial, e_bidir)

        hinge = torch.from_numpy(y_cluster).float().to(device)
        hinge[hinge == 0] = -1

        reference = spatial.index_select(0, e_spatial[1])
        neighbors = spatial.index_select(0, e_spatial[0])
        d = torch.sum((reference - neighbors) ** 2, dim=-1)

        val_loss = torch.nn.functional.hinge_embedding_loss(
            d, hinge, margin=self.hparams["margin"], reduction="mean"
        )

        result = pl.EvalResult(checkpoint_on=val_loss)
        result.log("val_loss", val_loss)

        cluster_true = 2 * len(batch.layerless_true_edges[0])
        cluster_true_positive = y_cluster.sum()
        cluster_positive = len(e_spatial[0])

        result.log_dict(
            {
                "eff": torch.tensor(cluster_true_positive / cluster_true),
                "pur": torch.tensor(cluster_true_positive / cluster_positive),
            }
        )

        return result

    def optimizer_step(
        self,
        current_epoch,
        batch_nb,
        optimizer,
        optimizer_idx,
        second_order_closure=None,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # warm up lr
        if (self.hparams["warmup"] is not None) and (
            self.trainer.global_step < self.hparams["warmup"]
        ):
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["lr"]

        # update params
        optimizer.step()
        optimizer.zero_grad()
