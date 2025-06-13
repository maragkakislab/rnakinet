import torch
import torchmetrics
import pytorch_lightning as pl
from types import SimpleNamespace
import numpy as np
import re
from sklearn.metrics import roc_auc_score
from collections import defaultdict


class Permute(torch.nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 1)


class RNNPooling(torch.nn.Module):
    def forward(self, x):
        output = x[0]  # All states
        h_n = x[1]  # Last hidden states
        max_pooled, _ = torch.max(output, dim=1)
        mean_pooled = torch.mean(output, dim=1)
        last_states = h_n.permute(1, 0, 2).flatten(start_dim=1)
        concat = torch.cat(
            [max_pooled, mean_pooled, last_states], dim=1)
        return concat


class RNAkinet(pl.LightningModule):
    def __init__(
            self,
            lr=1e-3,
            warmup_steps=1000,
            wd=0.01,
            logging_steps=1,
    ):

        super().__init__()
        self.lr = lr
        self.wd = wd
        self.warmup_steps=warmup_steps

        self.head = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3),
            torch.nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3),
            torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3),
            torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3),
            torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3),
            
            
            Permute(),
            torch.nn.GRU(input_size=128, hidden_size=32, num_layers=1,
                         batch_first=True, bidirectional=True, dropout=0.0),
            RNNPooling(),
            torch.nn.Linear(1*(2*3*32), 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 1),

        )

        self.acc = torchmetrics.classification.Accuracy(task="binary")
        self.ce = torch.nn.BCELoss()
            
        self.training_step_counter = 0
        self.cumulative_loss = 0
        self.logging_steps = logging_steps
        self.cumulative_acc = 0


    def forward(self, x):
        out = self.head(x)
        out = torch.sigmoid(out)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.head.parameters(), 
            lr=self.lr, 
            weight_decay=self.wd
        )
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            total_iters=self.warmup_steps
        )
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx, dataloader_idx=None):
        x, y, _ = train_batch #exp info not used
        loss, preds = self.compute_loss(x, y, 'train', return_preds=True)
        self.log('train_loss', loss, on_epoch=True)
        sch = self.lr_schedulers()
        sch.step()
            
        self.training_step_counter += 1
        self.cumulative_loss += loss.item()
        self.cumulative_acc +=self.acc(preds, y)

        if self.training_step_counter % self.logging_steps == 0:
            avg_loss = self.cumulative_loss / self.logging_steps
            avg_acc = self.cumulative_acc / self.logging_steps
            
            self.log('train_loss_cum', avg_loss, on_step=True, on_epoch=False)
            self.log('train_acc_cum', avg_acc, on_step=True, on_epoch=False)
            self.cumulative_acc = 0
            self.cumulative_loss = 0
            
        return loss

    def validation_step(self, val_batch, batch_idx, dataloader_idx=None):
        x, y = val_batch
        loss, preds = self.compute_loss(x, y, 'valid', return_preds=True)
        self.log('valid_loss', loss, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        xs, ids = batch
        res = self.forward(xs)
        return res, ids

    def compute_loss(self, x, y, loss_type, return_preds=False):
        is_predicted_modified = self(x)
        mod_loss = self.ce(is_predicted_modified, y)

        acc = self.acc(is_predicted_modified, y)
        # log overall accuracy
        self.log(f'{loss_type} acc', acc, on_epoch=True)

        if (return_preds):
            return mod_loss, is_predicted_modified
        return mod_loss


