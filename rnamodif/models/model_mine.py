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


class MyModel(pl.LightningModule):
    def __init__(
            self,
            lr=1e-3,
            warmup_steps=1000,
            wd=0.01,
            logging_steps=1):

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
                         batch_first=True, bidirectional=True, dropout=0.2),
            RNNPooling(),
            torch.nn.Linear(1*(2*3*32), 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 1),

        )


        self.acc = torchmetrics.classification.Accuracy(task="binary")
        self.ce = torch.nn.BCEWithLogitsLoss()
            
        self.training_step_counter = 0
        self.cumulative_loss = 0
        self.logging_steps = logging_steps
        self.cumulative_acc = 0


    def forward(self, x):
        out = self.head(x)
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
        x, y, exp = train_batch
        loss, preds = self.compute_loss(x, y, exp, 'train', return_preds=True)
        self.log('train_loss', loss, on_epoch=True)
        sch = self.lr_schedulers()
        sch.step()
            
        self.training_step_counter += 1
        self.cumulative_loss += loss.item()
        self.cumulative_acc +=self.acc(torch.sigmoid(preds), y)

        if self.training_step_counter % self.logging_steps == 0:
            avg_loss = self.cumulative_loss / self.logging_steps
            avg_acc = self.cumulative_acc / self.logging_steps
            
            self.log('train_loss_cum', avg_loss, on_step=True, on_epoch=False)
            self.log('train_acc_cum', avg_acc, on_step=True, on_epoch=False)
            self.cumulative_acc = 0
            self.cumulative_loss = 0
            
        return loss

    def validation_step(self, val_batch, batch_idx, dataloader_idx=None):
        x, y, identifier = val_batch
        exp = identifier['exp']
        loss, preds = self.compute_loss(x, y, exp, 'valid', return_preds=True)
        self.log('valid_loss', loss, on_epoch=True)
        return {'preds': preds, 'identifier': identifier}

    def predict_step(self, batch, batch_idx):
        xs, ids = batch
        logits = self.forward(xs)
        res = torch.sigmoid(logits)
        return res, ids

    def compute_loss(self, x, y, exp, loss_type, return_preds=False):
        seq_predictions_logits = self(x)
        mod_loss = self.ce(seq_predictions_logits, y)

        is_predicted_modified = torch.sigmoid(seq_predictions_logits)
        acc = self.acc(is_predicted_modified, y)
        # log overall accuracy
        self.log(f'{loss_type} acc', acc, on_epoch=True)

        # log accuracy for all unique experiments in the batch separately
        exps = np.array(exp)
        for e in np.unique(exp):
            indicies = exps == e
            batch_size = sum(indicies)
            if (batch_size > 0):
                self.log(f'{loss_type} {e} acc', self.acc(
                    is_predicted_modified[exps == e], y[exps == e]), on_epoch=True)

        if (return_preds):
            return mod_loss, is_predicted_modified
        return mod_loss

    def validation_epoch_end(self, outputs):
        # Aggregate all validation predictions into auroc metrics
        read_to_preds = {}
        read_to_label = {}
        read_to_exp = {}
        for log in outputs:
            preds = log['preds'].cpu().numpy()
            ids = log['identifier']

            # TODO Detaching slow?
            for i, (readid, pred, label, exp) in enumerate(zip(ids['readid'], preds, ids['label'].detach().cpu(), ids['exp'])):
                if (readid not in read_to_preds.keys()):
                    read_to_preds[readid] = []
                    read_to_label[readid] = label
                    read_to_exp[readid] = exp
                read_to_preds[readid].append(pred)

        read_to_preds_meanpool = {}
        read_to_preds_maxpool = {}
        for k, v in read_to_preds.items():
            read_to_preds_meanpool[k] = np.array(v).mean()
            read_to_preds_maxpool[k] = np.array(v).max()
        
        for tup in self.trainer.datamodule.valid_auroc_tuples:
            pos = tup[0]
            neg = tup[1]
            name = tup[2]
            mean = get_auroc_score([pos,neg], read_to_exp, read_to_preds_meanpool, read_to_label)
            maximum = get_auroc_score([pos,neg], read_to_exp, read_to_preds_maxpool, read_to_label)
            self.log(f'{name} auroc MEAN', mean)
            self.log(f'{name} auroc MAX', maximum)
            
def get_auroc_score(exps, read_to_exp, read_to_preds, read_to_label):
    keys = [k for k, exp in read_to_exp.items() if exp in exps]
    filtered_labels = np.array([read_to_label[k] for k in keys])
    filtered_preds = np.array([read_to_preds[k] for k in keys])
    try:
        auroc = roc_auc_score(filtered_labels, filtered_preds)
    except ValueError as error:
        print(error)
        auroc = -0.01

    return auroc

