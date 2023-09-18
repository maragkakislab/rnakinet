import torch
import torchmetrics
import pytorch_lightning as pl
from types import SimpleNamespace
import numpy as np
import re
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from rnamodif.models.modules import ConvNet, RNNEncoder, MLP, Attention, Permute
from sklearn.metrics import average_precision_score

class Small_CNN(pl.LightningModule):
    def __init__(
            self,
            pooling='max',
            lr=1e-3,
            warmup_steps=1000,
            wd=0.01,
            logging_steps=1,
            pos_weight=1.0,
    ):

        super().__init__()
        self.lr = lr
        self.wd = wd
        self.warmup_steps=warmup_steps

        if(pooling=='rnn'):
            self.architecture = torch.nn.Sequential(
                ConvNet(),
                Permute(),
                RNNEncoder(input_size=128, hidden_size=32, num_layers=1, dropout=0.2),
                MLP(input_size=2*3*32, hidden_size=30),
            )
        if(pooling=='att'):
            self.architecture = torch.nn.Sequential(
                ConvNet(),
                Permute(),
                Attention(input_dim=128, len_limit=400000),
                MLP(128, 30),
            )
        if(pooling=='max'):
            self.architecture = torch.nn.Sequential(
                ConvNet(),
                torch.nn.AdaptiveMaxPool1d(1),
                torch.nn.Flatten(),
                MLP(128, 30),
            )
        # if(pooling=='max2'):
        #     self.architecture = torch.nn.Sequential(
        #         ResConvNet(num_layers=3),
        #         torch.nn.AdaptiveMaxPool1d(1),
        #         torch.nn.Flatten(),
        #         MLP(128, 30),
        #     )
        # if(pooling=='max3'):
        #     depth = 6
        #     init_channels = 8
        #     self.architecture = torch.nn.Sequential(
        #         BigConvNet(num_layers=depth, num_channels_initial=init_channels),
        #         torch.nn.AdaptiveMaxPool1d(1),
        #         torch.nn.Flatten(),
        #         MLP(init_channels*(2**(depth-1)), 30),
        #     )

        self.acc = torchmetrics.classification.Accuracy(task="binary")
        self.ce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(self.device))
            
        # Logging utils to fix, TODO resolve with a custom callback?
        self.training_step_counter = 0
        self.cumulative_loss = 0
        self.cumulative_acc = 0
        self.logging_steps = logging_steps

    def forward(self, x):
        return self.architecture(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(), 
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
        logits = self(x)
        loss = self.ce(logits, y)
        
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        probs = self.logits_to_probs(logits)
        
        metrics = self.get_metrics(probs, y, exp)
        self.log_cumulative_train_metrics(loss, metrics['acc'])
        for metric, value in metrics.items():
            self.log(f'train {metric}', value, on_epoch=True, on_step=False)

        self.lr_schedulers().step()
        return loss
    
    def validation_step(self, val_batch, batch_idx, dataloader_idx=None):
        x, y, identifier = val_batch
        exp = identifier['exp']
        logits=self(x)
        loss = self.ce(logits, y)
        
        self.log('valid_loss', loss, on_epoch=True, on_step=False)
        probs = self.logits_to_probs(logits)
        metrics = self.get_metrics(probs, y, exp)
        for metric, value in metrics.items():
            self.log(f'valid {metric}', value, on_epoch=True, on_step=False)
            
        return {
            'preds': probs.detach().cpu().numpy(), 
            'identifier': {
                'readid':identifier['readid'],
                'label':identifier['label'].detach().cpu(),
                'exp':identifier['exp'],
            }
        }

    
    def predict_step(self, batch, batch_idx):
        xs, ids = batch
        logits = self.forward(xs)
        res = self.logits_to_probs(logits)
        return res, ids
    
    def get_metrics(self, predictions, labels, exps):
        metrics = {}
        metrics['acc']=self.acc(predictions, labels)
        exps = np.array(exps)
        for e in np.unique(exps):
            indices = exps == e
            if (sum(indices) > 0): #If Non-empty
                metrics[f'{e} acc'] = self.acc(predictions[exps == e], labels[exps == e])
        return metrics

    def log_cumulative_train_metrics(self, loss, accuracy):
        self.training_step_counter += 1
        self.cumulative_loss += loss.item()
        self.cumulative_acc += accuracy

        if self.training_step_counter % self.logging_steps == 0:
            avg_loss = self.cumulative_loss / self.logging_steps
            avg_acc = self.cumulative_acc / self.logging_steps
            
            self.log(f'train_loss_cum', avg_loss, on_step=True, on_epoch=False)
            self.log(f'train_acc_cum', avg_acc, on_step=True, on_epoch=False)
            self.cumulative_acc = 0
            self.cumulative_loss = 0
            
    def logits_to_probs(self, logits):
        return torch.sigmoid(logits)

    def validation_epoch_end(self, outputs):
        # Aggregate all validation predictions into auroc metrics
        read_to_pred, read_to_label, read_to_exp = self.aggregate_outputs(outputs)

        for tup in self.trainer.datamodule.valid_auroc_tuples:
            pos = tup[0]
            neg = tup[1]
            name = tup[2]
            auroc = get_auroc_score([pos,neg], read_to_exp, read_to_pred, read_to_label)
            self.log(f'{name} auroc', auroc)
            
            auprc = get_auprc_score([pos,neg], read_to_exp, read_to_pred, read_to_label)
            self.log(f'{name} auprc', auprc)

    def aggregate_outputs(self, outputs):
        read_to_preds = {}
        read_to_label = {}
        read_to_exp = {}
        for log in outputs:
            preds = log['preds']#.cpu().numpy()
            ids = log['identifier']

            for i, (readid, pred, label, exp) in enumerate(zip(ids['readid'], preds, ids['label'], ids['exp'])):
                read_to_label[readid] = label
                read_to_exp[readid] = exp
                #TODO remove for optimization
                assert len(pred) == 1 
                read_to_preds[readid] = pred[0]

        return read_to_preds, read_to_label, read_to_exp


def get_auroc_score(exps, read_to_exp, read_to_preds, read_to_label):
    keys = [k for k, exp in read_to_exp.items() if exp in exps]
    filtered_labels = np.array([read_to_label[k] for k in keys])
    filtered_preds = np.array([read_to_preds[k] for k in keys])
    if(len(filtered_preds)==0):
        return -0.01
    try:
        auroc = roc_auc_score(filtered_labels, filtered_preds)
    except ValueError as error:
        print(error)
        auroc = -0.01

    return auroc

def get_auprc_score(exps, read_to_exp, read_to_preds, read_to_label):
    keys = [k for k, exp in read_to_exp.items() if exp in exps]
    filtered_labels = np.array([read_to_label[k] for k in keys])
    filtered_preds = np.array([read_to_preds[k] for k in keys])
    if(len(filtered_preds)==0):
        return -0.01
    try:
        auprc = average_precision_score(filtered_labels, filtered_preds)
    except ValueError as error:
        print(error)
        auprc = -0.01

    return auprc

