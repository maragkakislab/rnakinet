import RODAN

import torch
import torchmetrics
import pytorch_lightning as pl
from types import SimpleNamespace
import numpy as np
import re
from sklearn.metrics import roc_auc_score
from RODAN.basecall import load_model
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


class RodanPretrainedUnlimited(pl.LightningModule):
    def __init__(
            self,
            lr=1e-3,
            warmup_steps=1000,
            wd=0.01,
            len_limit=400000,
            weighted_loss=False,
            logging_steps=1,
            frozen_layers=0,
            pos_weight=1.0,
    
    ):

        super().__init__()

        # Loading the RODAN backbone
        torchdict = torch.load(
            '/home/jovyan/RNAModif/RODAN/rna.torch', map_location="cpu")
        origconfig = torchdict["config"]
        args = {
            'debug': False,  # True prints out more info
            'arch': None,
        }

        # Architecture composition
        self.trainable_rodan, device = load_model(
            '/home/jovyan/RNAModif/RODAN/rna.torch', 
            config=SimpleNamespace(**origconfig), 
            args=SimpleNamespace(**args)
        )
#         self.head = torch.nn.Sequential(
#             Permute(),
#             torch.nn.GRU(input_size=768, hidden_size=32, num_layers=1,
#                          batch_first=True, bidirectional=True, dropout=0.1),
#             RNNPooling(),
#             torch.nn.Linear(1*(2*3*32), 30),
#             torch.nn.ReLU(),
#             torch.nn.Linear(30, 1),

#         )

        # self.head = torch.nn.Sequential(
        #     Permute(),
        #     SelfAttention(768,1, len_limit=len_limit),
        #     torch.nn.Flatten(),
        #     torch.nn.AdaptiveMaxPool1d(1),
        # )
        
        self.head = torch.nn.Sequential(
            Permute(),
            Attention(768, len_limit=len_limit),
            torch.nn.Linear(768,1)
        )
        
        # self.head = torch.nn.Sequential(
        #     Permute(),
        #     torch.nn.Linear(768,1),
        #     torch.nn.Flatten(),
        #     torch.nn.AdaptiveMaxPool1d(1),
        # )
        # print('RESETING PARAMS')
        # def weight_reset(m):
        #     for layer in m.children():
        #         if hasattr(layer, 'reset_parameters'):
        #             layer.reset_parameters()
                    
        # self.trainable_rodan.apply(weight_reset)
        
        
        if (frozen_layers > 0):
            # print('FREEZING', frozen_layers, 'layers')
            # print('FREEZING ALLLLLL', frozen_layers, 'layers')
            freeze_rodan(self, freeze_till=frozen_layers, verbose=0)
            # for param in self.trainable_rodan.parameters():
                # param.requires_grad = False
            # for name, layer in self.trainable_rodan.named_modules():
                # if ('drop' in name):
                    # layer.p = 0.0
        
        
        self.lr = lr
        self.wd = wd
        self.warmup_steps = warmup_steps

        self.acc = torchmetrics.classification.Accuracy(task="binary")
        if(weighted_loss):
            print('using weighted loss')
            self.ce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(self.device))
        else:
            self.ce = torch.nn.BCEWithLogitsLoss()
            
        self.training_step_counter = 0
        self.cumulative_loss = 0
        self.logging_steps = logging_steps
        self.cumulative_acc = 0
        

    def forward(self, x):
        feature_vector = self.trainable_rodan.convlayers(x)
        out = self.head(feature_vector)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=[
                {'params': self.trainable_rodan.parameters()},
                {'params': self.head.parameters()}
            ], 
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
            
            self.cumulative_loss = 0
            self.cumulative_acc = 0
        return loss

    def validation_step(self, val_batch, batch_idx, dataloader_idx=None):
        x, y, identifier = val_batch
        exp = identifier['exp']
        loss, preds = self.compute_loss(x, y, exp, 'valid', return_preds=True)
        self.log('valid_loss', loss, on_epoch=True)
        return {'preds':preds.detach().cpu().numpy(), 'label':y.detach().cpu().numpy(), 'exp':identifier['exp']}

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
        exp_to_preds = defaultdict(list)
        exp_to_label = defaultdict(list)
        p = np.zeros(len(outputs))
        l = np.zeros(len(outputs))
        for i,log in enumerate(outputs):
            exp = log['exp'][0]
            exp_to_preds[exp].append(log['preds'][0])
            exp_to_label[exp].append(log['label'][0])
        
        for tup in self.trainer.datamodule.valid_auroc_tuples:
            try:
                preds = exp_to_preds[tup[0]]+exp_to_preds[tup[1]]
                labels = exp_to_label[tup[0]]+exp_to_label[tup[1]]
                auroc = roc_auc_score(labels, preds)
                self.log(f'{tup[2]} auroc', auroc)
            except ValueError as error:
                print(error)
            



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





import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, max_len=400000):
        super(PositionalEncoding, self).__init__()
        self.input_dim = input_dim

        # Create the positional encoding matrix
        positional_encoding = torch.zeros(max_len, input_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_dim, 2).float() * (-math.log(10000.0) / input_dim))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)

        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        x = x + self.positional_encoding[:, :x.size(1), :]
        return x

class SelfAttention(nn.Module):
    def __init__(self, input_dim, attention_dim, len_limit):
        super(SelfAttention, self).__init__()
        self.attention_dim = attention_dim
        self.input_dim = input_dim

        self.positional_encoding = PositionalEncoding(input_dim, max_len = len_limit)
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        

    def forward(self, x):
        # Input x shape: (batch_size, sequence_length, input_dim)
        batch_size, sequence_length, _ = x.size()

        # Apply positional encoding
        x = self.positional_encoding(x)

        queries = self.query(x)  # (batch_size, sequence_length, attention_dim)
        keys = self.key(x)       # (batch_size, sequence_length, attention_dim)
        values = self.value(x)   # (batch_size, sequence_length, attention_dim)

        # Compute the attention scores
        attention_scores = torch.bmm(queries, keys.transpose(1, 2))  # (batch_size, sequence_length, sequence_length)
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.attention_dim, dtype=torch.float))
        attention_weights = F.softmax(attention_scores, dim=2)  # Normalize weights along the sequence_length dimension

        # Compute the self-attention output
        attention_output = torch.bmm(attention_weights, values)  # (batch_size, sequence_length, attention_dim)

        return attention_output

    
    
class Attention(nn.Module):
    def __init__(self, input_dim, len_limit):
        super(Attention, self).__init__()
        self.attention = nn.Linear(input_dim, 1)

        # Initialize the position encoding matrix
        pe_matrix = torch.zeros(len_limit, input_dim)

        # Compute the position indices
        position = torch.arange(0, len_limit).unsqueeze(1)

        # Compute the dimension indices
        div_term = torch.exp(torch.arange(0, input_dim, 2) * -(math.log(10000.0) / input_dim))

        # Compute the sinusoidal encoding
        pe_matrix[:, 0::2] = torch.sin(position * div_term)
        pe_matrix[:, 1::2] = torch.cos(position * div_term)

        # Register the position encoding matrix as a buffer to be saved with the model
        self.register_buffer('pe_matrix', pe_matrix)
    def forward(self, x):
        # Positional encoding
        x = x + self.pe_matrix[:x.size(1), :]
        # Compute attention scores
        attention_scores = self.attention(x).squeeze(-1)

        # Apply softmax to obtain attention weights
        attention_weights = F.softmax(attention_scores, dim=-1).unsqueeze(-1)

        # Apply attention weights to the input sequence
        context_vector = torch.sum(x * attention_weights, dim=-2)

        return context_vector
    
    
    
    
def freeze_rodan(model, freeze_till, verbose=1):
    # freeze_till max is 21
    for name, module in model.named_modules():
        if (name in ['', 'trainable_rodan', 'trainable_rodan.convlayers']):
            continue
        pattern = r"conv\d+"
        match = re.search(pattern, name)
        if (match):
            conv_index = int(match.group(0)[4:])
            if (conv_index > freeze_till):
                # print('breaking')
                break
        if ('drop' in name):
            # module.eval()
            module.p = 0.0

        for param in module.parameters():
            param.requires_grad = False

    if (verbose == 1):
        for name, module in model.named_modules():
            if (len(list(module.parameters())) > 0):
                print(
                    all([p.requires_grad for p in list(module.parameters())]), name)