from RODAN.basecall import load_model
import torch
import torchmetrics
import pytorch_lightning as pl
from torch.nn import functional as F
from types import SimpleNamespace
from rnamodif.architectures.bonito_pretrained import RNNPooler
from bonito_pulled.bonito.nn import Permute
import numpy as np

class RodanPretrained(pl.LightningModule):
    def __init__(self, pretrained_lr=5e-4, my_layers_lr=2e-3, warmup_steps = 10000):
        super().__init__()

        #TODO fix module importing without hacking imports
        #TODO vocab ATCG - but rna is AUCG
        
        torchdict = torch.load('/home/jovyan/RNAModif/RODAN/rna.torch', map_location="cpu")
        # torchdict = torch.load('./RODAN/rna.torch', map_location="cpu")
        origconfig = torchdict["config"]
        d = origconfig
        n = SimpleNamespace(**d)
        args = {
            'debug':False, #True prints out more info
            'arch':None,
        }
        model, device = load_model('/home/jovyan/RNAModif/RODAN/rna.torch', config=n, args=SimpleNamespace(**args))

        #ORIGINAL
        # seq_model = torch.nn.Sequential(
        #     model,
        #     # RNNPooler(features_to_pool=5, seq_len=420),
        #     Permute((1,0,2)),
        #     torch.nn.Flatten(),
        #     torch.nn.Linear(420*5, 100),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(100,1),
        # )
        
        #CUT ACUGN CLASSIFIER layer
        model.final = torch.nn.Identity()
        seq_model = torch.nn.Sequential(
            model,
            torch.nn.Linear(768, 1),
            Permute((1,2,0)),
            torch.nn.MaxPool1d(420),
            torch.nn.Flatten(),
        )
        
        self.model = seq_model
        
        self.pretrained_layers_lr = pretrained_lr
        self.my_layers_lr = my_layers_lr
        self.warmup_steps = warmup_steps
        
        self.acc = torchmetrics.classification.BinaryAccuracy()
        self.cm = torchmetrics.classification.BinaryConfusionMatrix(normalize='true')
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        #different LR for my own layers (higher)
        my_layers_lr = self.my_layers_lr
        pretrained_layers_lr = self.pretrained_layers_lr
        
        pretrained_layers_count = 1 #one item in the seq_model definition
        custom_layers_count = len(list(self.model.children())) - pretrained_layers_count
        lr_list = [pretrained_layers_lr, *[my_layers_lr for _ in range(custom_layers_count)]] 
        groups = [{'params': list(m.parameters()), 'lr': lr} for (m, lr) in zip(self.model.children(), lr_list)]
        optimizer = torch.optim.AdamW(groups, lr=self.pretrained_layers_lr, weight_decay=0.01)
        
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=self.warmup_steps)
        return [optimizer], [scheduler]
    
    def training_step(self, train_batch, batch_idx, dataloader_idx=None):
        x,y,exp = train_batch
        output = self(x)
        loss = F.binary_cross_entropy_with_logits(output, y)
        self.log('train_loss', loss, on_epoch=True)
        self.log_metrics(output, y.int(), exp, 'train')

        sch = self.lr_schedulers()
        sch.step()
        current_lr_list = sch.get_last_lr()
        pretrained_lr = current_lr_list[0]
        my_layer_lr = current_lr_list[-1] #the same for all custom layers
        self.log('learning rate (pretrained part)', pretrained_lr)
        self.log('learning rate (custom part)', my_layer_lr)

        return loss
    
    def validation_step(self, val_batch, batch_idx, dataloader_idx=None):
        x,y,exp = val_batch
        output = self(x)
        loss = F.binary_cross_entropy_with_logits(output, y)
        self.log('valid_loss', loss, on_epoch=True) #ADDED on_epoch=True
        self.log_metrics(output, y.int(), exp, 'valid')
    
    def log_metrics(self, output, labels, exp, prefix):
        self.log_cm(output, labels, prefix)
        acc = self.acc(output, labels)
        self.log(f'{prefix} acc', acc, on_epoch=True)
        
        exps = np.array(exp)
        for e in np.unique(exp):
            indicies = exps == e
            batch_size = sum(indicies)
            if(batch_size>0):
                e_output = output[exps==e]
                e_labels = labels[exps==e]
                
                self.log_cm(e_output, e_labels, f'{e} {prefix}')
                
                e_acc = self.acc(e_output, e_labels)
                self.log(f'{e} {prefix} acc', e_acc, on_epoch=True, batch_size=batch_size)
        
        
        
    def log_cm(self, output, labels, prefix):
        
        cm = self.cm(output, labels)
        true_negatives_perc = cm[0][0]
        false_negatives_perc = cm[1][0]
        true_positives_perc = cm[1][1]
        false_positives_perc = cm[0][1]
        
        batch_size = len(output)
        if(true_positives_perc+false_negatives_perc > 0):
            self.log(f'{prefix} true_positives_perc', true_positives_perc, on_epoch=True, batch_size=batch_size)
            self.log(f'{prefix} false_negatives_perc', false_negatives_perc, on_epoch=True, batch_size=batch_size)
            
        if(true_negatives_perc+false_positives_perc > 0):
            self.log(f'{prefix} true_negatives_perc', true_negatives_perc, on_epoch=True, batch_size=batch_size)
            self.log(f'{prefix} false_positives_perc', false_positives_perc, on_epoch=True, batch_size=batch_size)
    