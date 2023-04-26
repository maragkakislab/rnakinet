from RODAN.basecall import load_model
import torch
import torchmetrics
import pytorch_lightning as pl
from torch.nn import functional as F
from types import SimpleNamespace
from rnamodif.architectures.bonito_pretrained import RNNPooler
from bonito_pulled.bonito.nn import Permute
import numpy as np

class RodanPretrainedSeqcallerFrozen(pl.LightningModule):
    def __init__(self, lr=1e-3, warmup_steps=1000):
        super().__init__()
        torchdict = torch.load('/home/jovyan/RNAModif/RODAN/rna.torch', map_location="cpu")
        origconfig = torchdict["config"]
        d = origconfig
        n = SimpleNamespace(**d)
        args = {
            'debug':False, #True prints out more info
            'arch':None,
        }
        self.trainable_rodan, device = load_model('/home/jovyan/RNAModif/RODAN/rna.torch', config=n, args=SimpleNamespace(**args))
        for par in self.trainable_rodan.parameters():
            par.requires_grad = False
        # self.mod_neuron = torch.nn.Linear(5,1)
        self.mod_neuron = torch.nn.Linear(768,1)
        
        self.lr = lr
        
        self.acc = torchmetrics.classification.BinaryAccuracy()
        self.auroc = torchmetrics.classification.AUROC(task='binary')
        # self.ce = torch.nn.CrossEntropyLoss()
        self.binary_ce = torch.nn.BCEWithLogitsLoss()
        self.warmup_steps = warmup_steps
        
    def forward(self, x):
        x = self.trainable_rodan.convlayers(x)
        x = x.permute(0,2,1)
        
        return self.mod_neuron(x)
        
        
        # x = self.trainable_rodan.final(x)
        # x = torch.sigmoid(x)
        # mod_pred = self.mod_neuron(x) #no need for sigmoid, i use logits BCE
        # return mod_pred
    
    
    def configure_optimizers(self):
        optimizer =  torch.optim.AdamW(params=[{'params':self.trainable_rodan.parameters()},{'params':self.mod_neuron.parameters()}], lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=self.warmup_steps)
        
        return [optimizer], [scheduler]
    
    def training_step(self, train_batch, batch_idx, dataloader_idx=None):
        x,y,exp = train_batch
        loss = self.compute_loss(x, y, exp,'train')
        self.log('train_loss', loss, on_epoch=True)
        
        sch = self.lr_schedulers()
        sch.step()
        current_lr_list = sch.get_last_lr()
        pretrained_lr = current_lr_list[0]
        my_layer_lr = current_lr_list[0] #The same
        self.log('learning rate (pretrained part)', pretrained_lr)
        self.log('learning rate (custom part)', my_layer_lr)
        
        return loss
    
    def validation_step(self, val_batch, batch_idx, dataloader_idx=None):
        x,y,exp = val_batch
        loss = self.compute_loss(x, y, exp, 'valid')
        self.log('valid_loss', loss, on_epoch=True) #ADDED on_epoch=True
        # self.log_metrics(output, y.int(), exp, 'valid')
    
    def predict_step(self, batch, batch_idx, expanded=False):
        xs, ids = batch
        _, mod_pred = self.forward(xs)
        is_predicted_modified = mod_pred.squeeze(dim=-1)
        #ADDED
        is_predicted_modified = torch.sigmoid(is_predicted_modified)
        if(expanded):
            return is_predicted_modified
        values, indicies = is_predicted_modified.max(dim=-1)
        return values, ids
    
    
    def compute_loss(self, x, y, exp, loss_type):
        predicted_mod_logits = self(x)
        seq_predictions_logits, _ = predicted_mod_logits.squeeze().max(dim=1)
        mod_loss = self.binary_ce(seq_predictions_logits, y.flatten())
        
        is_predicted_modified = torch.sigmoid(seq_predictions_logits)
        acc = self.acc(is_predicted_modified, y.flatten())
        # auroc = self.auroc(is_predicted_modified, y.flatten())
        self.log(f'{loss_type} acc', acc, on_epoch=True)
        # self.log(f'{loss_type} auroc', auroc, on_epoch=True)
        
        exps = np.array(exp)
        for e in np.unique(exp):
            indicies = exps == e
            batch_size = sum(indicies)
            if(batch_size>0):
                e_predicted_labels = is_predicted_modified[indicies]
                e_y = y[indicies]
                
                e_acc = self.acc(e_predicted_labels, e_y.flatten())
                # e_auroc = self.auroc(e_predicted_labels, e_y.flatten())
                
                self.log(f'{e} {loss_type} acc', e_acc, on_epoch=True, batch_size=batch_size)
                # self.log(f'{e} {loss_type} auroc', e_auroc, on_epoch=True, batch_size=batch_size)
                
        return mod_loss
    
    
        
