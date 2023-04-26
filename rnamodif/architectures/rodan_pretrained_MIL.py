from RODAN.basecall import load_model
from RODAN.model import Swish
import torch
import torchmetrics
import pytorch_lightning as pl
from torch.nn import functional as F
from types import SimpleNamespace
import numpy as np

class Permute(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

    def to_dict(self, include_weights=False):
        return {'dims': self.dims}

class RodanPretrainedMIL(pl.LightningModule):
    def __init__(self, pretrained_lr=5e-4, my_layers_lr=2e-3, warmup_steps = 10000):
        super().__init__()
        # rodan_path = './RODAN/rna.torch'
        print('TODO fix rodan path')
        rodan_path = '/home/jovyan/RNAModif/RODAN/rna.torch'

        torchdict = torch.load(rodan_path, map_location="cpu")
        origconfig = torchdict["config"]
        d = origconfig
        n = SimpleNamespace(**d)
        args = {
            'debug':False, #True prints out more info
            'arch':None,
        }
        self.trainable_rodan, device = load_model(rodan_path, config=n, args=SimpleNamespace(**args))
        self.original_rodan, _ = load_model(rodan_path, config=n, args=SimpleNamespace(**args))
        
        # for par in list(self.trainable_rodan.parameters())[:-2]:
            # par.requires_grad = False
    
        for par in list(self.original_rodan.parameters()):
            par.requires_grad = False
    
        
        self.seq_model = torch.nn.Sequential(
            # torch.nn.Linear(5,1),
            Permute((1,2,0)),
            torch.nn.Conv1d(in_channels=10, out_channels=1, kernel_size=5),
        )
        
        
        # self.flatten = torch.nn.Flatten()
        self.softmax = torch.nn.Softmax(dim=-1)
        
        self.pretrained_layers_lr = pretrained_lr
        self.my_layers_lr = my_layers_lr
        self.warmup_steps = warmup_steps
        
        self.acc = torchmetrics.classification.BinaryAccuracy()
        self.cm = torchmetrics.classification.BinaryConfusionMatrix(normalize='true') 
        
        
    def forward(self, x):
        
        x_t = self.softmax(self.trainable_rodan(x))
        x_o = self.softmax(self.original_rodan(x))
        x = torch.cat([x_t,x_o], dim=-1)
        
        x = self.seq_model(x)
        
        x, indicies = torch.max(x, axis=-1)
        # x = self.flatten(x)
        # print(x.size())
        
        return x
        
        
        # bx = self.original_rodan(x)
        
        # txs = []
        # for i in range(0,self.window_size,self.subwindow_size):
        #     sub_x = x[:,:,i:i+self.subwindow_size]
        #     tx = self.trainable_rodan(sub_x)
        #     tx = self.seq_model(tx)
        #     txs.append(tx)
            
        # r = torch.stack(txs, dim=-1)
        # r = 1 - torch.prod(1-r, axis=-1)
        # r, indicies = torch.max(r, axis=-1)
        
        # return r
    
    def configure_optimizers(self):
        my_layers_lr = self.my_layers_lr
        pretrained_layers_lr = self.pretrained_layers_lr
        layer_to_lr = [(self.trainable_rodan, pretrained_layers_lr), *[(child, my_layers_lr) for child in self.seq_model.children()]]
        groups = [{'params':list(m.parameters()), 'lr':lr} for (m,lr) in layer_to_lr]
        
        optimizer = torch.optim.AdamW(groups, lr=self.pretrained_layers_lr, weight_decay=0.01) #Turn off WD for Transfer learning?
        # optimizer = torch.optim.AdamW([{'params':self.trainable_rodan.parameters()}, {'params':self.seq_model.parameters()}], weight_decay=0.01, lr=self.my_layers_lr)
        
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
    
    # def my_loss(self, output, y):
        # torch.log(torch.log(output,
    
    def validation_step(self, val_batch, batch_idx, dataloader_idx=None):
        x,y,exp = val_batch
        output = self(x)
        loss = F.binary_cross_entropy_with_logits(output, y)
        self.log('valid_loss', loss, on_epoch=True)
        self.log_metrics(output, y.int(), exp, 'valid')
        
    def predict_step(self, batch, batch_idx):
        xs, ids = batch
        logits = self.forward(xs)
        res = torch.sigmoid(logits)
        return res, ids
    
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
    