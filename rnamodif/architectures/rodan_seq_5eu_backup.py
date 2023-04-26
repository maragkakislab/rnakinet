from RODAN.basecall import load_model
import torch
import torchmetrics
import pytorch_lightning as pl
from torch.nn import functional as F
from types import SimpleNamespace
from rnamodif.architectures.bonito_pretrained import RNNPooler
from bonito_pulled.bonito.nn import Permute
import numpy as np
import re
from numpy.linalg import inv
from sklearn.metrics import roc_auc_score

class RodanPretrainedSeqcaller5eu(pl.LightningModule):
    def __init__(self, lr=1e-3, warmup_steps=1000, wd=0.01, freeze=False, fr_layers=0, corr_loss=False):
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
        
        
#         class Unsqueeze(torch.nn.Module):
#             def __init__(self):
#                 super().__init__()

#             def forward(self, x):
#                 return x.unsqueeze(dim=-1)
            
#         class GRUflatten(torch.nn.Module):
#             def __init__(self):
#                 super().__init__()

#             def forward(self, x):
#                 return x[1].permute(1,0,2).flatten(start_dim=1)
            
        self.head = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=768, out_channels= 64, kernel_size=3),
            # torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3),
            
            torch.nn.Conv1d(in_channels=64, out_channels=8, kernel_size=3),
            # torch.nn.BatchNorm1d(8),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3),
            
            torch.nn.Flatten(),
            
            #AVG POOL MODEL
            # torch.nn.AdaptiveAvgPool1d(1),
            
            # RNN HEAD
            # Unsqueeze(),
            # torch.nn.GRU(input_size=1, hidden_size=8, num_layers=1, batch_first=True, bidirectional=True),
            # GRUflatten(),
            # #TODO activate??
            # torch.nn.Linear(16, 1)
            
            #STANDARD MODEL
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(360,100), #4096 model
            # torch.nn.Linear(728,100), #8192 model
            torch.nn.ReLU(),
            torch.nn.Linear(100,1)
        )
        
        print('FREEZING', fr_layers, 'layers')
        if(freeze):
            freeze_rodan(self, freeze_till=fr_layers, verbose=0)
        
        #TODO DROP LINEAR LAYER so it doesnt show up in model print
        print('Enabling gradients')
        torch.set_grad_enabled(True)
        
        self.lr = lr
        self.wd = wd
        self.acc = torchmetrics.classification.Accuracy(task="binary")
        

        if(corr_loss):
            print('using corrected BCE loss!')
            #TODO set correction matrix to <40-80> percent FP
            # T = torch.tensor([[1.0, 0.0],
            #       [0.8, 0.2]]).cuda()
            T = torch.tensor([[2.0, 0.0],
                  [0.0, 1.0]]).cuda()
            def binary_cross_entropy_corrected(pred, y): 
                pred = torch.sigmoid(pred)
                y = y.flatten()
                y_true = torch.stack([1-y,y]).permute(1,0)
                y_pred = torch.stack([1-pred, pred]).squeeze().permute(1,0)
                uncorrected_losses = F.binary_cross_entropy(y_pred, y_true, reduction='none')
                losses_corrected = torch.matmul(T, uncorrected_losses.T).T
                corrected_loss = losses_corrected.mean()
                return corrected_loss
            self.ce = binary_cross_entropy_corrected
        else:
            self.ce = torch.nn.BCEWithLogitsLoss()
            
        
        self.warmup_steps = warmup_steps
    
    def forward(self, x):
        feature_vector = self.trainable_rodan.convlayers(x)
        out = self.head(feature_vector)
        return out
    
    def configure_optimizers(self):
        optimizer =  torch.optim.AdamW(params=[{'params':self.trainable_rodan.parameters()},{'params':self.head.parameters()}], lr=self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=self.warmup_steps)
        return [optimizer], [scheduler]
    
    def training_step(self, train_batch, batch_idx, dataloader_idx=None):
        x,y,exp = train_batch
        loss = self.compute_loss(x, y, exp,'train')
        self.log('train_loss', loss, on_epoch=True)
        sch = self.lr_schedulers()
        sch.step()
        current_lr_list = sch.get_last_lr()
        return loss
    
    def validation_step(self, val_batch, batch_idx, dataloader_idx=None):
        x,y,identifier = val_batch
        exp = identifier['exp']
        loss, preds = self.compute_loss(x, y, exp, 'valid', return_preds=True)
        self.log('valid_loss', loss, on_epoch=True) #ADDED on_epoch=True
        return {'preds':preds, 'identifier':identifier}
    
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
        self.log(f'{loss_type} acc', acc, on_epoch=True)
        
        exps = np.array(exp)
        for e in np.unique(exp):
            indicies = exps == e
            batch_size = sum(indicies)
            if(batch_size>0):
                self.log(f'{loss_type} {e} acc', self.acc(is_predicted_modified[exps==e], y[exps==e]), on_epoch=True)
        
        if(return_preds):
            return mod_loss, is_predicted_modified#.detach().cpu().numpy()
        return mod_loss
        
    
    def validation_epoch_end(self, outputs):
        read_to_preds = {}
        read_to_label = {}
        read_to_exp = {}
        for log in outputs:
            preds = log['preds'].cpu().numpy()
            ids = log['identifier']
            
            #TODO Detaching slow?
            for i, (readid, pred, label, exp) in enumerate(zip(ids['readid'],preds, ids['label'].detach().cpu(), ids['exp'])):
                if(readid not in read_to_preds.keys()):
                    read_to_preds[readid] = []
                    read_to_label[readid] = label
                    read_to_exp[readid] = exp
                read_to_preds[readid].append(pred)
                
        read_to_preds_meanpool = {}
        read_to_preds_maxpool = {}
        for k,v in read_to_preds.items():
            read_to_preds_meanpool[k] = np.array(v).mean()
            read_to_preds_maxpool[k] = np.array(v).max()
        
        auroc_2022_mean = get_auroc_score(['5eu_2022_chr1_pos', '5eu_2022_chr1_neg'], read_to_exp, read_to_preds_meanpool, read_to_label)
        auroc_2022_max = get_auroc_score(['5eu_2022_chr1_pos', '5eu_2022_chr1_neg'], read_to_exp, read_to_preds_maxpool, read_to_label)
        self.log(f'valid 2022 auroc (meanpool)', auroc_2022_mean)
        self.log(f'valid 2022 auroc (maxpool)', auroc_2022_max)
        
        auroc_2020_mean = get_auroc_score(['5eu_2020_pos', 'UNM_2020'], read_to_exp, read_to_preds_meanpool, read_to_label)
        auroc_2020_max = get_auroc_score(['5eu_2020_pos', 'UNM_2020'], read_to_exp, read_to_preds_maxpool, read_to_label)
        self.log(f'valid 2020 auroc (meanpool)', auroc_2020_mean)
        self.log(f'valid 2020 auroc (maxpool)', auroc_2020_max)
        
        auroc_nanoid_mean = get_auroc_score(['Nanoid_pos', 'Nanoid_neg'], read_to_exp, read_to_preds_meanpool, read_to_label)
        auroc_nanoid_max = get_auroc_score(['Nanoid_pos', 'Nanoid_neg'], read_to_exp, read_to_preds_maxpool, read_to_label)
        self.log(f'valid nanoid auroc (meanpool)', auroc_nanoid_mean)
        self.log(f'valid nanoid auroc (maxpool)', auroc_nanoid_max)
        
        

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
    
            
def freeze_rodan(model, freeze_till, verbose=0):
    #freeze_till max is 21
    for name, module in model.named_modules():
        if(name in ['','trainable_rodan', 'trainable_rodan.convlayers']):
            continue
        pattern = r"conv\d+"
        match = re.search(pattern, name)
        if(match):
            conv_index = int(match.group(0)[4:])
            if(conv_index > freeze_till):
                # print('breaking')
                break
        if('drop' in name):
            # module.eval()
            module.p = 0.0

        for param in module.parameters():
            param.requires_grad = False
    
    if(verbose == 1):
        for name, module in model.named_modules():
            if(len(list(module.parameters()))>0):
                print(all([p.requires_grad for p in list(module.parameters())]), name)