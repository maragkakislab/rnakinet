from RODAN.basecall import load_model
import torch
import torchmetrics
import pytorch_lightning as pl
from torch.nn import functional as F
from types import SimpleNamespace
from rnamodif.architectures.bonito_pretrained import RNNPooler
from bonito_pulled.bonito.nn import Permute
import numpy as np

class RodanPretrainedSeqcaller(pl.LightningModule):
    def __init__(self, num_classes=5, lr=1e-3, warmup_steps=1000):
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
        self.mod_neuron = torch.nn.Linear(768,num_classes)
        self.dropout = torch.nn.Dropout(p=0.5)
        # self.mod_neuron = torch.nn.Linear(768,1)
        # self.flatten = torch.nn.Flatten()
        
        #TODO add to OPTIMIZER when uncommenting
        # self.final = torch.nn.Linear(420,num_classes)
        
        self.lr = lr
        
        self.acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.cm = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=num_classes)
       
        print('using weighted loss')
        # weights = torch.tensor([1/(4*5) ,1/5, 1/5, 1/5, 1/5]).to(self.device)
        weights = torch.tensor([(2*1)/(4*5) ,1/5, 1/5, 1/5, 1/5]).to(self.device)
        
        print(weights)
        self.ce = torch.nn.CrossEntropyLoss(weight=weights)
        
        self.warmup_steps = warmup_steps
        
        self.exp_to_label_mapper = {
            'm6A_0_covid': 0,
            's4U_0_covid': 0,
            'ac4C_0_covid' : 0,
            'remdesivir_0_covid' : 0,
            
            'm6A_33_covid': 1,
            'm6A_10_covid': 1,
            'm6A_5_covid' : 1,
            
            's4U_33_covid': 2,
            's4U_10_covid': 2,
            's4U_5_covid': 2,
            
            'ac4C_33_covid' : 3,
            'ac4C_10_covid' : 3,
            
            'remdesivir_33_covid' : 4,
            'remdesivir_5_covid' : 4,
            
        }
    
    def exps_to_labels(self, exps):
        return torch.tensor([self.exp_to_label_mapper[e] for e in exps]).to(self.device)
        
    def forward(self, x):
        feature_vector = self.trainable_rodan.convlayers(x)
        feature_vector = feature_vector.permute(0,2,1)
        mod_pred = self.mod_neuron(feature_vector) #no need for sigmoid, i use logits BCE
        mod_pred = self.dropout(mod_pred)
        final_pred, _ = torch.max(mod_pred, dim=-2)
        # mod_pred = torch.nn.functional.relu(mod_pred)
        # mod_pred = self.flatten(mod_pred)
        # final_pred = self.final(mod_pred)
        return final_pred
    
    
    def configure_optimizers(self):
        # optimizer =  torch.optim.AdamW(params=[{'params':self.trainable_rodan.parameters()},{'params':self.mod_neuron.parameters()}, {'params':self.final.parameters()}], lr=self.lr, weight_decay=0.01)
        optimizer =  torch.optim.AdamW(params=[{'params':self.trainable_rodan.parameters()},{'params':self.mod_neuron.parameters()}], lr=self.lr, weight_decay=0.01)
        
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=self.warmup_steps)
        
        return [optimizer], [scheduler]
    
    def training_step(self, train_batch, batch_idx, dataloader_idx=None):
        x,y,exp = train_batch
        loss = self.compute_loss(x, y, exp,'train')['loss']
        self.log('train_loss', loss, on_epoch=True)
        
        sch = self.lr_schedulers()
        sch.step()
        current_lr_list = sch.get_last_lr()
        # pretrained_lr = current_lr_list[0]
        # my_layer_lr = current_lr_list[0] #The same
        # self.log('learning rate (pretrained part)', pretrained_lr)
        # self.log('learning rate (custom part)', my_layer_lr)
        
        return loss
    
    def validation_step(self, val_batch, batch_idx, dataloader_idx=None):
        x,y,exp = val_batch
        logs = self.compute_loss(x, y, exp, 'valid')
        loss = logs['loss']
        self.log('valid_loss', loss, on_epoch=True) #ADDED on_epoch=True
        # cm = logs['cm']
        # self.log_metrics(output, y.int(), exp, 'valid')
        return logs
        
    
    # def predict_step(self, batch, batch_idx, expanded=False):
    #     xs, ids = batch
    #     _, mod_pred = self.forward(xs)
    #     is_predicted_modified = mod_pred.squeeze(dim=-1)
    #     #ADDED
    #     is_predicted_modified = torch.sigmoid(is_predicted_modified)
    #     if(expanded):
    #         return is_predicted_modified
    #     values, indicies = is_predicted_modified.max(dim=-1)
    #     return values, ids
    
    
    def compute_loss(self, x, y, exp, loss_type):
        seq_predictions_logits = self(x)
        y = self.exps_to_labels(exp)
        mod_loss = self.ce(seq_predictions_logits, y.flatten())
        
        is_predicted_modified = torch.softmax(seq_predictions_logits, dim=-1)
        acc = self.acc(is_predicted_modified, y.flatten())
        self.log(f'{loss_type} acc', acc, on_epoch=True)
        
        # cm = self.cm(is_predicted_modified, y.flatten())
        
        cms = {}
        exps = np.array(exp)
        for e in np.unique(exp):
            indicies = exps == e
            batch_size = sum(indicies)
            if(batch_size>0):
                sub_cm = self.cm(is_predicted_modified[exps==e], y[exps==e])
                cms[e] = sub_cm
        

        # return {'loss':mod_loss, 'cm':cm}
        return {'loss':mod_loss, 'cms':cms}
    
    
    def validation_epoch_end(self, outputs):
        id_to_dset = lambda i: ','.join([k for k,v in self.exp_to_label_mapper.items() if v==i])
        labels = [id_to_dset(i) for i in range(1+max(self.exp_to_label_mapper.values()))]
        
        agg_cms = {}
        
        for log in outputs:
            cm_dict = log['cms']
            for exp, cm in cm_dict.items():
                if(exp not in agg_cms.keys()):
                    agg_cms[exp] = [cm.detach().cpu().numpy()]
                else:
                    agg_cms[exp].append(cm.detach().cpu().numpy())
        for k,v in agg_cms.items():
            agg_cms[k] = sum(v)
            # self.logger.experiment.log_confusion_matrix(matrix=agg_cms[k], labels=labels, file_name=f"{k}-confusion-matrix.json")
        
        high_mod_cm = sum([agg_cms[exp] for exp in self.exp_to_label_mapper.keys() if (('_0_covid' in exp) or ('_33_covid' in exp))])
        self.logger.experiment.log_confusion_matrix(matrix=high_mod_cm, labels=labels, file_name=f"high-mod-confusion-matrix.json")
        
        mid_mod_cm = sum([agg_cms[exp] for exp in self.exp_to_label_mapper.keys() if (('_0_covid' in exp) or ('_10_covid' in exp))])
        self.logger.experiment.log_confusion_matrix(matrix=mid_mod_cm, labels=[label.replace('33','10') for label in labels], file_name=f"mid-mod-confusion-matrix.json")
        
        low_mod_cm = sum([agg_cms[exp] for exp in self.exp_to_label_mapper.keys() if (('_0_covid' in exp) or ('_5_covid' in exp))])
        self.logger.experiment.log_confusion_matrix(matrix=low_mod_cm, labels=[label.replace('33','5') for label in labels], file_name=f"low-mod-confusion-matrix.json")
            
            
        # cms = sum([x['cm'] for x in outputs]).detach().cpu().numpy()
        # id_to_dset = lambda i: ','.join([k for k,v in self.exp_to_label_mapper.items() if v==i])
        # labels = [id_to_dset(i) for i in range(1+max(self.exp_to_label_mapper.values()))]
        # self.logger.experiment.log_confusion_matrix(matrix=cms, labels=labels, file_name="confusion-matrix.json")
        
