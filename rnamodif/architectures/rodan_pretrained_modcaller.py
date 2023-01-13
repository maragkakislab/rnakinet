from RODAN.basecall import load_model
import torch
import torchmetrics
import pytorch_lightning as pl
from torch.nn import functional as F
from types import SimpleNamespace
from rnamodif.architectures.bonito_pretrained import RNNPooler
from bonito_pulled.bonito.nn import Permute
import numpy as np

class RodanPretrainedModcaller(pl.LightningModule):
    #TODO warmup steps?
    def __init__(self, lr=5e-4, warmup_steps=1000):
        super().__init__()
        #TODO fix module importing without hacking imports
        #TODO vocab ATCG - but rna is AUCG
        torchdict = torch.load('/home/jovyan/RNAModif/RODAN/rna.torch', map_location="cpu")
        origconfig = torchdict["config"]
        d = origconfig
        n = SimpleNamespace(**d)
        args = {
            'debug':False, #True prints out more info
            'arch':None,
        }
        self.trainable_rodan, device = load_model('/home/jovyan/RNAModif/RODAN/rna.torch', config=n, args=SimpleNamespace(**args))
        self.original_rodan, _ = load_model('/home/jovyan/RNAModif/RODAN/rna.torch', config=n, args=SimpleNamespace(**args))
        for par in self.original_rodan.parameters():
            par.requires_grad = False
        
        self.mod_neuron = torch.nn.Linear(768,1)
        #TODO add 1 more layer + nonlinearity?
    
        self.lr = lr
        
        self.acc = torchmetrics.classification.BinaryAccuracy()
        self.cm = torchmetrics.classification.BinaryConfusionMatrix(normalize='true')
        self.auroc = torchmetrics.classification.AUROC(task='binary')
        # self.ce = torch.nn.CrossEntropyLoss()
        self.kl = torch.nn.KLDivLoss(log_target=True)
        self.binary_ce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.warmup_steps = warmup_steps
        print('TODO check pos/neg weights are OK and not overweighting posivies')
        
    def forward(self, x):
        feature_vector = self.trainable_rodan.convlayers(x)
        feature_vector = feature_vector.permute(0,2,1)
        
        mod_pred = self.mod_neuron(feature_vector) #no need for sigmoid, i use logits BCE
        
        og_preds = self.trainable_rodan.final(feature_vector)
        og_preds = torch.nn.functional.log_softmax(og_preds, 2)
        og_preds =  og_preds.permute(1, 0, 2)
              
        return og_preds, mod_pred
    
    
    def configure_optimizers(self):
        # + LR scheduling (i have custom head)
        #TODO i can do custom LR {'params': model.classifier.parameters(), 'lr': 1e-3}
        optimizer =  torch.optim.AdamW(params=[{'params':self.trainable_rodan.parameters()},{'params':self.mod_neuron.parameters()}], lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=self.warmup_steps)
        
        return [optimizer], [scheduler]
    
    def training_step(self, train_batch, batch_idx, dataloader_idx=None):
        x,y,exp = train_batch
        loss = self.compute_loss(x, y, exp,'train')
        self.log('train_loss', loss, on_epoch=True)
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
        # values = torch.sigmoid(values)
        return values, ids
        # predicted_labels = (values > 0.5).int().unsqueeze(dim=1)
        # res = torch.sigmoid(logits)
        # return res, ids
    
    def compute_loss(self, x, y, exp, loss_type):
        # "vocab": [ '<PAD>', 'A', 'C', 'G', 'T' ]}
        predicted_basecall, predicted_mod_logits = self(x)
        rodan_basecall = self.original_rodan(x)
        is_any_A_map = (rodan_basecall.argmax(dim=2)==1).int().float()  # ==1 because A is index 1 in vocabuary
        mask = y.squeeze().repeat((420,1))
        is_mod_A_map = is_any_A_map*mask #compute only for label 1s (positive sequences)
        #TODO redo squeeze to dim=-1 like in pred step to aviod batchsize=1 problem dimension removed
        is_predicted_modified_logits = predicted_mod_logits.squeeze().permute(1,0) #Remove permute and do max(dim=-1 or dim=1)
        is_N_map = (rodan_basecall.argmax(dim=2)==0).int().float()
        is_not_N_map = 1-is_N_map

        
        N_count = is_N_map.sum()
        pos_count = is_mod_A_map.sum()
        neg_count = (1-is_mod_A_map).sum() - N_count
        total_count = (pos_count+neg_count)*x.size()[0] 
        zero_weight = (total_count)/neg_count #negative
        one_weight = (total_count)/pos_count #positive
        
        
        #TODO argmax (MIL) over all As - punish 1 of As and all non-As
        mod_loss = self.binary_ce(is_predicted_modified_logits, is_mod_A_map)
        basecalling_loss = self.kl(predicted_basecall, rodan_basecall) #TODO remove orig rodan inference if im not using this
        
        if(pos_count == 0):
            weighted_mod_loss = (mod_loss*is_not_N_map).mean()
            
        else:
            weight_map = torch.tensor([zero_weight,one_weight]).cuda()
            weights = weight_map.repeat(420,1).permute(1,0) #420 <- len of feature vector for sequence of 4096
            zerom = (weights[0]*(1-is_mod_A_map).permute(1,0)).permute(1,0)
            onem = (weights[1]*is_mod_A_map.permute(1,0)).permute(1,0)
            weighter = zerom+onem
            weighted_mod_loss = ((mod_loss*weighter)*is_not_N_map).mean()
        
        
        #TODO make neighnours also positive? What percentage of As is in a read in general according to rodan?
        self.log(f'{loss_type} mod loss', mod_loss.mean(), on_epoch=True)
        self.log(f'{loss_type} weighted mod loss', weighted_mod_loss, on_epoch=True)
        self.log(f'{loss_type} basecalling loss', basecalling_loss, on_epoch=True)
        
        #COmputing accuracy for As only - also count loss only from these?
        maxes = []
        for i in range(x.size()[0]): # i = batch element number (0-32), can have any num of As, so there needs to be for loop
            A_positions_preds = (is_predicted_modified_logits[:,i][is_any_A_map [:,i] == 1])
            probas = torch.sigmoid(A_positions_preds)
            if(probas.numel() == 0):
                value = torch.tensor(0).to(self.device)
            else:
                value, _ = probas.max(dim=0)
            maxes.append(value)
        
        #TODO do balanced accuracy?
        acc_on_As = self.acc(torch.stack(maxes), y.flatten())
        self.log(f'{loss_type} acc on As', acc_on_As, on_epoch=True)
        #TODO compute auroc from max only? Not all reads
        auroc = self.auroc(torch.stack(maxes), y.flatten())
        self.log(f'{loss_type} auroc on As', auroc, on_epoch=True)
        
        
        is_predicted_modified = torch.sigmoid(is_predicted_modified_logits)
        values, indicies = is_predicted_modified.max(dim=0)
        predicted_labels = (values > 0.5).int().unsqueeze(dim=1) #remove >0.5, obsolete
        acc = self.acc(predicted_labels, y)
        self.log(f'{loss_type} acc', acc, on_epoch=True)
        
        
        exps = np.array(exp)
        for e in np.unique(exp):
            indicies = exps == e
            batch_size = sum(indicies)
            if(batch_size>0):
                e_predicted_labels = predicted_labels[exps==e]
                e_y = y[exps==e]
                
                e_acc = self.acc(e_predicted_labels, e_y)
                self.log(f'{e} {loss_type} acc', e_acc, on_epoch=True, batch_size=batch_size)
                
        
        
        return weighted_mod_loss
        # return basecalling_loss + mod_loss.mean()
    
    
        
