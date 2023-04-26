from RODAN.basecall import load_model
import torch
import torchmetrics
import pytorch_lightning as pl
from torch.nn import functional as F
from types import SimpleNamespace
from rnamodif.architectures.bonito_pretrained import RNNPooler
from bonito_pulled.bonito.nn import Permute
import numpy as np
from torcheval.metrics import BinaryAUROC

class RodanPretrainedModcaller(pl.LightningModule):
    #TODO warmup steps?
    def __init__(self, use_Ns_for_loss=True, weighted_loss=False, use_basecalling_loss=False, loss_only_from_As=False, predict_only_on_As = False, lr=5e-4, warmup_steps=1000):
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
        # self.auroc2 = BinaryAUROC()
        # self.ce = torch.nn.CrossEntropyLoss()
        self.kl = torch.nn.KLDivLoss(log_target=True, reduction='none')
        # self.nll = torch.nn.NLLLoss()
        self.binary_ce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.warmup_steps = warmup_steps
        self.use_Ns_for_loss = use_Ns_for_loss
        self.weighted_loss = weighted_loss
        self.use_basecalling_loss = use_basecalling_loss
        self.loss_only_from_As = loss_only_from_As
        self.predict_only_on_As = predict_only_on_As
        print('Only A test predicts:', self.predict_only_on_As)
        def balanced_acc(pred, target):
            s = torchmetrics.functional.specificity(pred,target, task='binary')
            r = torchmetrics.functional.recall(pred,target, task='binary')
            return ((s+r)/2) #Fixed, rerun
        self.balanced_acc = balanced_acc
        
    def forward(self, x):
        feature_vector = self.trainable_rodan.convlayers(x)
        feature_vector = feature_vector.permute(0,2,1)
        
        mod_pred = self.mod_neuron(feature_vector) #no need for sigmoid, i use logits BCE
        
        og_preds = self.trainable_rodan.final(feature_vector)
        # og_preds = torch.nn.functional.log_softmax(og_preds, 2) #log Not desired for KL loss
        og_preds = torch.nn.functional.softmax(og_preds, 2)
        
        og_preds =  og_preds.permute(1, 0, 2)
              
        return og_preds, mod_pred
    
    
    def configure_optimizers(self):
        # + LR scheduling (i have custom head)
        #TODO i can do custom LR {'params': model.classifier.parameters(), 'lr': 1e-3}
        #TODO WD=0 for pretrained model?
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
        self.log('valid_loss', loss, on_epoch=True) 
        # self.log_metrics(output, y.int(), exp, 'valid')
    
    def predict_step(self, batch, batch_idx):
        xs, ids = batch
        _, mod_pred = self.forward(xs)
               
        is_predicted_modified = mod_pred.squeeze(dim=-1)
        is_predicted_modified = torch.sigmoid(is_predicted_modified)
        
        #Predicting only on As - parametrize
        if self.predict_only_on_As:
            rodan_basecall = self.original_rodan(xs).detach()
            is_any_A_map = (rodan_basecall.argmax(dim=2)==1).int().float()  # ==1 because A is index 1 in vocabuary
            is_predicted_modified_As = is_predicted_modified*is_any_A_map.permute(1,0)
            values, indicies = torch.max(is_predicted_modified_As,dim=-1)
            return values, ids
        
        
        #PRedicting over whole sequence
        values, indicies = torch.max(is_predicted_modified,dim=-1)
        return values, ids
    
    def compute_loss(self, x, y, exp, loss_type):
        # "vocab": [ '<PAD>', 'A', 'C', 'G', 'T' ]}
        predicted_basecall, predicted_mod_logits = self(x)
        #TODO redo squeeze to dim=-1 like in pred step to aviod batchsize=1 problem dimension removed
        is_predicted_modified_logits = predicted_mod_logits.squeeze().permute(1,0) #Remove permute and do max(dim=-1 or dim=1)
        
        rodan_basecall = self.original_rodan(x).detach()
        is_any_A_map = (rodan_basecall.argmax(dim=2)==1).int().float()  # ==1 because A is index 1 in vocabuary
        mask = y.squeeze().repeat((420,1))
        is_mod_A_map = is_any_A_map*mask #compute only for label 1s (positive sequences)
        is_N_map = (rodan_basecall.argmax(dim=2)==0).int().float()
        is_not_N_map = 1-is_N_map
        N_percentage = (is_N_map.sum()/(is_N_map.sum()+is_not_N_map.sum()))
        self.log(f'{loss_type} N_percentage', N_percentage,on_epoch=True) 

        if(self.use_Ns_for_loss):
            is_not_N_map = torch.ones_like(is_not_N_map)


        N_count = is_N_map.sum()
        pos_count = is_mod_A_map.sum()
        neg_count = (1-is_mod_A_map).sum() - N_count
        total_count = pos_count+neg_count
        if(self.weighted_loss):
            zero_weight = (total_count)/neg_count #negative
            one_weight = (total_count)/pos_count #positive
        else:
            zero_weight = 1
            one_weight = 1
        self.log(f'{loss_type} avg As in positive sequence', pos_count/y.sum() ,on_epoch=True)
        
        #TODO argmax (MIL) over all As - punish 1 of As and all non-As
        mod_loss = self.binary_ce(is_predicted_modified_logits, is_mod_A_map)
        
        
        # print(predicted_basecall.size())
        # print(rodan_basecall.size())
        basecalling_loss = self.kl(predicted_basecall, rodan_basecall).mean() #TODO remove orig rodan inference if im not using this
        # basecalling_loss = self.nll(predicted_basecall, rodan_basecall)    
        
        # return (mod_loss*is_not_N_map).mean() + basecalling_loss
        
        if(pos_count == 0):
            weighted_mod_loss = (mod_loss*is_not_N_map).mean()
            
        else:
            zerom = (torch.mul((1-is_mod_A_map), zero_weight).permute(1,0)).permute(1,0)
            onem = (torch.mul(is_mod_A_map, one_weight).permute(1,0)).permute(1,0)
            weighter = zerom+onem
            weighted_mod_loss = ((mod_loss*weighter)*is_not_N_map).mean()
        
        
        #TODO make neighnours also positive? What percentage of As is in a read in general according to rodan?
        self.log(f'{loss_type} mod loss', mod_loss.mean(), on_epoch=True)
        self.log(f'{loss_type} weighted mod loss', weighted_mod_loss, on_epoch=True)
        self.log(f'{loss_type} basecalling loss', basecalling_loss, on_epoch=True)
        
        #COmputing accuracy for As only - also count loss only from these?
        
        
        
        all_As_probas = []
        all_nonAs_probas = []
        all_As_labels = []
        max_As_probas = []
        max_As_labels = []
        only_As_losses = []
        for i in range(x.size()[0]): # i = batch element number (0-32), can have any num of As, so there needs to be for loop
            #TODO do non-As accuracy too (all y = 0)
            A_positions_preds = (is_predicted_modified_logits[:,i][is_any_A_map [:,i] == 1])
            nonA_positions_preds = (is_predicted_modified_logits[:,i][is_any_A_map [:,i] != 1])
            
            
            sample_loss = self.binary_ce(A_positions_preds, y[i].repeat(len(A_positions_preds)))
            only_As_losses.append(sample_loss)
            
            probas = torch.sigmoid(A_positions_preds)
            nonA_probas = torch.sigmoid(nonA_positions_preds)
            
            all_As_probas.append(probas)
            all_nonAs_probas.append(nonA_probas)
                
            A_labels = y[i].int().repeat(len(probas),1)
            all_As_labels.append(A_labels)
            
            if(len(probas) > 0):
                max_A_prob, indicies = torch.max(probas, dim=0)
                max_As_labels.append(y[i].int())
                max_As_probas.append(max_A_prob.unsqueeze(dim=0))
                
        only_As_loss = torch.sum(torch.cat(only_As_losses))
        if(self.loss_only_from_As):
            weighted_mod_loss = only_As_loss
        
        As_probas = torch.cat(all_As_probas)
        nonAs_probas = torch.cat(all_nonAs_probas)
        
        As_labels = torch.cat(all_As_labels).flatten()
        acc_on_As = self.acc(As_probas, As_labels)
        
        acc_on_nonAs = self.acc(nonAs_probas, nonAs_probas*0)
        
        #CANT compute AUROC = pytorch memory leaks...
        # auroc_on_As = self.auroc(As_probas, As_labels)
        
        max_As_probas = torch.cat(max_As_probas)
        max_As_labels = torch.cat(max_As_labels)
        acc_on_max_As = self.acc(max_As_probas, max_As_labels)
        
        self.log(f'{loss_type} acc on max As', acc_on_max_As, on_epoch=True)
        
        self.log(f'{loss_type} acc on As', acc_on_As, on_epoch=True)
        self.log(f'{loss_type} balanced acc on As', self.balanced_acc(As_probas, As_labels), on_epoch=True)
        
        self.log(f'{loss_type} acc on nonAs', acc_on_nonAs, on_epoch=True)
        self.log(f'{loss_type} balanced acc on nonAs', self.balanced_acc(nonAs_probas, nonAs_probas*0), on_epoch=True)
        # self.log(f'{loss_type} auroc on As', auroc_on_As)#on_epoch=True)
        
        
        
        is_predicted_modified = torch.sigmoid(is_predicted_modified_logits)
        # values, indicies = is_predicted_modified.max(dim=0)
        #NEW
        values, indicies = torch.max(is_predicted_modified,dim=0)
        
        # predicted_labels = (values > 0.5).int().unsqueeze(dim=1) #remove >0.5, obsolete
        # acc = self.acc(predicted_labels, y)
        acc = self.acc(values, y.flatten())
        # auroc = self.auroc(values, y.flatten())
        
        self.log(f'{loss_type} acc', acc, on_epoch=True)
        # self.log(f'{loss_type} auroc', auroc)#on_epoch=True)
        
        exps = np.array(exp)
        for e in np.unique(exp):
            indicies = exps == e
            batch_size = sum(indicies)
            if(batch_size>0):
                e_predicted_labels = values[exps==e]
                # e_predicted_labels = predicted_labels[exps==e]
                e_y = y[exps==e]
                
                e_acc = self.acc(e_predicted_labels, e_y.flatten())
                # e_auroc = self.auroc(e_predicted_labels, e_y.flatten())
                
                #Computed from max over all positions
                self.log(f'{e} {loss_type} acc', e_acc, on_epoch=True)
                # self.log(f'{e} {loss_type} auroc', e_auroc)#on_epoch=True, batch_size=batch_size)
                
        if(self.use_basecalling_loss):
            #TODO balance losses
            return weighted_mod_loss+basecalling_loss
        else:
            return weighted_mod_loss
    
    
        
