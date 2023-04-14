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

class RodanPretrainedSeqcaller5eu(pl.LightningModule):
    def __init__(self, num_classes=2, lr=1e-3, warmup_steps=1000, freeze=False, fr_layers=0, corr_loss=False):
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
        
        
#         self.conv_1 = torch.nn.Conv1d(in_channels=768, out_channels= 64, kernel_size=3)
#         self.maxpool = torch.nn.MaxPool1d(kernel_size=3)
#         self.conv_2 = torch.nn.Conv1d(in_channels=64, out_channels=8, kernel_size=3)
#         self.flatten = torch.nn.Flatten()
        
#         self.dropout = torch.nn.Dropout(p=0.5)
#         self.linear_1 = torch.nn.Linear(360,100)
#         self.linear_2 = torch.nn.Linear(100,2)
        
        self.head = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=768, out_channels= 64, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3),
            torch.nn.Conv1d(in_channels=64, out_channels=8, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3),
            torch.nn.Flatten(),
            torch.nn.Dropout(p=0.5),
            #TODO do maxpool here?
            torch.nn.Linear(360,100),
            torch.nn.ReLU(),
            torch.nn.Linear(100,1)
        )
        
        
        # print('freezing')
        # for par in list(self.rodan.parameters()):
        #     par.requires_grad = False
        print('FREEZING', fr_layers, 'layers')
        if(freeze):
            freeze_rodan(self, freeze_till=fr_layers, verbose=1)
        
        
        print('Enabling gradients')
        torch.set_grad_enabled(True)
        
        
        
        self.lr = lr
        self.acc = torchmetrics.classification.Accuracy(task="binary")
        # self.cm = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=num_classes)
        # print('weighting BCE loss!')
        # self.ce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([0.5]))
        # self.ce = torch.nn.BCEWithLogitsLoss()
        
        #TODO loss correction method - custom loss?
        # self.loss_correction_matrix_inv = inv(np.array([[1.0,0.0],[0.8,0.2]]))
        #TODO try inverse again? (was converted to long before - messed up)
        # im = torch.tensor(np.array([[1.0,0.0],[0.8,0.2]]), requires_grad=False).float().cuda()#to(self.device)

#         def binary_cross_entropy_corrected(pred, y): 
#             pred = torch.sigmoid(pred).flatten()
#             batch_size = y.size()[0]
#             y = y.long().flatten()
#             #TODO check 1d tensor * 1d tensor is 1d tensor
#             pos_part = pred.log()*im[np.ones(batch_size, dtype=np.int32),y]#y
#             neg_part = (1-pred).log()*im[np.zeros(batch_size, dtype=np.int32),y]#(1-y)
#             loss = -(pos_part + neg_part)
#             loss = torch.clamp(loss, min=-100, max=100)
#             return loss.mean()
        

        if(corr_loss):
            print('using corrected BCE loss!')
            T = torch.tensor([[1.0, 0.0],
                  [0.8, 0.2]]).cuda()
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
        # out = torch.nn.functional.relu(feature_vector)
        # out = self.conv_1(out)
        # out = torch.nn.functional.relu(out)
        # out = self.maxpool(out)
        # out = self.conv_2(out)
        # out = torch.nn.functional.relu(out)
        # out = self.maxpool(out)
        # out = self.flatten(out)
        # # out = self.dropout(out)
        # out = self.linear_1(out)
        # out = torch.nn.functional.relu(out)
        # out = self.linear_2(out)
        return out
        
        # print(feature_vector.size()) #64,768,420
        # feature_vector = feature_vector.permute(0,2,1) #64,420,768
        # out = self.linear_1(feature_vector) #64, 420, 8
        # out = torch.nn.ReLU(out) #64, 420, 8
        # out = out.permute(0,2,1)
        # out = self.conv_1(out)
        
        
        
        # mod_pred = self.dropout(mod_pred)
        # final_pred, _ = torch.max(mod_pred, dim=-2)
        # return final_pred #no need for sigmoid, i use logits BCE
    
    
    def configure_optimizers(self):
        optimizer =  torch.optim.AdamW(params=[{'params':self.trainable_rodan.parameters()},{'params':self.head.parameters()}], lr=self.lr, weight_decay=0.01)
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
        x,y,exp = val_batch
        loss = self.compute_loss(x, y, exp, 'valid')
        self.log('valid_loss', loss, on_epoch=True) #ADDED on_epoch=True
        # return logs
    
    def predict_step(self, batch, batch_idx):
        xs, ids = batch
        logits = self.forward(xs)
        res = torch.sigmoid(logits)
        return res, ids
    
    def compute_loss(self, x, y, exp, loss_type):
        seq_predictions_logits = self(x)
        mod_loss = self.ce(seq_predictions_logits, y)
        
        is_predicted_modified = torch.sigmoid(seq_predictions_logits)
        acc = self.acc(is_predicted_modified, y)
        self.log(f'{loss_type} acc', acc, on_epoch=True)
        
#         cms = {}
        exps = np.array(exp)
        for e in np.unique(exp):
            indicies = exps == e
            batch_size = sum(indicies)
            if(batch_size>0):
                # sub_cm = self.cm(is_predicted_modified[exps==e], torch.squeeze(torch.nn.functional.one_hot(y.long()[exps==e], num_classes=2)))
#                 cms[e] = sub_cm
                self.log(f'{loss_type} {e} acc', self.acc(is_predicted_modified[exps==e], y[exps==e]), on_epoch=True)
        return mod_loss
        
#         return {'loss':mod_loss, 'cms':cms}
    
#     def validation_epoch_end(self, outputs):
#         agg_cms = {}
        
#         for log in outputs:
#             cm_dict = log['cms']
#             for exp, cm in cm_dict.items():
#                 if(exp not in agg_cms.keys()):
#                     agg_cms[exp] = [cm.detach().cpu().numpy()]
#                 else:
#                     agg_cms[exp].append(cm.detach().cpu().numpy())
#         for k,v in agg_cms.items():
#             agg_cms[k] = sum(v)
        
#         for exp in agg_cms.keys():
#             self.logger.experiment.log_confusion_matrix(matrix=agg_cms[exp], file_name=f"{exp}.json")
            
            
def freeze_rodan(model, freeze_till, verbose=1):
    #freeze_till max is 21
    for name, module in model.named_modules():
        if(name in ['','trainable_rodan', 'trainable_rodan.convlayers']):
            continue
        pattern = r"conv\d+"
        match = re.search(pattern, name)
        if(match):
            conv_index = int(match.group(0)[4:])
            if(conv_index > freeze_till):
                print('breaking')
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