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
from fast_ctc_decode import beam_search, viterbi_search
import distance


class RodanEvent(pl.LightningModule):
    #TODO remove vocab map for default?
    #TODO profile - is data loading bottleneck?
    def __init__(self, vocab_map, lr=5e-4, warmup_steps=1000):
        super().__init__()
        torchdict = torch.load('/home/jovyan/RNAModif/RODAN/rna.torch', map_location="cpu")
        origconfig = torchdict["config"]
        d = origconfig
        n = SimpleNamespace(**d)
        args = {
            'debug':False, #True prints out more info
            'arch':None,
        }
        self.rodan, device = load_model('/home/jovyan/RNAModif/RODAN/rna.torch', config=n, args=SimpleNamespace(**args))
        
        self.mod_neuron = torch.nn.Linear(768,1)
        self.lr = lr
        self.acc = torchmetrics.classification.BinaryAccuracy()
        self.warmup_steps = warmup_steps
            
        self.vocab_map = self.setup_vocab_map_for_ctc(vocab_map)
    
    def setup_vocab_map_for_ctc(self, vocab_map):
        new_vocab = {}
        new_vocab['N'] = 0 #unused
        for k,v in vocab_map.items():
            new_vocab[k] = vocab_map[k]+1
        new_vocab['X'] = 5 #5 because i torch.cat in forward, mod_pred is last place
        return new_vocab
        
    def forward(self, x):
        x = x.unsqueeze(dim=1) #introduce channel dim
        feature_vector = self.rodan.convlayers(x)
        feature_vector = feature_vector.permute(0,2,1)
        
        mod_pred = self.mod_neuron(feature_vector) #no need for sigmoid, i use logits BCE
        og_preds = self.rodan.final(feature_vector)
        
        bases_preds = torch.cat([og_preds, mod_pred], dim=2) #last number is mod_A prediction
        
        #TODO og rodan trained on novoa data - bad test set?
        bases_preds = torch.nn.functional.log_softmax(bases_preds, 2)
        bases_preds =  bases_preds.permute(1, 0, 2)
              
        return bases_preds
    
    
    def configure_optimizers(self):
        # + LR scheduling (i have custom head)
        #TODO i can do custom LR {'params': model.classifier.parameters(), 'lr': 1e-3}
        #TODO WD=0 for pretrained model?
        optimizer =  torch.optim.AdamW(params=[{'params':self.rodan.parameters()},{'params':self.mod_neuron.parameters()}], lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=self.warmup_steps)
        
        return [optimizer], [scheduler]
    
    def training_step(self, train_batch, batch_idx, dataloader_idx=None):
        # x,y,exp = train_batch
        x,y = train_batch
        
        output = self(x)
        loss = self.compute_loss(output, y, 'train')
        self.log('train_loss', loss)
        
        sch = self.lr_schedulers()
        sch.step()
        current_lr_list = sch.get_last_lr()
        pretrained_lr = current_lr_list[0]
        my_layer_lr = current_lr_list[0] #The same
        self.log('learning rate (pretrained part)', pretrained_lr)
        self.log('learning rate (custom part)', my_layer_lr)
        
        self.log_metrics(output,y, 'train')
        
        return loss
    
    def validation_step(self, val_batch, batch_idx, dataloader_idx=None):
        x,y  = val_batch
        output = self(x)
        loss = self.compute_loss(output, y, 'valid')
        self.log('valid_loss', loss) 
        
        self.log_metrics(output,y, 'valid')
        
    
    def log_metrics(self, output, y, loss_type):
        
        alphabet = ''.join(list(self.vocab_map.keys()))
        
        #TODO softmaxing already log_sofrmaxed output?? check beamsearch code
        permuted_output = torch.softmax(output.permute(1,0,2), dim=2).detach().cpu().numpy()
        batch_size = permuted_output.shape[0]
        
        correct = 0
        dist_percs = [] 
        for i in range(batch_size):
            seq, path = beam_search(permuted_output[i,:,:], alphabet=alphabet)
            mod_label = 'X' in y[i]
            mod_pred = 'X' in seq
            if(mod_label == mod_pred):
                correct+=1
                
            lev_distance = distance.levenshtein(y[i], seq)
            lev_distance_perc = lev_distance/len(y[i])
            dist_percs.append(lev_distance_perc)
        
                
        self.log(f'{loss_type} window accuracy', correct/batch_size)
        self.log(f'{loss_type} lev distance perc', sum(dist_percs)/batch_size)
        
    
    def compute_loss(self, output, y, loss_type):
        output_size = output.size()
        signal_len = output_size[0]
        batch_size = output_size[1]
        signal_lengths = torch.tensor(signal_len).repeat(batch_size)
        label_lengths = torch.tensor([len(label) for label in y])
        # numericalized_labels = torch.tensor([[self.vocab_map[char] for char in label] for label in y])
        numericalized_labels = torch.tensor([self.vocab_map[char] for label in y for char in label])
        loss = torch.nn.functional.ctc_loss(
            output, 
            numericalized_labels, 
            signal_lengths, 
            label_lengths, 
            reduction='mean', 
            blank=0 #label for the N character
        )
        
        return loss
    
        
