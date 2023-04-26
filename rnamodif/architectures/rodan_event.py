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
import itertools
import random


class RodanEvent(pl.LightningModule):
    def __init__(self, vocab_map={'A':0,'C':1,'G':2,'T':3}, lr=5e-4, warmup_steps=1000, loss_type, lossweight):
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
        
        # print('freezing')
        # for par in list(self.rodan.parameters()):
        #     par.requires_grad = False
        
        print('Enabling gradients')
        torch.set_grad_enabled(True)

        
        self.mod_neuron = torch.nn.Linear(768,1)
        #inicializing mod A weights to the same as A
        with torch.no_grad():
            self.mod_neuron.weight.copy_(self.rodan.final.weight.data[1,:].clone()) #A is 1st neuron (0 is blank)
            self.mod_neuron.bias.copy_(self.rodan.final.bias.data[1].clone())
        
        
        self.lr = lr
        self.acc = torchmetrics.classification.BinaryAccuracy()
        self.warmup_steps = warmup_steps
        self.lossweight = lossweight
    
        self.vocab_map = self.setup_vocab_map_for_ctc(vocab_map)
        self.alphabet = ''.join(list(self.vocab_map.keys()))
        
        loss_map = {
            'mil':self.compute_loss_MIL,
            'max':self.compute_loss_pusher,
            'mich':self.compute_loss_mich,
            'ctc_mix':self.compute_loss_mix_ctc,
            'ctc':self.compute_loss,
        }
        self.loss_compute = loss_map[loss_type]
            
    
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
        
        # bases_preds = torch.nn.functional.log_softmax(bases_preds, 2)
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
        x,y, exp = train_batch
        
        output = self(x)
        loss = self.loss_compute(output,y,'train')

        self.log('train_loss', loss)
        
        sch = self.lr_schedulers()
        sch.step()
        current_lr_list = sch.get_last_lr()
        pretrained_lr = current_lr_list[0]
        my_layer_lr = current_lr_list[0] #The same
        self.log('learning rate (pretrained part)', pretrained_lr)
        self.log('learning rate (custom part)', my_layer_lr)
        
        # print('skipping metrics')
        # self.log_metrics(output,y, exp, 'train')
        
        return loss
    
    def validation_step(self, val_batch, batch_idx, dataloader_idx=None):
        x,y, exp  = val_batch
        output = self(x)
        loss = self.loss_compute(output,y,'valid')
        self.log('valid_loss', loss) 
        self.log_metrics(output,y, exp, 'valid')
        
    def predict_step(self, batch, batch_idx):
        x,y,exp = batch
        output = self(x)
        return output, y
        
    
    def log_metrics(self, output, y, exp, loss_type):
        #TODO softmaxing already log_sofrmaxed output?? check beamsearch code
        #TODO bottleneck?
        permuted_output = torch.softmax(output.permute(1,0,2), dim=2).detach().cpu().numpy()
        
        self.compute_metrics(permuted_output, y, loss_type)
        
        exps = np.array(exp)
        for e in np.unique(exp):
            indicies = exps == e
            batch_size = sum(indicies)
            if(batch_size>0):
                self.compute_metrics(permuted_output[exps==e], np.array(y)[exps==e], loss_type, log_prefix=e)
                
    def compute_metrics(self, permuted_output, y, loss_type, log_prefix=''):
        batch_size = permuted_output.shape[0]
        correct = 0
        dist_percs = [] 
        
        mod_A_perc_pred_pos = []
        mod_A_perc_pred_neg = []
        for i in range(batch_size):
            seq, path = beam_search(permuted_output[i,:,:], alphabet=self.alphabet)
            #TODO how to compute X probability?
            mod_label = 'X' in y[i]
            mod_pred = 'X' in seq
            if(mod_label == mod_pred):
                correct+=1
            
            pred_x_count = seq.count('X')
            seq_a_count = y[i].count('X') + y[i].count('A')
            if(seq_a_count > 0):
                if(mod_label):
                    mod_A_perc_pred_pos.append(pred_x_count/seq_a_count)
                else:
                    mod_A_perc_pred_neg.append(pred_x_count/seq_a_count)
            
            lev_distance = distance.levenshtein(y[i], seq)
            lev_distance_perc = lev_distance/len(y[i])
            dist_percs.append(lev_distance_perc)

        self.log(f'{log_prefix} {loss_type} window accuracy', correct/batch_size)
        self.log(f'{log_prefix} {loss_type} lev distance perc', sum(dist_percs)/batch_size)
        if(len(mod_A_perc_pred_pos) > 0):
            self.log(f'{log_prefix} {loss_type} predicted mod perc pos', sum(mod_A_perc_pred_pos)/len(mod_A_perc_pred_pos))
        if(len(mod_A_perc_pred_neg) > 0):
            self.log(f'{log_prefix} {loss_type} predicted mod perc neg', sum(mod_A_perc_pred_neg)/len(mod_A_perc_pred_neg))  
        
    
    def compute_loss(self, output, y, loss_type):
        output = torch.nn.functional.log_softmax(output, 2)
        
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
    
    def compute_loss_MIL(self, output, y, loss_type):
        output = torch.nn.functional.log_softmax(output, 2)
        
        output_size = output.size()
        signal_len = output_size[0]
        batch_size = output_size[1]
        
        signal_lengths = torch.tensor(signal_len).repeat(batch_size)
        label_lengths = torch.tensor([len(label) for label in y])
        
        pos_indicies = [i for i, label in enumerate(y) if 'X' in label]
        neg_indicies = [i for i, label in enumerate(y) if 'X' not in label]

        negative_samples = output[:,neg_indicies,:]
        negative_labels = np.array(y)[neg_indicies]
        neg_numericalized_labels = torch.tensor([self.vocab_map[char] for label in negative_labels for char in label])
        neg_loss = torch.nn.functional.ctc_loss(
            negative_samples, 
            neg_numericalized_labels, 
            signal_lengths[neg_indicies], 
            label_lengths[neg_indicies], 
            reduction='mean', 
            blank=0 #label for the N character
        )

        pos_losses = []
        for pos_id in pos_indicies:
            preds_logits = output[:,[pos_id],:]
            label = y[pos_id]

            sample_losses = []
            for var_label in generate_variations(label.replace('X','A'), limit=100):
                num_label = torch.tensor([self.vocab_map[char] for char in var_label])
                pos_sample_loss = torch.nn.functional.ctc_loss(
                    preds_logits,
                    num_label,
                    signal_lengths[[pos_id]],
                    label_lengths[[pos_id]],
                    reduction='mean',
                    blank=0)
                sample_losses.append(pos_sample_loss)
            # min_loss = torch.min(torch.stack(sample_losses))
            # pos_losses.append(min_loss)
            mean_loss = torch.mean(torch.stack(sample_losses))
            pos_losses.append(mean_loss)
            
        pos_loss = torch.mean(torch.stack(pos_losses)) #TODO sum?
        
        self.log(f'{loss_type} neg ctc loss', neg_loss)
        self.log(f'{loss_type} pos ctc loss', pos_loss)  
        
        # output = torch.nn.functional.log_softmax(output, 2)
        # weights = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        
        # weights = torch.tensor([0.6, 0.0, 0.0, 0.0, 0.0, 0.4])
        weights = torch.tensor([0.3, 0.0, 0.0, 0.0, 0.0, 0.7])
        
        
        # TODO dont just take pos indicies? = loss disbalance between pos and neg?
        smooth_weight = 5
        label_smoothing_loss = smooth_weight * -((output[:,pos_indicies,:] * weights.to(output.device)).mean())
        
        self.log(f'{loss_type} smoothing loss', label_smoothing_loss) 
        
        # pos_weight = 2
        # pos_loss = pos_loss * pos_weight
        return neg_loss + pos_loss + label_smoothing_loss
    

    def compute_loss_mich(self, output, y, loss_type):
        output_size = output.size()
        signal_len = output_size[0]
        batch_size = output_size[1]
        signal_lengths = torch.tensor(signal_len).repeat(batch_size)
        label_lengths = torch.tensor([len(label) for label in y])

        
        #TODO X of A for positive sequences?
        numericalized_labels = torch.tensor([self.vocab_map[char] if char != 'X' else self.vocab_map['A'] for label in y for char in label])
        # numericalized_labels = torch.tensor([self.vocab_map[char] for label in y for char in label])
        
        #TODO other losses - doesnt do softmax(log(x)) but log(softmax(x)) !!!
        output_logged = torch.nn.functional.log_softmax(output, 2)
        loss = torch.nn.functional.ctc_loss(
            output_logged, 
            numericalized_labels,
            signal_lengths, 
            label_lengths, 
            reduction='mean', 
            blank=0 #label for the N character
        )
        
        # self.vocab_map = {'A':0,'C':1,'G':2,'T':3}
        
        print('TODO REDO TO BOOLS!!! DOESNT WORK WITH 0 and 1')
        pos_indicies = np.array([1 if 'X' in l else 0 for i,l in enumerate(y)])
        neg_indicies = 1-pos_indicies
        
        
        #TODO argmax only places where A or X are predicted, ignore CTG
        pos_A_probs = torch.mean(torch.exp(output_logged[:,pos_indicies,1])) #A probs
        pos_X_probs = torch.mean(torch.exp(output_logged[:,pos_indicies,5])) #X probs
        neg_A_probs = torch.mean(torch.exp(output_logged[:,neg_indicies,1])) #A probs
        neg_X_probs = torch.mean(torch.exp(output_logged[:,neg_indicies,5])) #X probs
        
        probs = torch.tensor([pos_A_probs, pos_X_probs, neg_A_probs, neg_X_probs]).to(self.device)
        targets = torch.tensor([0.7, 0.3, 1.0, 0.0]).float().to(self.device)
        mse_loss = torch.nn.functional.mse_loss(probs, targets, reduction='mean')
        mse_loss = self.lossweight*mse_loss
        
        
        #LABEL SMOOTHING
        #high weight = enforce probabilities here
        # weights = torch.tensor([0.8, 0.2, 0.0, 0.0, 0.0, 0.0])
        weights = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        #TODO dont just take pos indicies? = loss disbalance between pos and neg?
        smooth_weight = 5
        label_smoothing_loss = smooth_weight * -((output_logged[:,pos_indicies,:] * weights.to(output_logged.device)).mean())
        #TODO put As everwhere + enforce X with label smoothing
        
        # label_smoothing_loss = smooth_weight * -((output_logged[:,pos_indicies,:] * weights.to(output_logged.device)).mean())
        # ind = torch.argmax(output[:,pos_indicies,:], dim=-1) == 5 #where is X predicted
        # probs_to_be_pushed = output[ind][:,1] #A probs
        # push_towards = torch.ones_like(probs_to_be_pushed).to(probs_to_be_pushed.device)
        # torch.nn.binary_cross_entropy_with_logits(probs_to_be_pushed, push_towards)
        
        #TODO first A in a sequence push down if there is too many As
        
        
        self.log(f'{loss_type} ctc loss', loss)
        self.log(f'{loss_type} mse loss', mse_loss) 
        self.log(f'{loss_type} smoothing loss', label_smoothing_loss) 
        
        return loss + mse_loss #+ label_smoothing_loss
        # return loss + label_smoothing_loss
    
    
    def compute_loss_mix_ctc(self, output, y, loss_type):
        output_size = output.size()
        signal_len = output_size[0]
        batch_size = output_size[1]
        # signal_lengths = torch.tensor(signal_len).repeat(batch_size)
        # label_lengths = torch.tensor([len(label) for label in y])

        
        #TODO X of A for positive sequences?
        # numericalized_labels = torch.tensor([self.vocab_map[char] if char != 'X' else self.vocab_map['A'] for label in y for char in label])
        
        y = np.array(y)
        pos_indicies = np.array([1 if 'X' in l else 0 for i,l in enumerate(y)], dtype=bool)
        neg_indicies = np.array(1-pos_indicies, dtype=bool)
        
        # self.vocab_map = {'A':0,'C':1,'G':2,'T':3}
        
        numericalized_labels = torch.tensor([self.vocab_map[char] for label in y[neg_indicies] for char in label])
        output_logged = torch.nn.functional.log_softmax(output, 2)
        signal_lengths = torch.tensor(signal_len).repeat(sum(neg_indicies))
        label_lengths = torch.tensor([len(label) for label in y[neg_indicies]])
        # print(neg_indicies)
        # print(output_logged[:,neg_indicies,:].size())
        # print(numericalized_labels.size())
        # print(signal_lengths.size())
        # print(label_lengths.size())
        neg_loss = torch.nn.functional.ctc_loss(
            output_logged[:,neg_indicies,:], 
            numericalized_labels,
            signal_lengths, 
            label_lengths, 
            reduction='mean', 
            blank=0 #label for the N character
        )
        
        numericalized_labels = torch.tensor([self.vocab_map[char] if char != 'X' else self.vocab_map['A'] for label in y[pos_indicies] for char in label])
        #TODO dont sum X and A logits but average two CTC losses with As and Xs?
        output_merged = output[:,:,:5]
        output_merged[:,:,1] += output[:,:,5]
        output_logged = torch.nn.functional.log_softmax(output_merged, 2)
        signal_lengths = torch.tensor(signal_len).repeat(sum(pos_indicies))
        label_lengths = torch.tensor([len(label) for label in y[pos_indicies]])
        pos_loss = torch.nn.functional.ctc_loss(
            output_logged[:,pos_indicies,:], 
            numericalized_labels,
            signal_lengths, 
            label_lengths, 
            reduction='mean', 
            blank=0 #label for the N character
        )

        
        output_logged = torch.nn.functional.log_softmax(output, 2)
        #TODO argmax only places where A or X are predicted, ignore CTG
        # pos_A_probs = torch.mean(torch.exp(output_logged[:,pos_indicies,1])) #A probs
        pos_X_probs = torch.mean(torch.exp(output_logged[:,pos_indicies,5])) #X probs
        # neg_A_probs = torch.mean(torch.exp(output_logged[:,neg_indicies,1])) #A probs
        # neg_X_probs = torch.mean(torch.exp(output_logged[:,neg_indicies,5])) #X probs
        
        # probs = torch.tensor([pos_A_probs, pos_X_probs, neg_A_probs, neg_X_probs]).to(self.device)
        # targets = torch.tensor([0.7, 0.3, 1.0, 0.0]).float().to(self.device)
        targets = torch.tensor([0.3]).float().to(self.device)
        
        mse_loss = torch.nn.functional.mse_loss(pos_X_probs, targets, reduction='mean')
        mse_loss = self.lossweight*mse_loss
        
        
        #LABEL SMOOTHING
        #high weight = enforce probabilities here
        # weights = torch.tensor([0.8, 0.2, 0.0, 0.0, 0.0, 0.0])
        # weights = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        #TODO dont just take pos indicies? = loss disbalance between pos and neg?
        # smooth_weight = 10
        # label_smoothing_loss = smooth_weight * -((output_logged[:,pos_indicies,:] * weights.to(output_logged.device)).mean())
        #TODO put As everwhere + enforce X with label smoothing
        
        # label_smoothing_loss = smooth_weight * -((output_logged[:,pos_indicies,:] * weights.to(output_logged.device)).mean())
        # ind = torch.argmax(output[:,pos_indicies,:], dim=-1) == 5 #where is X predicted
        # probs_to_be_pushed = output[ind][:,1] #A probs
        # push_towards = torch.ones_like(probs_to_be_pushed).to(probs_to_be_pushed.device)
        # torch.nn.binary_cross_entropy_with_logits(probs_to_be_pushed, push_towards)
        
        #TODO first A in a sequence push down if there is too many As
        
        
        self.log(f'{loss_type} ctc loss', pos_loss+neg_loss)
        self.log(f'{loss_type} mse loss', mse_loss) 
        # self.log(f'{loss_type} smoothing loss', label_smoothing_loss) 
        
        return pos_loss + neg_loss + mse_loss #label_smoothing_loss
        
        # return pos_loss + neg_loss + mse_loss + label_smoothing_loss
        # return loss + label_smoothing_loss
    def compute_loss_pusher(self, output, y, loss_type):
        output_size = output.size()
        signal_len = output_size[0]
        batch_size = output_size[1]
        signal_lengths = torch.tensor(signal_len).repeat(batch_size)
        label_lengths = torch.tensor([len(label) for label in y])
        numericalized_labels = torch.tensor([self.vocab_map[char] if char != 'X' else self.vocab_map['A'] for label in y for char in label])
        pos_indicies = torch.tensor([1 if 'X' in l else 0 for i,l in enumerate(y)]).float().to(self.device)
        vals, _ = torch.max(output[:,:,-1], dim=0)
        loss = torch.nn.functional.ctc_loss(
            torch.nn.functional.log_softmax(output, 2), 
            numericalized_labels,
            signal_lengths, 
            label_lengths, 
            reduction='mean', 
            blank=0 #label for the N character
        )
        ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(vals, pos_indicies)
        self.log(f'{loss_type} ctc loss', loss)
        self.log(f'{loss_type} ce loss', ce_loss) 
        # return ce_loss
        return loss + ce_loss   
    
# RODAN CTC LOSS IF
# if args.labelsmoothing:
#     losses = ont.ctc_label_smoothing_loss(out, label, label_len, smoothweights)
#     loss = losses["ctc_loss"]
# else:
#     loss = torch.nn.functional.ctc_loss(out, label, event_len, label_len, reduction="mean", blank=config.vocab.index('<PAD>'), zero_infinity=True)

# def ctc_label_smoothing_loss(log_probs, targets, lengths, weights):
#     T, N, C = log_probs.shape
#     log_probs_lengths = torch.full(size=(N, ), fill_value=T, dtype=torch.int64)
#     loss = torch.nn.functional.ctc_loss(log_probs.to(torch.float32), targets, log_probs_lengths, lengths, reduction='mean', zero_infinity=True)
#     label_smoothing_loss = -((log_probs * weights.to(log_probs.device)).mean())
#     return {'loss': loss + label_smoothing_loss, 'ctc_loss': loss, 'label_smooth_loss': label_smoothing_loss}


def generate_variations(seq, limit):
    num_as = seq.count('A')
    num_replacements = int(0.33 * num_as) #33 percent modified TODO parametrize
    variations = set()
    variations = []
    
    idxs = [pos for pos, char in enumerate(seq) if char == 'A']
    # combos = list(itertools.combinations(range(num_as), num_replacements))
    combos = [[random.randint(0,num_as-1) for _ in range(num_replacements)] for _ in range(limit)]
    
    # random.shuffle(combos)
    for combo in combos:
        new_seq = list(seq)
        for order in combo:
            new_seq[idxs[order]] = 'X'
        variations.append(''.join(new_seq))

    return variations