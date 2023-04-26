from bonito_pulled.bonito.util import load_model
from bonito_pulled.bonito.util import __models__
import torch
from bonito_pulled.bonito.nn import Permute
import torchmetrics
import pytorch_lightning as pl
from torch.nn import functional as F


class RNNPooler(torch.nn.Module):
    def __init__(self, features_to_pool, seq_len):
        super().__init__()
        self.max_pool = torch.nn.MaxPool1d(seq_len)
        self.avg_pool = torch.nn.AvgPool1d(seq_len)
        self.flatten = torch.nn.Flatten()
        
    def forward(self, x):
        x = torch.swapaxes(x, 0,-1)
        
        max_pool = self.flatten(self.max_pool(x))
        avg_pool = self.flatten(self.avg_pool(x))
        last = x[:,:,-1] #TODO check dims
        stack = torch.stack([max_pool, avg_pool, last]) #concating
        stack = torch.swapaxes(stack, 0, -1)
        return stack
        

class BonitoPretrained(pl.LightningModule):
    def __init__(self, pretrained_lr=5e-4, my_layers_lr=2e-3, warmup_steps = 10000):
        #LR default 2e-3, doc 5e-4
        super().__init__()
        #TODO there are multiple models, this one is _fast
        # dna_r10.4_e8.1_sup@v3.4 , 5e-4 #OFFICIAL PRETRAINED ARGS 
        dirname = __models__/'dna_r10.4.1_e8.2_fast@v3.5.1'
        # dirname = __models__/'dna_r10.4_e8.1_sup@v3.4'
        
        model = load_model(
            dirname, 
            self.device, 
            weights=None, #Default loadds from directory
            half=False, #half precision is handled by PL
            chunksize=1000,  #4000 default
            batchsize=64,  #64 default
            overlap=0,  #500 default
            quantize=False, #None/False default
            use_koi=False #True uses weird modules
        )

        # model = model.encoder[:-1] #SKipping the CRF encoder

        seq_model = torch.nn.Sequential(
            model,
            # Permute((1,2,0)), #TODO commented out for my pooler
            #TODO maxpool make it dynamic for any seq length? not hardcoded 2000 for 10000 length
            # torch.nn.MaxPool1d(200), _fast model
            RNNPooler(features_to_pool=320, seq_len=200),
            #TODO pooling the wrong dimension? (length vs features)
            #TODO why 320 features, crf should output 256?
            # torch.nn.MaxPool1d(167), #maxpooling over the whole RNN length, instead do convolution maybe? or maxpool + take last and first vectors
            torch.nn.Flatten(),
            #TODO activation?
            # torch.nn.Linear(320, 1), _fast model
            
            torch.nn.Linear(960, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100,1),
        )
        
        
        self.model = seq_model
        
        self.pretrained_layers_lr = pretrained_lr
        self.my_layers_lr = my_layers_lr
        self.warmup_steps = warmup_steps
        
        self.acc = torchmetrics.Accuracy()
        self.cm = torchmetrics.ConfusionMatrix(num_classes=2, normalize='true')
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        #different LR for my own layers (higher)
        my_layers_lr = self.my_layers_lr
        pretrained_layers_lr = self.pretrained_layers_lr
        lr_list = [pretrained_layers_lr, *[my_layers_lr for _ in range(5)]] #5 custom layers after pretrained model
        groups = [{'params': list(m.parameters()), 'lr': lr} for (m, lr) in zip(self.model.children(), lr_list)]
        optimizer = torch.optim.AdamW(groups, lr=self.pretrained_layers_lr, weight_decay=0.01)
        
        
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=self.warmup_steps)
        return [optimizer], [scheduler]
    
    def training_step(self, train_batch, batch_idx, dataloader_idx=None):
        x,y = train_batch
        output = self(x)
        loss = F.binary_cross_entropy_with_logits(output, y)
        self.log('train_loss', loss, on_epoch=True)
        self.log_metrics(output, y.int(), 'train')

        sch = self.lr_schedulers()
        sch.step()
        current_lr_list = sch.get_last_lr()
        pretrained_lr = current_lr_list[0]
        my_layer_lr = current_lr_list[-1] #the same for all custom layers
        self.log('learning rate (pretrained part)', pretrained_lr)
        self.log('learning rate (custom part)', my_layer_lr)

        return loss
    
    def validation_step(self, val_batch, batch_idx, dataloader_idx=None):
        x,y = val_batch
        output = self(x)
        loss = F.binary_cross_entropy_with_logits(output, y)
        self.log('valid_loss', loss, on_epoch=True) #ADDED on_epoch=True
        self.log_metrics(output, y.int(), 'valid')
    
    def log_metrics(self, output, labels, prefix):
        cm = self.cm(output, labels)
        true_negatives_perc = cm[0][0]
        false_negatives_perc = cm[0][1]
        true_positives_perc = cm[1][1]
        false_positives_perc = cm[1][0]
        self.log(f'{prefix} true_negatives_perc', true_negatives_perc, on_epoch=True)
        self.log(f'{prefix} false_negatives_perc', false_negatives_perc, on_epoch=True)
        self.log(f'{prefix} true_positives_perc', true_positives_perc, on_epoch=True)
        self.log(f'{prefix} false_positives_perc', false_positives_perc, on_epoch=True)
        
        acc = self.acc(output, labels)
        self.log(f'{prefix} acc', acc, on_epoch=True)
        
        
        
        