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
        # print('rnn_pooler_shape')
        # print(x.shape)
        x = torch.swapaxes(x, 0,-1)
        
        max_pool = self.flatten(self.max_pool(x))
        # print(max_pool.shape)
        avg_pool = self.flatten(self.avg_pool(x))
        # print(avg_pool.shape)
        last = x[:,:,-1] #?? right dims?
        # print(last.shape)
        stack = torch.stack([max_pool, avg_pool, last]) #concating
        stack = torch.swapaxes(stack, 0, -1)
        # print(stack.shape)
        # stack = self.flatten(stack)
        return stack
        

class BonitoPretrained(pl.LightningModule):
    def __init__(self, learning_rate=2e-3, warmup_steps = 100000):
        #LR default 2e-3, doc 5e-4
        super().__init__()
        #TODO there are multiple models, this one is _fast
        
        # dna_r10.4_e8.1_sup@v3.4 , 5e-4 #OFFICIAL PRETRAINED ARGS 
        #TODO load pretrained LR scheduler
        dirname = __models__/'dna_r10.4.1_e8.2_fast@v3.5.1'
        # dirname = __models__/'dna_r10.4_e8.1_sup@v3.4'
        
        self.warmup_steps = warmup_steps
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
            #TODO activation function? Is RNN output activated by relu or smth?
            RNNPooler(features_to_pool=320, seq_len=200),
            #TODO pooling the wrong dimension!!?? (length vs features)
            # torch.nn.MaxPool1d(167), #maxpooling over the whole RNN length, instead do convolution maybe? or maxpool + take last and first vectors
            torch.nn.Flatten(),
            #TODO activation?
            # torch.nn.Linear(320, 1), _fast model
            
            torch.nn.Linear(960, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100,1),
            
            #TODO sigmoid vs CE with logits
            
        )#.half()
        
        
        # ).to('cuda').half()
                # x = torch.rand(32,1,10000).to('cuda').half()
        # seq_model(x).shape
        self.model = seq_model
        
        self.learning_rate = learning_rate
        self.acc = torchmetrics.Accuracy()
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0) #wd 0.01
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=self.warmup_steps)

        return [optimizer], [scheduler]
    
    def training_step(self, train_batch, batch_idx):
        x,y = train_batch

        output = self(x)
        # loss = F.binary_cross_entropy(output, y)
        loss = F.binary_cross_entropy_with_logits(output, y)
        self.log('train_loss', loss)
        acc =self.acc(output, y.int())
        self.log('train acc', acc)

        sch = self.lr_schedulers()
        sch.step()
        
        self.log('learning rate (scheduled)', sch.get_last_lr()[0])

        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x,y = val_batch
        output = self(x)
        # loss = F.binary_cross_entropy(output, y)
        loss = F.binary_cross_entropy_with_logits(output, y)
        self.log('valid_loss', loss)
        acc = self.acc(output, y.int())
        self.log('valid acc', acc)