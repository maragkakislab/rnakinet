from bonito_pulled.bonito.util import load_model
from bonito_pulled.bonito.util import __models__
import torch
from bonito_pulled.bonito.nn import Permute
import torchmetrics
import pytorch_lightning as pl
from torch.nn import functional as F


class BonitoPretrained(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        #TODO there are multiple models, this one is _fast
        dirname = __models__/'dna_r10.4.1_e8.2_fast@v3.5.1'
        model = load_model(
            dirname, 
            self.device, 
            weights=None, 
            half=False, 
            chunksize=None, 
            batchsize=None, 
            overlap=None, 
            quantize=False, 
            use_koi=False #True uses weird modules
        )
        

        # model = model.encoder[:-1] #SKipping the CRF encoder

        seq_model = torch.nn.Sequential(
            model,
            Permute((1,2,0)),
            #TODO maxpool make it dynamic for any seq length? not hardcoded 2000 for 10000 length
            torch.nn.MaxPool1d(200), #maxpooling over the whole RNN length, instead do convolution maybe?
            torch.nn.Flatten(),
            #TODO activation?
            torch.nn.Linear(320, 1),
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
      optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0) #wd 0.01
      return optimizer
    
    def training_step(self, train_batch, batch_idx):
      x,y = train_batch
    
      output = self(x)
      # loss = F.binary_cross_entropy(output, y)
      loss = F.binary_cross_entropy_with_logits(output, y)
      self.log('train_loss', loss)
      acc =self.acc(output, y.int())
      self.log('train acc', acc)
      return loss
    
    def validation_step(self, val_batch, batch_idx):
      x,y = val_batch
      output = self(x)
      # loss = F.binary_cross_entropy(output, y)
      loss = F.binary_cross_entropy_with_logits(output, y)
      self.log('valid_loss', loss)
      acc = self.acc(output, y.int())
      self.log('valid acc', acc)