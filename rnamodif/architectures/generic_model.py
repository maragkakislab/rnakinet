import pytorch_lightning as pl
from torch import nn
import torch
from torch.nn import functional as F
from torch.nn import RNN, GRU
import torchmetrics
from resnet1d.resnet1d import ResNet1D
from tst import Transformer
from transformer.src.benchmark import LSTM, BiGRU, ConvGru, FullyConv, FFN

class Transfo(pl.LightningModule):
    def __init__(self, learning_rate, skip_embedding):
        super().__init__()
        self.learning_rate = learning_rate

        self.net = Transformer(d_input=1, d_model=1, d_output=1,input_length=1000, q=8,v=8,h=8, N=4, attention_size=50, chunk_mode=None, pe='regular', skip_embedding=skip_embedding)

        
        # self.net = LSTM(input_dim=1, hidden_dim=64, output_dim=1, num_layers=3)
        
        # BiGRU(input_dim=1, hidden_dim=64, output_dim=1, num_layers=3)
        # ConvGru(input_dim=1, hidden_dim=64, output_dim=1, num_layers=3)
        # FullyConv(input_dim=1, hidden_dim=64, output_dim=1)
        # FFN(input_dim=1, hidden_dim=64, output_dim=1, num_layers=3)

        self.acc = torchmetrics.Accuracy()

    def forward(self, x):
        #TODOOOOOOOO Orignial code modifies - removed sigmoid
        #Transformer
        return self.net.forward(torch.swapaxes(x,-1,-2))
        
        #LSTM
        # out = self.net.forward(torch.swapaxes(x,-1,-2))
        # return out[:,-1,:]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x,y = train_batch
        output = self(x)
        loss = F.binary_cross_entropy_with_logits(output, y)
        self.log('train_loss', loss)
        acc =self.acc(output, y.int())
        self.log('train acc', acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x,y = val_batch
        output = self(x)
        loss = F.binary_cross_entropy_with_logits(output, y)
        self.log('valid_loss', loss)
        acc = self.acc(output, y.int())
        self.log('valid acc', acc)

        
class MyConvGru(pl.LightningModule):
    def __init__(self, learning_rate,bidirectional):
        super().__init__()
        self.learning_rate = learning_rate

        self.net = ConvGru(input_dim=1, hidden_dim=64, output_dim=1, num_layers=3, dropout=0, bidirectional=bidirectional, kernel_size=50)
        self.acc = torchmetrics.Accuracy()

    def forward(self, x):
        out = self.net.forward(torch.swapaxes(x,-1,-2))
        return out[:,-1,:]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x,y = train_batch
        output = self(x)
        loss = F.binary_cross_entropy_with_logits(output, y)
        self.log('train_loss', loss)
        acc =self.acc(output, y.int())
        self.log('train acc', acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x,y = val_batch
        output = self(x)
        loss = F.binary_cross_entropy_with_logits(output, y)
        self.log('valid_loss', loss)
        acc = self.acc(output, y.int())
        self.log('valid acc', acc)
