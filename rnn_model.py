import pytorch_lightning as pl
from torch import nn
import torch
from torch.nn import functional as F
from torch.nn import RNN, GRU
import torchmetrics
from resnet1d.resnet1d import ResNet1D


class RNN(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
      super().__init__()
      self.learning_rate = learning_rate

      self.rnn = torch.nn.GRU(input_size=1, hidden_size=128, num_layers=3, dropout=0, batch_first=True)
      self.fc = torch.nn.Linear(128, 1)

      self.acc = torchmetrics.Accuracy()

    def forward(self, x):
      hidden = torch.zeros(3, x.size(0), 128, device=self.device).requires_grad_()
      x = torch.swapaxes(x,-1,-2)
      out, hidden =  self.rnn(x, hidden.detach())
      out = out[:,-1,:]
      return self.fc(out)
      

    def configure_optimizers(self):
      # print("LEARNING RATE:",self.learning_rate)
      optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.01) #wd 0.01
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

 