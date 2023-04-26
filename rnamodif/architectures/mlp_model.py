import pytorch_lightning as pl
from torch import nn
import torch
from torch.nn import functional as F
import torchmetrics

class MLP(pl.LightningModule):
    def __init__(self, learning_rate, input_size):
      super().__init__()
      self.learning_rate = learning_rate


      self.net = nn.Sequential(
          nn.Flatten(),
          nn.Linear(input_size,100),
          nn.ReLU(),
          nn.Linear(100,100),
          nn.ReLU(),
          nn.Linear(100,1),
      )


      self.acc = torchmetrics.Accuracy()

    def forward(self, x):
      return self.net(x)

    def configure_optimizers(self):
      # print("LEARNING RATE:",self.learning_rate)
      optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.01) #wd 0.01
      return optimizer

    def training_step(self, train_batch, batch_idx):
      x,y, _ = train_batch

      output = self.net(x)
      # loss = F.binary_cross_entropy(output, y)
      loss = F.binary_cross_entropy_with_logits(output, y)
      self.log('train_loss', loss)
      acc =self.acc(output, y.int())
      self.log('train acc', acc)
      return loss

    def validation_step(self, val_batch, batch_idx):
      x,y, _ = val_batch
      output = self.net(x)
      # loss = F.binary_cross_entropy(output, y)
      loss = F.binary_cross_entropy_with_logits(output, y)
      self.log('valid_loss', loss)
      acc = self.acc(output, y.int())
      self.log('valid acc', acc)