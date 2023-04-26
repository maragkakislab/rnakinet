import pytorch_lightning as pl
from torch import nn
import torch
from torch.nn import functional as F
import torchmetrics
from resnet1d.resnet1d import ResNet1D
from torchaudio.transforms import Spectrogram, MelSpectrogram
import numpy as np

class CNN2d(pl.LightningModule):
    def __init__(self, learning_rate):
      super().__init__()
      self.learning_rate = learning_rate


      # CUSTOM CNN
      self.spec = Spectrogram()
      self.net = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
        nn.MaxPool2d(kernel_size=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
        nn.MaxPool2d(kernel_size=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
        # nn.MaxPool2d(kernel_size=2),
        nn.ReLU(),
        nn.Flatten(),
        # nn.Dropout(0.5),
        nn.Linear(in_features=5888, out_features=100),
        nn.ReLU(),
        nn.Linear(in_features=100, out_features=1),
        nn.Sigmoid()  
      )

      self.acc = torchmetrics.Accuracy()

    def forward(self, x):
      x = self.spec(x)
      x = torch.log10(x)
      return self.net(x)

    def configure_optimizers(self):
      # print("LEARNING RATE:",self.learning_rate)
      optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.01) #wd 0.01
      return optimizer

    def training_step(self, train_batch, batch_idx):
      x,y, _ = train_batch
      output = self.forward(x)
      # loss = F.binary_cross_entropy(output, y)
      loss = F.binary_cross_entropy_with_logits(output, y)
      self.log('train_loss', loss)
      acc =self.acc(output, y.int())
      self.log('train acc', acc)
      return loss

    def validation_step(self, val_batch, batch_idx):
      x,y, _ = val_batch
      output = self.forward(x)
      # loss = F.binary_cross_entropy(output, y)
      loss = F.binary_cross_entropy_with_logits(output, y)
      self.log('valid_loss', loss)
      acc = self.acc(output, y.int())
      self.log('valid acc', acc)

      
