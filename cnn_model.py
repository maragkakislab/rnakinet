import pytorch_lightning as pl
from torch import nn
import torch
from torch.nn import functional as F
import torchmetrics
from resnet1d.resnet1d import ResNet1D

class CNN(pl.LightningModule):
    def __init__(self, learning_rate):
      super().__init__()
      self.learning_rate = learning_rate

      # PREIMPLEMENTED CNN
      preimplemented_net = ResNet1D(
        in_channels=1,
        base_filters=64, #64
        # ratio=1.0,
        # filter_list = [64, 160, 160, 400, 400, 1024, 1024],
        # m_blocks_list = [2, 2, 2, 3, 3, 4, 4],
        kernel_size=25, #3
        stride=1,
        # groups_width=16,
        groups=1,
        n_block=3, #TODO?
        verbose=False,
        n_classes=1) 
      
      self.net = nn.Sequential(
          preimplemented_net,
          nn.Sigmoid()
      )

      # CUSTOM CNN
      # self.net = nn.Sequential(
      #   nn.Conv1d(in_channels=5, out_channels=32, kernel_size=3),
      #   nn.MaxPool1d(kernel_size=5),
      #   nn.ReLU(),
      #   nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
      #   nn.MaxPool1d(kernel_size=5),
      #   nn.ReLU(),
      #   nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3),
      #   nn.MaxPool1d(kernel_size=5),
      #   nn.ReLU(),
      #   nn.Flatten(),
      #   nn.Dropout(0.5),
      #   nn.Linear(in_features=384, out_features=8),
      #   nn.Sigmoid()  
      # )

      self.acc = torchmetrics.Accuracy()

    def forward(self, x):
      return self.net(x)

    def configure_optimizers(self):
      # print("LEARNING RATE:",self.learning_rate)
      optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.01) #1e-3
      return optimizer

    def training_step(self, train_batch, batch_idx):
      x,y = train_batch
      output = self.net(x)
      loss = F.binary_cross_entropy(output, y)
      self.log('train_loss', loss)
      acc =self.acc(output, y.int())
      self.log('train acc', acc)
      return loss

    def validation_step(self, val_batch, batch_idx):
      x,y = val_batch
      output = self.net(x)
      loss = F.binary_cross_entropy(output, y)
      self.log('valid_loss', loss)
      acc = self.acc(output, y.int())
      self.log('valid acc', acc)

      
