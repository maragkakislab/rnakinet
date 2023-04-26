import pytorch_lightning as pl
from torch import nn
import torch
from torch.nn import functional as F
import torchmetrics
from resnet1d.resnet1d import ResNet1D

class CNN(pl.LightningModule):
    def __init__(self, learning_rate, base_filters, kernel_size, n_block):
      super().__init__()
      self.learning_rate = learning_rate

      # PREIMPLEMENTED CNN
      preimplemented_net = ResNet1D(
        in_channels=1,
        base_filters=base_filters, #64
        # ratio=1.0,
        # filter_list = [64, 160, 160, 400, 400, 1024, 1024],
        # m_blocks_list = [2, 2, 2, 3, 3, 4, 4],
        kernel_size=kernel_size, #3
        stride=1,
        # groups_width=16,
        groups=1,
        n_block=n_block, #3#TODO?
        verbose=False,
        n_classes=1) 
      
      self.net = nn.Sequential(
          preimplemented_net,
          # nn.Sigmoid()
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

      self.acc = torchmetrics.Accuracy(task='binary')

    def forward(self, x):
      x = x.permute(0,1).unsqueeze(1)
      return self.net(x)

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
      acc =self.acc(torch.sigmoid(output), y.int())
      self.log('train acc', acc)
      return loss

    def validation_step(self, val_batch, batch_idx):
      x,y = val_batch
      output = self(x)
      # loss = F.binary_cross_entropy(output, y)
      loss = F.binary_cross_entropy_with_logits(output, y)
      self.log('valid_loss', loss)
      acc = self.acc(torch.sigmoid(output), y.int())
      self.log('valid acc', acc)

      
