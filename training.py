from cnn_model import CNN
import multiprocessing
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger
from dataloading import get_demo_dataset, get_my_dataset

train_dset, valid_dset = get_my_dataset()

model = CNN(1e-4)

#TODO more workers?
train_loader = DataLoader(train_dset, batch_size=256, num_workers=32,
                          pin_memory=True, persistent_workers=True)
valid_loader = DataLoader(valid_dset, batch_size=32, num_workers=32,
                          pin_memory=True, persistent_workers=True)

logger = CometLogger(api_key="TEVQbgxxvilM1WdTyqZLJ57ac", project_name='RNAModif')
trainer= pl.Trainer(
    max_steps = 1000000, logger=logger, accelerator='gpu', #max_epochs=-1,
    auto_lr_find=False, val_check_interval=500, log_every_n_steps=500, profiler="simple",benchmark=True) #try benchmark=tru
trainer.fit(model, train_loader, valid_loader)

