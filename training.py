from cnn_model import CNN
from dataloading import get_demo_dataset
import multiprocessing
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger


model = CNN(1e-5)
cpu_cores = multiprocessing.cpu_count()
train_dset, valid_dset = get_demo_dataset(cpu_cores=cpu_cores-2) #cpu cores min 3

train_loader = DataLoader(train_dset, batch_size=128)#, num_workers = cpu_cores-2-4)
valid_loader = DataLoader(valid_dset, batch_size=128)#, num_workers=4)

logger = CometLogger(api_key="TEVQbgxxvilM1WdTyqZLJ57ac", project_name='RNAModif')
#TODO use benchmark option?
trainer= pl.Trainer(
    max_steps =100000, logger=logger, accelerator='gpu', 
    auto_lr_find=False, val_check_interval=250) #Accelerator
trainer.fit(model, train_loader, valid_loader)
