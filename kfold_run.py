from dataloading import get_my_dataset, get_kfold_split_func
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from bonito_pretrained import BonitoPretrained

#OBSOLETE
def run_kfold(total_k, current_k):
    print('TODO ADD WORKER PARALELIZATION')
    print(f'CURRENT FOLD {current_k}/{total_k}')
    experiment_name = f'bonito_2022data_kfold_{current_k}/{total_k}'

    model = BonitoPretrained(pretrained_lr=1e-4, my_layers_lr=1e-3, warmup_steps = 10000)

    train_dset, valid_dset = get_my_dataset(valid_limit=10000, window=1000, 
                                            pos_files = 'pos_2022', neg_files='neg_2022',
                                            split_method=get_kfold_split_func(total_k=total_k, current_k=current_k))

    train_loader = DataLoader(train_dset, batch_size=256, num_workers=32,
                              pin_memory=True, persistent_workers=True)
    valid_loader = DataLoader(valid_dset, batch_size=64, num_workers=32,
                              pin_memory=True, persistent_workers=True)


    checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints_pl/{experiment_name}", save_top_k=2, monitor="valid_loss")
    logger = CometLogger(api_key="TEVQbgxxvilM1WdTyqZLJ57ac", project_name='RNAModif', experiment_name=experiment_name)
    trainer= pl.Trainer(
        max_steps = 5000000, logger=logger, accelerator='gpu',
        auto_lr_find=False, val_check_interval=25000, #500 
        log_every_n_steps=25000, benchmark=True, precision=16,
        callbacks=[checkpoint_callback])#, profiler="simple")
    trainer.fit(model, train_loader, valid_loader)