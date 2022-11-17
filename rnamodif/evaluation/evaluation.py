import pytorch_lightning as pl
from torch.utils.data import DataLoader
from rnamodif.data_utils.dataloading import get_my_valid_dataset_unlimited
from rnamodif.data_utils.split_methods import get_fullvalid_split

def model_dataset_eval(model, dataset, multiprocessing):
    if multiprocessing:
        valid_loader = DataLoader(dataset, batch_size=128, num_workers=32,
                                  pin_memory=True, persistent_workers=True)
    else:
        valid_loader = DataLoader(dataset, batch_size=32) 
    
    trainer= pl.Trainer(accelerator='gpu', benchmark=True, precision=16)
    # trainer.test(model, valid_loader)
    metrics = trainer.validate(model, valid_loader)
    return metrics
    
def get_trained_model(architecture, checkpoint):
    return architecture().load_from_checkpoint(checkpoint)

def run_eval(config):
    if('read_blacklist' not in config.keys()):
        config['read_blacklist'] = []
    if('multiprocessing' not in config.keys()):
        config['multiprocessing'] = False
    #TODO remove need for splits
    dset = get_my_valid_dataset_unlimited(
        window=config['window'],
        pos_files=config['pos_files'],
        neg_files=config['neg_files'],
        split_method=config['split'],
        verbose=0,
        read_blacklist=config['read_blacklist'],
    )
    model = get_trained_model(config['arch'], config['checkpoint'])
    
    return model_dataset_eval(model, dset, config['multiprocessing'])
    