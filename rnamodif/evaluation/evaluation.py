import pytorch_lightning as pl
from torch.utils.data import DataLoader
from rnamodif.data_utils.dataloading import get_valid_dataset_unlimited
from rnamodif.data_utils.workers import worker_init_fn_multisplit, worker_init_simple_fn

def model_dataset_eval(model, dataset, workers):
    if workers>1:
        valid_loader = DataLoader(dataset, batch_size=32, num_workers=workers,
                                  pin_memory=True, worker_init_fn = worker_init_fn_multisplit)
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
        config['read_blacklist'] = None
    if('workers' not in config.keys()):
        config['workers'] = 1
        
    splits = config['split'](pos_files=config['pos_files'], neg_files=config['neg_files'])
    dset = get_valid_dataset_unlimited(
        splits=splits,
        window=config['window'],
        verbose=0,
        read_blacklist=config['read_blacklist'],
    )
    model = get_trained_model(config['arch'], config['checkpoint'])
    
    return model_dataset_eval(model, dset, config['workers'])

def run_test(dataset, checkpoint, workers, architecture, batch_size=32, profiler=None):
    test_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, pin_memory=True, worker_init_fn = worker_init_simple_fn) 
    # test_loader = DataLoader(dataset, batch_size=batch_size)
    
    model = architecture().load_from_checkpoint(checkpoint)
    # trainer = pl.Trainer(accelerator='gpu', profiler=profiler)
    trainer = pl.Trainer(accelerator='gpu', profiler=profiler, precision=16)
    
    predictions = trainer.predict(model, test_loader, return_predictions=True)
    return predictions