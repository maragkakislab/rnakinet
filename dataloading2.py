import pytorch_lightning as pl
from datamap import experiment_files
from data_utils.split_methods import get_kfold_split_func, get_default_split
from data_utils.trimming import primer_trim
import random
from torch.utils.data import IterableDataset
from itertools import cycle, islice
from torch.utils.data import DataLoader
from pathlib import Path
from ont_fast5_api.fast5_interface import get_fast5_file
from scipy import stats
import torch
import numpy as np
from data_utils.workers import worker_init_simple_fn

class nanopore_datamodule(pl.LightningDataModule):
    def __init__(self, pos_files='pos_2022', neg_files='neg_2022', split_method=get_default_split, verbose=True, workers=32, batch_size=256, valid_limit=None):
        super().__init__()
        
        self.workers = workers
        self.batch_size = batch_size
        self.valid_limit = valid_limit
        
        split = split_method(pos_files=pos_files, neg_files=neg_files)
    
        train_pos_files = split['train_pos_files']
        train_neg_files = split['train_neg_files']
        random.shuffle(train_pos_files)
        random.shuffle(train_neg_files)
        
        valid_pos_files = split['valid_pos_files']
        valid_neg_files = split['valid_neg_files']
        
        if(verbose):
            print('valid files indicies')
            for files in [valid_pos_files, valid_neg_files]:
                print(sorted([int(Path(x).stem.split('_')[-1]) for x in files]))

            print('train files indicies')
            for files in [train_pos_files, train_neg_files]:
                print(sorted([int(Path(x).stem.split('_')[-1]) for x in files]))
    
        self.train_pos_files = train_pos_files
        self.train_neg_files = train_neg_files
        
        self.valid_pos_files = valid_pos_files[0:1] #TODOOOOOOOOOOOOO REMOVE
        self.valid_neg_files = valid_neg_files[0:1]
        
    def setup(self, stage=None):
        if(stage == 'fit' or stage==None):
            self.train_dataset_pos = MyIterableDatasetSingle(self.train_pos_files, label=1, window=1000)
            self.train_dataset_neg = MyIterableDatasetSingle(self.train_neg_files, label=0, window=1000)
            
            self.valid_dataset = MyMappedDatasetMixed(self.valid_pos_files, self.valid_neg_files, window=1000, limit=self.valid_limit)
            
            # self.valid_dataset_pos = MyIterableDatasetSingle(self.valid_pos_files, label=1, window=1000, limit=self.valid_limit//2)
            # self.valid_dataset_neg = MyIterableDatasetSingle(self.valid_neg_files, label=0, window=1000, limit=self.valid_limit//2)
        
    def train_dataloader(self):
        #TODO worker file distribution is exhaustive check
        #TODO properly mix dataloaders, not 50/50 exactly
        workers_per_dataset = self.workers//2
        batch_size_per_dataset = self.batch_size//2
        
        pos_loader =  DataLoader(self.train_dataset_pos, batch_size=batch_size_per_dataset, num_workers=workers_per_dataset, pin_memory=True, worker_init_fn=worker_init_simple_fn)
        neg_loader =  DataLoader(self.train_dataset_neg, batch_size=batch_size_per_dataset, num_workers=workers_per_dataset, pin_memory=True, worker_init_fn=worker_init_simple_fn)
        
        return {'a':pos_loader,'b':neg_loader}
    
    
    def val_dataloader(self):
        # workers_per_dataset = self.workers//2
        # workers_per_dataset = 1 #TODO fix (loop error pytorch)
        # batch_size_per_dataset = self.batch_size//2
        
        val_loader =  DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=1, pin_memory=True)
        return val_loader
        
        
        # pos_loader =  DataLoader(self.valid_dataset_pos, batch_size=batch_size_per_dataset, num_workers=workers_per_dataset, pin_memory=True)
        # neg_loader =  DataLoader(self.valid_dataset_neg, batch_size=batch_size_per_dataset, num_workers=workers_per_dataset, pin_memory=True)
        # return [pos_loader, neg_loader]
        

class MyIterableDatasetSingle(IterableDataset):
    def __init__(self, files, label, window, limit=None):
        self.files = files
        self.label = label
        self.window = window
        self.limit = limit
   
    def get_stream(self):
        return cycle(process_files(files=self.files, label=self.label, window=self.window))
  
    def __iter__(self):
        return self.get_stream()

class MyMappedDatasetMixed(IterableDataset):
    def __init__(self, pos_files, neg_files, window, limit=None):
        self.pos_files = pos_files
        self.neg_files = neg_files
        self.window = window
        self.limit = limit
   
    def get_stream(self):
        pos_gen = process_files(pos_files, label=1, window=self.window)
        neg_gen = process_files(neg_files, label=0, window=self.window)
        count = 0
        while (count < self.limit):
            if(count < self.limit//2):
                yield next(pos_gen)
            else:
                yield next(neg_gen)
  
    def __iter__(self):
        return self.get_stream()

def process_files(files, label, window):
    for fast5 in files:
        # print(Path(fast5).stem, label)
        with get_fast5_file(fast5, mode='r') as f5:
            for read in f5.get_reads(): #Slow
                x = process_read(read, window)
                y = np.array(label)
                #TODO put to tensors?
                yield x.reshape(-1,1).swapaxes(0,1), np.array([y], dtype=np.float32)

def process_read(read, window):
    s = read.get_raw_data(scale=True)  # Expensive
    s = stats.zscore(s)

    skip = primer_trim(signal=s[:26000]) #TODO remove 26000 limit

    last_start_index = len(s)-window
    if(last_start_index < skip):
        skip = last_start_index

    #Using torch rand becasue of multiple workers
    pos = torch.randint(skip, last_start_index+1, (1,))

    #TODO remove reshape
    return s[pos:pos+window].reshape((window, 1))