import pytorch_lightning as pl
from rnamodif.data_utils.datamap import experiment_files
from rnamodif.data_utils.split_methods import get_kfold_split_func, get_default_split
from rnamodif.data_utils.trimming import primer_trim
import random
from torch.utils.data import IterableDataset, Dataset
from itertools import cycle, islice
from torch.utils.data import DataLoader
from pathlib import Path
from ont_fast5_api.fast5_interface import get_fast5_file
from scipy import stats
import torch
import numpy as np
from rnamodif.data_utils.workers import worker_init_simple_fn, worker_init_fn
from rnamodif.data_utils.sorting import get_experiment_sort


class nanopore_datamodule(pl.LightningDataModule):
    def __init__(self, pos_files='pos_2022', neg_files='neg_2022', split_method=get_default_split, verbose=0, workers=32, batch_size=256, valid_limit=None, window=1000, read_blacklist=None):
        #TODO read blacklist -> rename to positives read blacklist (pos only so far)
        
        """
        verbose == 1 -> print file indicies
        verbose == 2 -> + print loading file numbers
        """
        super().__init__()
        
        self.workers = workers
        self.batch_size = batch_size
        self.valid_limit = valid_limit
        self.verbose = verbose
        self.window = window
        
        if(read_blacklist):
            self.process_files_fn = process_files_filtered(read_blacklist)
        else:
            self.process_files_fn = process_files
        
        split = split_method(pos_files=pos_files, neg_files=neg_files)
    
        train_pos_files = split['train_pos_files']
        train_neg_files = split['train_neg_files']
        random.shuffle(train_pos_files)
        random.shuffle(train_neg_files)
        
        valid_pos_files = split['valid_pos_files']
        valid_neg_files = split['valid_neg_files']
        
        if(verbose >= 1):
            print('valid files indicies')
            for files in [valid_pos_files, valid_neg_files]:
                print(sorted([get_experiment_sort(pos_files)(x) for x in files]))

            print('train files indicies')
            for files in [train_pos_files, train_neg_files]:
                print(sorted([get_experiment_sort(pos_files)(x) for x in files]))
    
        self.train_pos_files = train_pos_files
        self.train_neg_files = train_neg_files
        
        self.valid_pos_files = valid_pos_files
        self.valid_neg_files = valid_neg_files
        
    def setup(self, stage=None):
        if(stage == 'fit' or stage==None):
            self.train_dataset = MyIterableDatasetMixed(self.train_pos_files, self.train_neg_files, window=self.window, verbose=self.verbose, process_files_fn=self.process_files_fn)
            self.valid_dataset = MyMappedDatasetMixed(self.valid_pos_files, self.valid_neg_files, window=self.window, limit=self.valid_limit, verbose=self.verbose, process_files_fn=self.process_files_fn)
            
    def train_dataloader(self):
        #TODO worker file distribution is exhaustive check
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.workers, pin_memory=True, worker_init_fn=worker_init_fn)
        return train_loader
    
    def val_dataloader(self):
        val_loader =  DataLoader(self.valid_dataset, batch_size=self.batch_size)
        return val_loader
        

class MyIterableDatasetMixed(IterableDataset):
    def __init__(self, pos_files, neg_files, window, process_files_fn, verbose=0):
        self.positive_files = pos_files
        self.negative_files = neg_files
        self.window = window
        self.verbose = verbose
        self.process_files_fn = process_files_fn
   
    def get_stream(self):
        pos_gen = self.process_files_fn(files=self.positive_files, label=1, window=self.window, verbose=self.verbose)
        neg_gen = self.process_files_fn(files=self.negative_files, label=0, window=self.window, verbose=self.verbose)
        while True:
            if(torch.rand(1) > 0.5):
                yield next(pos_gen)
            else:
                yield next(neg_gen)
  
    def __iter__(self):
        return self.get_stream()
        
class MyMappedDatasetMixed(Dataset):
    def __init__(self, pos_files, neg_files, window, process_files_fn, limit=None, verbose=0):
        self.pos_files = pos_files
        self.neg_files = neg_files
        self.window = window
        self.limit = limit
        self.verbose = verbose
        self.process_files_fn = process_files_fn
        self.items = self.get_data()
   
    def get_data(self):
        #TODO check if i sample from all files equally
        items = []
        pos_gens = []
        for pos_file in self.pos_files:
            pos_gens.append(self.process_files_fn([pos_file], label=1, window=self.window, verbose=self.verbose))

        neg_gens = []
        for neg_file in self.neg_files:
            neg_gens.append(self.process_files_fn([neg_file], label=0, window=self.window, verbose=self.verbose))
            
        count = 0
        current_pos_gen = 0
        current_neg_gen = 0
        while (count < self.limit):
            if(count < self.limit//2):
                items.append(next(pos_gens[current_pos_gen]))
                count+=1
                current_pos_gen = (current_pos_gen+1)%len(pos_gens)
            else:
                items.append(next(neg_gens[current_neg_gen]))
                count+=1
                current_neg_gen = (current_neg_gen+1)%len(neg_gens)
                
        return items     
  
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx]

def process_files(files, label, window, verbose):
    while True:
        #TODO shuffle?
        for fast5 in files:
            if(verbose ==2):
                print(f'{Path(fast5).stem}[-{label}-]')
            with get_fast5_file(fast5, mode='r') as f5:
                for read in f5.get_reads(): #Slow
                    x = process_read(read, window)
                    y = np.array(label)
                    #TODO put to tensors?
                    yield x.reshape(-1,1).swapaxes(0,1), np.array([y], dtype=np.float32)

#TODO remove duplicate
def process_files_filtered(blacklist):
    def fn_blacklist(files, label, window, verbose):
        while True:
            #TODO possible to optimize blacklist by storing file-wise reads (process files goes through 1 file at a time) and doing if read.id in filewise_reads_list
            for fast5 in files:
                if(verbose ==2):
                    print(f'{Path(fast5).stem}[-{label}-]')
                with get_fast5_file(fast5, mode='r') as f5:
                    for read in f5.get_reads(): #Slow
                        if(read.read_id in blacklist):
                            # print('skipping')
                            continue
                        x = process_read(read, window)
                        y = np.array(label)
                        #TODO put to tensors?
                        yield x.reshape(-1,1).swapaxes(0,1), np.array([y], dtype=np.float32)
    return fn_blacklist
                    
                    
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

