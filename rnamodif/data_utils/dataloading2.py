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
from rnamodif.data_utils.workers import worker_init_simple_fn, worker_init_fn, worker_init_fn_multisplit
from rnamodif.data_utils.generators import alternating_gen, uniform_gen

class nanopore_datamodule(pl.LightningDataModule):
    def __init__(self, splits, verbose=0, workers=32, batch_size=256, valid_limit=None, window=1000, read_blacklist=None):
        #TODO read blacklist -> rename to positives read blacklist (pos only so far)
        
        super().__init__()
        
        self.workers = workers
        self.batch_size = batch_size
        self.valid_limit = valid_limit
        self.verbose = verbose
        self.window = window
        self.positives_blacklist = read_blacklist
        self.splits = splits
        
        
    def setup(self, stage=None):
        if(stage == 'fit' or stage==None):
            def keycheck(dictionary, key):
                return key in dictionary.keys() and len(dictionary[key]) > 0
            
            self.train_dataset = MyIterableDatasetMixed(
                pos_files = [(s['exp'],s['train_pos_files']) for s in self.splits if keycheck(s, 'train_pos_files')], 
                neg_files = [(s['exp'],s['train_neg_files']) for s in self.splits if keycheck(s, 'train_neg_files')], 
                window=self.window, 
                verbose=self.verbose, 
                blacklist=self.positives_blacklist
            )
            self.valid_dataset = MyMappedDatasetMixed(
                pos_files = [(s['exp'],s['valid_pos_files']) for s in self.splits if keycheck(s, 'valid_pos_files')], 
                neg_files = [(s['exp'],s['valid_neg_files']) for s in self.splits if keycheck(s, 'valid_neg_files')], 
                window=self.window, 
                limit=self.valid_limit, 
                verbose=self.verbose, 
                blacklist=self.positives_blacklist
            )
                
            
    def train_dataloader(self):
        #TODO worker file distribution is exhaustive check
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.workers, pin_memory=True, worker_init_fn=worker_init_fn_multisplit)
        return train_loader
    
    def val_dataloader(self):
        val_loader =  DataLoader(self.valid_dataset, batch_size=self.batch_size)
        return val_loader
        

class MyIterableDatasetMixed(IterableDataset):
    def __init__(self, pos_files, neg_files, window, verbose=0, blacklist=None):
        #TODO shuffle here because workers get low amount of files -> shuffle doesnt matter
        #Shuffle before dataset? Where does worker get its 'copy' ? MAke sure all workers have disjoint sets
        self.positive_files = pos_files #TODO rename? array_of_arrays_of_pos_files
        self.negative_files = neg_files
        self.window = window
        self.verbose = verbose
        self.blacklist = blacklist
   
    def get_stream(self):
        pos_gens = [
            process_files(
                files=f, 
                exp=exp,
                label=1, 
                window=self.window, 
                verbose=self.verbose, 
                shuffle=True, 
                blacklist=self.blacklist) for (exp,f) in self.positive_files
        ]
        neg_gens = [
            process_files(
                files=f, 
                exp=exp,
                label=0, 
                window=self.window, 
                verbose=self.verbose, 
                shuffle=True, 
                blacklist=self.blacklist) for (exp,f) in self.negative_files
        ]
        #Uniformly sampling from all splits across single label
        pos_gen = uniform_gen(pos_gens)
        neg_gen = uniform_gen(neg_gens)
        #Uniformly sampling from positives/negatives
        gen = uniform_gen([pos_gen, neg_gen])
        while True:
            yield next(gen)
  
    def __iter__(self):
        return self.get_stream()
        
class MyMappedDatasetMixed(Dataset):
    def __init__(self, pos_files, neg_files, window, limit=None, verbose=0, blacklist=None):
        self.pos_files = pos_files
        self.neg_files = neg_files
        self.window = window
        self.limit = limit
        self.verbose = verbose
        self.blacklist = blacklist
        self.items = self.get_data()
        
        
    def get_data(self):
        #TODO check if i sample from all files equally
        #TODO determinize shuffling for deterministic valid set
        
        pos_gens = []
        for exp,files in self.pos_files:
            for file in files:
                gen = process_files([file], exp=exp,label=1, window=self.window, verbose=self.verbose, blacklist=self.blacklist, shuffle=True)
                pos_gens.append(gen)
        
        neg_gens = []
        for exp,files in self.neg_files:
            for file in files:
                gen = process_files([file], exp=exp,label=0, window=self.window, verbose=self.verbose, blacklist=self.blacklist, shuffle=True)
                neg_gens.append(gen)
        
        pos_gen = alternating_gen(pos_gens)
        neg_gen = alternating_gen(neg_gens)
        gen = alternating_gen([pos_gen, neg_gen])
        
        items = []
        for _ in range(self.limit):
            items.append(next(gen))
            
        return items
  
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx]

def process_files(files, exp, label, window, verbose, shuffle, blacklist=None):
    while True:
        if(shuffle):
            random.shuffle(files)
        for fast5 in files:
            if(verbose ==2):
                print(f'{Path(fast5).stem}[-{label}-]')
            with get_fast5_file(fast5, mode='r') as f5:
                for read in f5.get_reads():
                    if(blacklist):
                        if(read.read_id in blacklist):
                            #TODO possible to optimize blacklist by storing file-wise reads (process files goes through 1 file at a time) and doing if read.id in filewise_reads_list
                            continue
                    x = process_read(read, window)
                    y = np.array(label)
                    #TODO put to tensors?
                    yield x.reshape(-1,1).swapaxes(0,1), np.array([y], dtype=np.float32), exp
                    
                    
def process_read(read, window):
    s = read.get_raw_data(scale=True)  # Expensive
    s = stats.zscore(s)

    skip = primer_trim(signal=s[:26000]) #TODO remove 26000 limit?
    
    if(not window):
        return s[skip:]
    
    last_start_index = len(s)-window
    if(last_start_index < skip):
        # if sequence is not long enough, last #window signals is taken, ignoring the skip index
        skip = last_start_index

    #Using torch rand becasue of multiple workers
    pos = torch.randint(skip, last_start_index+1, (1,))

    #TODO remove reshape
    return s[pos:pos+window].reshape((window, 1))

