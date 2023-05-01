import pytorch_lightning as pl
import random
from torch.utils.data import IterableDataset, Dataset
from torch.utils.data import DataLoader
from pathlib import Path
from ont_fast5_api.fast5_interface import get_fast5_file
from scipy import stats
import torch
import numpy as np
from rnamodif.data_utils.generators import alternating_gen, uniform_gen
from rnamodif.data_utils.read_utils import process_read
from tqdm import tqdm
from rnamodif.data_utils.workers import worker_init_fn

class nanopore_datamodule_5eu(pl.LightningDataModule):
    def __init__(self, train_pos_files, train_neg_files, valid_exp_to_files_pos, valid_exp_to_files_neg, batch_size, window, per_dset_read_limit, shuffle_valid, workers):
        super().__init__()
        self.train_pos_files = train_pos_files
        self.train_neg_files = train_neg_files
        
        self.valid_exp_to_files_pos = valid_exp_to_files_pos
        self.valid_exp_to_files_neg = valid_exp_to_files_neg
        
        self.batch_size = batch_size
        self.window = window
        self.per_dset_read_limit = per_dset_read_limit
        self.shuffle_valid = shuffle_valid
        self.workers = workers
        
    def setup(self, stage=None):
        if(stage == 'fit' or stage==None):
            self.train_dataset = IterableDatasetMixed(
                pos_files = self.train_pos_files,
                neg_files = self.train_neg_files,
                window=self.window, 
            )
            
            self.valid_dataset = FullDataset(
                valid_exp_to_files_pos=self.valid_exp_to_files_pos, 
                valid_exp_to_files_neg=self.valid_exp_to_files_neg, 
                window=self.window, 
                stride=self.window, 
                per_dset_read_limit=self.per_dset_read_limit, 
                shuffle=self.shuffle_valid
            )
            
    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.workers, pin_memory=True, worker_init_fn=worker_init_fn)
        return train_loader
    
    def val_dataloader(self):
        val_loader =  DataLoader(self.valid_dataset, batch_size=self.batch_size)
        return val_loader
        

        
class IterableDatasetMixed(IterableDataset):
    def __init__(self, pos_files, neg_files, window, generator_type=uniform_gen):
        self.positive_files = pos_files
        self.negative_files = neg_files
        self.window = window
        self.generator_type = generator_type

    def process_files(self, files, label, window, exp):
        while True:
            # random.shuffle(files)
            fast5 = random.choice(files)
            # for fast5 in files:
            try:
                with get_fast5_file(fast5, mode='r') as f5:
                    for read in f5.get_reads():
                        x = process_read(read, window)
                        y = np.array(label)
                        if(len(x) == 0):
                            print('skipping')
                            continue

                        yield x.reshape(-1,1).swapaxes(0,1), np.array([y], dtype=np.float32), exp
            except OSError as error:
                print(error)
                continue

    def get_stream(self):
        #TODO remove exp from training loop to get rid of it
        pos_gen = self.process_files(files=self.positive_files, label=1, window=self.window, exp='pos')
        neg_gen = self.process_files(files=self.negative_files, label=0, window=self.window, exp='neg')
        gen = self.generator_type([pos_gen, neg_gen])
        while True:
            yield next(gen)

    def __iter__(self):
        return self.get_stream()        
        

class FullDataset(Dataset):
    def __init__(self, valid_exp_to_files_pos, valid_exp_to_files_neg, window, stride, per_dset_read_limit, shuffle):
        pos_gens = []
        for exp,files in valid_exp_to_files_pos.items():
            pos_gens.append(self.process_files_fully(files, exp, label=1, window=window, stride=stride, shuffle=shuffle))
        neg_gens = []
        for exp,files in valid_exp_to_files_neg.items():
            neg_gens.append(self.process_files_fully(files, exp, label=0, window=window, stride=stride, shuffle=shuffle))
            
        items = []
        print('Generating valid dataset')
        for gen in tqdm(pos_gens+neg_gens):
            reads_processed = 0
            last_read = None
            while True:
                x,y,identifier = next(gen)

                current_read = identifier['readid']
                if(last_read and last_read!=current_read):
                    reads_processed+=1
                    if(reads_processed>=per_dset_read_limit):
                        print(identifier['exp'], reads_processed)
                        break
                last_read = current_read
                items.append((x,y,identifier))

        self.items = items

    def process_files_fully(self, files, exp, label, window, stride, shuffle):
        for fast5 in files:
            if(shuffle):
                random.shuffle(files)
            with get_fast5_file(fast5, mode='r') as f5:
                for i, read in enumerate(f5.get_reads()):
                    x = process_read(read, window=None) #window = None for returning the whole signal
                    y = np.array(label)
                    for start in range(0, len(x), stride)[:-1]: #Cutoff the last incomplete signal
                        stop = start+window
                        identifier = {'file':str(fast5),
                                      'readid':read.read_id,
                                      'read_index_in_file':i,
                                      'start':start,
                                      'stop':stop,
                                      'label':label,
                                      'exp':exp,
                                     }

                        yield x[start:stop].reshape(-1,1).swapaxes(0,1), np.array([y], dtype=np.float32), identifier


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
            
    
class ExhaustiveDataset(IterableDataset):
        def __init__(self, files, window, stride):
            self.files = files
            self.window = window
            self.stride = stride
            
        def process_files_fully(self, files, window):
            for fast5 in files: 
                try:
                    with get_fast5_file(fast5, mode='r') as f5:
                        for i, read in enumerate(f5.get_reads()):
                            x = process_read(read, window=None) 
                            #TODO trim start of the read?
                            for start in range(0, len(x), self.stride):
                                stop = start+window
                                if(stop >= len(x)):
                                    continue
                                identifier = {'file':str(fast5),
                                              'readid':read.read_id,
                                              'read_index_in_file':i,
                                              'start':start,
                                              'stop':stop,
                                             }
                                yield x[start:stop].reshape(-1,1).swapaxes(0,1), identifier  
                except OSError as error:
                    print(error)
                    continue

        def __iter__(self):
            return self.process_files_fully(self.files, self.window)