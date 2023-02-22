import random
import pytorch_lightning as pl
from torch.utils.data import IterableDataset, Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import torch
import numpy as np
from taiyaki.mapped_signal_files import HDF5Reader
from rnamodif.data_utils.workers import worker_init_event_batch_fn


class event_datamodule(pl.LightningDataModule):
    def __init__(self, pos_read_path, neg_read_path, window, valid_limit, batch_size=256, workers=1):
        super().__init__()
        self.pos_read_path = pos_read_path
        self.neg_read_path = neg_read_path
        self.window = window
        self.workers=workers
        self.batch_size = batch_size
        self.vocab_map = {'A':0,'C':1,'G':2,'T':3}
        self.valid_limit = valid_limit
        #TODO label smoothing loss ctc (rodan)
        
    def setup(self, stage=None):
        self.train_dataset = MyIterableMixedDataset(
            pos_reader = HDF5Reader(self.pos_read_path),
            neg_reader = HDF5Reader(self.neg_read_path),
            window = self.window,
            split='train',
        )
        self.valid_dataset = MyIterableMixedDataset(
            pos_reader = HDF5Reader(self.pos_read_path),
            neg_reader = HDF5Reader(self.neg_read_path),
            window = self.window,
            split='valid',
            limit=self.valid_limit,
        )
        
        assert self.train_dataset.vocab_map == self.vocab_map
        assert self.valid_dataset.vocab_map == self.vocab_map
        
        print('vocab ok')
        
            
    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.workers, pin_memory=True, worker_init_fn=worker_init_event_batch_fn)
        return train_loader
    
    def val_dataloader(self):
        val_loader =  DataLoader(self.valid_dataset, batch_size=self.batch_size)
        return val_loader
        
class MyIterableMixedDataset(IterableDataset):
    def __init__(self, pos_reader, neg_reader, split, window, limit=None):
        pos_dset = MyIterableDataset(pos_reader, window=window, split=split, replace_A_with_mod_A=True)
        neg_dset = MyIterableDataset(neg_reader, window=window, split=split)
        assert pos_dset.vocab_map == neg_dset.vocab_map
        self.vocab_map = pos_dset.vocab_map
        self.limit = limit
        self.pos_dset = pos_dset
        self.neg_dset = neg_dset
    
    def get_stream(self):
        pos_iterator = iter(self.pos_dset)
        neg_iterator = iter(self.neg_dset)
        if(self.limit):
            for _ in range(self.limit//2):
                yield(next(pos_iterator))
                yield(next(neg_iterator))
        else:
            while True:
                yield(next(pos_iterator))
                yield(next(neg_iterator))
    
    def __iter__(self):
        return self.get_stream()
    
class MyIterableDataset(IterableDataset):
    def __init__(self, hdf5_reader, window, split, replace_A_with_mod_A=False):
        self.reader = hdf5_reader
        self.read_ids = self.reader.get_read_ids()
        self.window = window
        self.split = split
        nums = self.reader.get_alphabet_information().collapse_labels
        letters = list(self.reader.get_alphabet_information().collapse_alphabet)
        self.vocab_map = {x:y for (x,y) in zip(letters,nums)}
        self.vocab_map_reversed = {v:k for k,v in self.vocab_map.items()}
        self.available_batch_names = self.reader.batch_names
        self.replace_A_with_mod_A = replace_A_with_mod_A
        
        #TODO measure sequence distance on inference (F18 CTC vid)
        
    def get_random_sample(self):
        mapping = random.choice(self.batch_mappings)
        signal = self.process_signal_mapping(mapping)
        
        start = random.randint(0, len(signal)-self.window)
        end = start+self.window
        
        window_positions = (start,end)
        ref_beg, ref_end = np.searchsorted(mapping.Ref_to_signal, window_positions)
        bases_count  = ref_end-ref_beg
        window_ref = mapping.Reference[ref_beg:ref_end]
        
        #TODO how rodan limits data? base_len > smth etc...
        is_ok = (len(window_ref) == bases_count) and bases_count > 0
        event = signal[start:end]
        sequence = [self.vocab_map_reversed[index] for index in window_ref]
        
        return event, sequence, is_ok
        
    def process_signal_mapping(self, mapping):
        #Taken from RODAN code
        signal = (mapping.Dacs + mapping.offset) * mapping.range / mapping.digitisation

        med = np.median(signal)
        mad = mapping.offset * np.median(abs(signal-med))
        signal = (signal - mapping.shift_frompA) / mapping.scale_frompA
        return signal
    
    def load_new_batch(self):
        new_batch_name = random.choice(self.available_batch_names)
        reads_batch = self.reader._load_reads_batch(new_batch_name)
        
        reads_in_batch = len(list(reads_batch.keys()))
        split_index = int(reads_in_batch*0.8)
        if(self.split == 'train'):
            self.batch_mappings = list(reads_batch.values())[:split_index]
        if(self.split == 'valid'):
            self.batch_mappings = list(reads_batch.values())[split_index:]
        
        reads_batch_len = len(self.batch_mappings)
        # print('new batch', self.split, 'size', len(self.batch_mappings))
        
        #Sampling randomly for each element in batch before loading a new one
        self.batch_remaining_samples = reads_batch_len
    
    def get_stream(self):
        while True:
            if(len(self.available_batch_names)>1):
                if(self.batch_remaining_samples <=0):
                    self.load_new_batch()
                self.batch_remaining_samples -=1
            x,y, is_ok = self.get_random_sample()
            if(not is_ok):
                continue
            if(self.replace_A_with_mod_A):
                #TODO only half of them??
                y = ['X' if base=='A' else base for base in y]
            yield (np.array(x, dtype=np.float32),''.join(y))

    def __iter__(self):
        self.load_new_batch()
        return self.get_stream()
        

