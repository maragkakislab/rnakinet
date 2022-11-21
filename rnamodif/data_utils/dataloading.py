from pathlib import Path
from Fast5Fetch.fast5fetch.fast5data import get_all_fast5s
from Fast5Fetch.fast5fetch.fast5data import xy_generator_many, skew_generators
from torch.utils.data import Dataset, IterableDataset
import numpy as np
import time
import sys
import random
import numpy as np
import multiprocessing as mp
from pathlib import Path
from ont_fast5_api.fast5_interface import get_fast5_file
from scipy import stats
from rnamodif.data_utils.datamap import experiment_files
from bonito_pulled.bonito.reader import trim
import torch
import math
from rnamodif.data_utils.trimming import primer_trim
from rnamodif.data_utils.split_methods import *
from rnamodif.data_utils.dataloading2 import process_read

                     
def get_my_valid_dataset_unlimited(window=1000, pos_files = 'pos_2022', neg_files='neg_2022', split_method=get_default_split, verbose=1, read_blacklist=[]):
    #IN PROGRESS
    split = split_method(pos_files=pos_files, neg_files=neg_files)
    
    train_pos_files = split['train_pos_files']
    train_neg_files = split['train_neg_files']
    valid_pos_files = split['valid_pos_files']
    valid_neg_files = split['valid_neg_files']
    
    def myite_full(files, label, window):
        for fast5 in files:
            if(verbose==1):
                print(Path(fast5).stem,'starting', 'label', label)
            with get_fast5_file(fast5, mode='r') as f5:
                for i, read in enumerate(f5.get_reads()):
                    if(read.read_id in read_blacklist):
                        continue
                    x = process_read(read, window=None) #getting the full read and slicing later
                    y = np.array(label)
                    for start in range(0, len(x), window)[:-1]: #Cutoff the last incomplete signal
                        stop = start+window
                        identifier = {'file':str(fast5),
                                      'readid':read.read_id,
                                      'read_index_in_file':i,
                                      'start':start,
                                      'stop':stop,
                                      'label':label
                                     }
                        
                        yield x[start:stop].reshape(-1,1).swapaxes(0,1), np.array([y], dtype=np.float32)
                        #TODO resolve returning identifier for labelcleaning
                        # yield x[start:stop].reshape(-1,1).swapaxes(0,1), np.array([y], dtype=np.float32), identifier
                        
                        
    
    def mixed_generator_full(positive_files, negative_files, window):
        pos_gen = myite_full(positive_files, 1, window)
        neg_gen = myite_full(negative_files, 0, window)
        for pos in pos_gen:
            yield pos
        for neg in neg_gen:
            yield neg
    
    class FullValidDataset(IterableDataset):
        def __init__(self, positive_files, negative_files, window):
            self.positive_files = positive_files
            self.negative_files = negative_files
            self.window = window

        def __iter__(self):
            return mixed_generator_full(self.positive_files, self.negative_files, self.window)
    
    return FullValidDataset(valid_pos_files, valid_neg_files, window=window)

