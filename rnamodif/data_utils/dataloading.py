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

                     
def get_my_valid_dataset_unlimited(window=1000, pos_files = 'pos_2022', neg_files='neg_2022', split_method=get_default_split, verbose=1, read_blacklist=[]):
    #IN PROGRESS
    split = split_method(pos_files=pos_files, neg_files=neg_files)
    
    train_pos_files = split['train_pos_files']
    train_neg_files = split['train_neg_files']
    valid_pos_files = split['valid_pos_files']
    valid_neg_files = split['valid_neg_files']
    
    print('valid files indicies')
    if(verbose==1):
        for files in [valid_pos_files, valid_neg_files]:
            print(sorted([int(Path(x).stem.split('_')[-1]) for x in files]))
    
    
    def process_fast5_read_full(read, window):
        """ Normalizes and extracts specified region from raw signal """

        s = read.get_raw_data(scale=True)  # Expensive
        s = stats.zscore(s)

        #Using custom trim arguments according to Explore notebook
        # skip, _ = primer_trim(signal=s[:26000])
        skip = primer_trim(signal=s[:26000])
        

        last_start_index = len(s)-window
        if(last_start_index < skip):
            # if sequence is not long enough, last #window signals is taken, ignoring the skip index
            skip = last_start_index
        s = s[skip:]
        
        return s
    
    def myite_full(files, label, window):
        for fast5 in files:
            # yield (Path(fast5).stem + f'_lab_{label}')
            if(verbose==1):
                print(Path(fast5).stem,'starting', 'label', label)
            with get_fast5_file(fast5, mode='r') as f5:
                for i, read in enumerate(f5.get_reads()):
                    if(read.read_id in read_blacklist):
                        # print('skipping')
                        continue
                    # print(Path(fast5).stem, 'READ',i, 'TODO DELETE PRINT')
                    x = process_fast5_read_full(read, window)
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

