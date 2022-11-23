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
from rnamodif.data_utils.dataloading2 import process_read
from rnamodif.data_utils.generators import alternating_gen, uniform_gen, sequential_gen


def get_valid_dataset_unlimited(splits, window=1000, verbose=1, read_blacklist=[]):
    def process_files_fully(files, exp, label, window):
        for fast5 in files:
            if(verbose==1):
                print(Path(fast5).stem,'starting', 'label', label)
            with get_fast5_file(fast5, mode='r') as f5:
                for i, read in enumerate(f5.get_reads()):
                    if(read_blacklist):
                        if(read.read_id in read_blacklist):
                            continue
                    x = process_read(read, window=None) #getting the full read and slicing later
                    y = np.array(label)
                    for start in range(0, len(x), window)[:-1]: #Cutoff the last incomplete signal
                        stop = start+window
                        # identifier = {'file':str(fast5),
                        #               'readid':read.read_id,
                        #               'read_index_in_file':i,
                        #               'start':start,
                        #               'stop':stop,
                        #               'label':label
                        #              }
                        
                        yield x[start:stop].reshape(-1,1).swapaxes(0,1), np.array([y], dtype=np.float32), exp
                        #TODO resolve returning identifier for labelcleaning
                        # yield x[start:stop].reshape(-1,1).swapaxes(0,1), np.array([y], dtype=np.float32), identifier
    def keycheck(dictionary, key): #TODO remove duplicate
        return key in dictionary.keys() and len(dictionary[key]) > 0
    
    pos_files = [(s['exp'],s['valid_pos_files']) for s in splits if keycheck(s, 'valid_pos_files')]
    neg_files = [(s['exp'],s['valid_neg_files']) for s in splits if keycheck(s, 'valid_neg_files')]
    
    class FullDataset(IterableDataset):
        def __init__(self, positive_files, negative_files, window):
            self.positive_files = positive_files
            self.negative_files = negative_files
            self.window = window

        def __iter__(self):
            pos_gens = []
            for exp,files in self.positive_files:
                pos_gens.append(process_files_fully(files, exp, label=1, window=self.window))
            neg_gens = []
            for exp,files in self.negative_files:
                neg_gens.append(process_files_fully(files, exp, label=0, window=self.window))
            return sequential_gen(pos_gens+neg_gens)
    
    return FullDataset(pos_files, neg_files, window=window)

        