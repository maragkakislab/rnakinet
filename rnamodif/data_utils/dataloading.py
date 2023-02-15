from pathlib import Path
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
import torch
import math
from rnamodif.data_utils.trimming import primer_trim
from rnamodif.data_utils.read_utils import process_read
from rnamodif.data_utils.generators import alternating_gen, uniform_gen, sequential_gen

def get_valid_dataset_unlimited(splits, window=1000, verbose=1, read_blacklist=[], normalization='rodan', trim_primer=False):
    def process_files_fully(files, exp, label, window, normalization, trim_primer):
        for fast5 in files:
            if(verbose==1):
                print(Path(fast5).stem,'starting', 'label', label)
            with get_fast5_file(fast5, mode='r') as f5:
                for i, read in enumerate(f5.get_reads()):
                    if(read_blacklist):
                        if(read.read_id in read_blacklist):
                            continue
                    x = process_read(read, window=None, normalization=normalization, trim_primer=trim_primer) #getting the full read and slicing later
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
        def __init__(self, positive_files, negative_files, window, normalization, trim_primer):
            self.positive_files = positive_files
            self.negative_files = negative_files
            self.window = window
            self.normalization=normalization
            self.trim_primer=trim_primer

        def __iter__(self):
            pos_gens = []
            for exp,files in self.positive_files:
                pos_gens.append(process_files_fully(files, exp, label=1, window=self.window, normalization=self.normalization, trim_primer=self.trim_primer))
            neg_gens = []
            for exp,files in self.negative_files:
                neg_gens.append(process_files_fully(files, exp, label=0, window=self.window, normalization=self.normalization, trim_primer=self.trim_primer))
            return sequential_gen(pos_gens+neg_gens)
    
    return FullDataset(pos_files, neg_files, window=window, normalization=normalization, trim_primer=trim_primer)


class FullTestDataset(IterableDataset):
        def __init__(self, files, window, normalization, trim_primer, read_limit, stride):
            self.files = files
            self.window = window
            self.normalization=normalization
            self.trim_primer=trim_primer
            self.read_limit = read_limit
            self.stride = stride
            
        def process_files_fully(self, files, window, normalization, trim_primer):
            for fast5 in files: 
                with get_fast5_file(fast5, mode='r') as f5:
                    for i, read in enumerate(f5.get_reads()):
                        x = process_read(read, window=None, normalization=normalization, trim_primer=trim_primer) 
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

        def __iter__(self):
            if(self.read_limit):
                iterator = self.process_files_fully(self.files, window=self.window, normalization=self.normalization, trim_primer=self.trim_primer)
                def limited_iterator():
                    reads_processed = 0
                    last_read = None
                    while True:
                        signal, identifier = next(iterator)
                        current_read = identifier['readid']
                        if(last_read and last_read!=current_read):
                            reads_processed+=1
                            if(reads_processed >= read_limit):
                                break
                        last_read=current_read
                        yield((signal, identifier))
                return limited_iterator()
                        
            return self.process_files_fully(self.files, window=self.window, normalization=self.normalization, trim_primer=self.trim_primer)
        
def get_test_dataset(files, window=1000, normalization='rodan', trim_primer=False, stride=1000, read_limit=None):
    return FullTestDataset(files, window=window, normalization=normalization, trim_primer=trim_primer, read_limit=read_limit, stride=stride)
