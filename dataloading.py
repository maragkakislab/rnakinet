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


def get_demo_dataset(valid_limit=5000):
    #TODO currently using only one experiment, for real run use all of them
    pth = Path('../../meta/martinekv/store/seq/ont/experiments')
    fast5files_path_positives = Path(list((Path(list(pth.iterdir())[2])/'runs').iterdir())[0]/'fast5')
    fast5files_path_negatives = Path(list((Path(list(pth.iterdir())[0])/'runs').iterdir())[0]/'fast5')
    fast5s_positives = get_all_fast5s([fast5files_path_positives])
    fast5s_negatives = get_all_fast5s([fast5files_path_negatives])

    #TODO what window size ??? should be the whole read to not introduce more false positives?
    #TODO par = num_of_cpus
    #TODO last reads are typically longer = bias, validate on RANDOM files?
    both_generator_train = skew_generators(fast5s_positives[:-3], fast5s_negatives[:-3], 1, 0, window=1000, skew=0.5, par=1)
    both_generator_valid = skew_generators(fast5s_positives[-3:], fast5s_negatives[-3:], 1, 0, window=1000, skew=0.5, par=1)

    def mapper(ele):
        x,y = ele
        return x.reshape(-1,1).swapaxes(0,1), np.array([y], dtype=np.float32)

    class MyMixedDatasetTrain(IterableDataset):
        def __iter__(self):
            
            return map(mapper, both_generator_train)
        
    class MyMixedDatasetValid(Dataset):
        def __init__(self):
            self.valid_elements = []
            for _ in range(valid_limit):
                self.valid_elements.append(next(map(mapper,both_generator_valid)))
        
        def __len__(self):
            return len(self.valid_elements)
        
        def __getitem__(self, index):
            return self.valid_elements[index]


    
    return MyMixedDatasetTrain(), MyMixedDatasetValid()





def process_fast5_read(read, window, skip=1000, zscore=True):
    """ Normalizes and extracts specified region from raw signal """

    s = read.get_raw_data(scale=True)  # Expensive
    
    #TODO check normalization
    if zscore:
        s = stats.zscore(s)
    #TODO why do we skip the first 1000 signals?
    pos = random.randint(skip, len(s)-window)

    return s[pos:pos+window].reshape((window, 1))

       
def myite(files, label, window):
    for fast5 in files:
        #TODO now i only shuffle files and then read one to the end - randomize properly?
        with get_fast5_file(fast5, mode='r') as f5:
            for read in f5.get_reads():
                x = process_fast5_read(read, window)
                y = np.array(label)
                yield x.reshape(-1,1).swapaxes(0,1), np.array([y], dtype=np.float32)
    
def myUltimateIte(positive_files, negative_files, window):
        pos_gen = myite(positive_files, 1, window)
        neg_gen = myite(negative_files, 0, window)
        while True:
            if random.random()<0.5:
                # print('yielding positive\n')
                yield next(pos_gen)
            # print('yielding negative\n')
            yield next(neg_gen)
    
class MyMixedDatasetTrain(IterableDataset):
    def __init__(self, positive_files, negative_files, window):
        #TODO shuffle files
        self.positive_files = positive_files
        self.negative_files = negative_files
        self.ratio = 0.5
        self.pos_gen = myite(positive_files, 1, window)
        self.neg_gen = myite(negative_files, 0, window)
        self.ultimate = myUltimateIte(positive_files, negative_files, window)
    
    def __iter__(self): #former __next__
        return self.ultimate
            
    
                    
                    
def get_my_dataset():
    pth = Path('../../meta/martinekv/store/seq/ont/experiments')
    fast5files_path_positives = Path(list((Path(list(pth.iterdir())[2])/'runs').iterdir())[0]/'fast5')
    fast5files_path_negatives = Path(list((Path(list(pth.iterdir())[0])/'runs').iterdir())[0]/'fast5')
    fast5s_positives = get_all_fast5s([fast5files_path_positives])
    fast5s_negatives = get_all_fast5s([fast5files_path_negatives])
    train = MyMixedDatasetTrain(fast5s_positives[:-3], fast5s_negatives[:-3], 1000)
    valid = MyMixedDatasetTrain(fast5s_positives[-3:], fast5s_negatives[-3:], 1000)
    valid_limit = 1000
    class MyMixedDatasetValid(Dataset):
        def __init__(self):
            self.valid_elements = []
            for _ in range(valid_limit):
                self.valid_elements.append(next(iter(valid)))
        
        def __len__(self):
            return len(self.valid_elements)
        
        def __getitem__(self, index):
            return self.valid_elements[index]
    valid = MyMixedDatasetValid()
    return train, valid
    
    