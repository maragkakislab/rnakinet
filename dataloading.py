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
from datamap import experiment_files
from bonito_pulled.bonito.reader import trim


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



def process_fast5_read(read, window, skip=1000, zscore=True, smartskip = True):
    """ Normalizes and extracts specified region from raw signal """

    s = read.get_raw_data(scale=True)  # Expensive
    
    if zscore:
        s = stats.zscore(s)
    
    #TODO trying out bonito skipping
    if(smartskip):
        skip, _ = trim(s[:8000])
        
    last_start_index = len(s)-window
    if(last_start_index < skip):
        print('SKIP too long', skip, last_start_index)
        # if sequence is not long enough, last #window signals is taken, ignoring the skip index
        skip = last_start_index
    pos = random.randint(skip, last_start_index)

    #TODO remove reshape
    return s[pos:pos+window].reshape((window, 1))

def myite(files, label, window):
    #TODO now i only shuffle files and then read one to the end - randomize properly?
    #I assume fast5 files have moreless random reads inside, randomizing across fast5file+read
    #Is quite expensive (opening and closing fast5 files constantly, iterating through reads until
    # index is reached...
    while True:
        #TODO why do random operations freeze trainer? shuffle in another way?
        # random.shuffle(files)
        for fast5 in files:
            with get_fast5_file(fast5, mode='r') as f5:
                for read in f5.get_reads():
                    x = process_fast5_read(read, window)
                    y = np.array(label)
                    yield x.reshape(-1,1).swapaxes(0,1), np.array([y], dtype=np.float32)
                
def myite_valid(files, label, window, limit):
    per_file_limit = limit//len(files)
    rem = limit%len(files) #even out remains after division
    for fast5 in files:
        # print(Path(fast5).stem,'starting')
        with get_fast5_file(fast5, mode='r') as f5:
            for i, read in enumerate(f5.get_reads()):
                if(i>=per_file_limit): 
                    # print(Path(fast5).stem, 'limit', per_file_limit,'iterated',i, 'labl',label)
                    if(rem > 0):
                        rem-=1
                    else:
                        break
                x = process_fast5_read(read, window)
                y = np.array(label)
                yield x.reshape(-1,1).swapaxes(0,1), np.array([y], dtype=np.float32)
                


def mixed_generator_valid(positive_files, negative_files, window, limit, valid_files_count):
    pos_gen = myite_valid(positive_files, 1, window, limit//2)
    neg_gen = myite_valid(negative_files, 0, window, limit//2)
    for i in range(limit):
        # 50/50 split for labels
        if i < limit//2:
            yield next(pos_gen)
        yield next(neg_gen)
            
def mixed_generator(positive_files, negative_files, window):
        pos_gen = myite(positive_files, 1, window)
        neg_gen = myite(negative_files, 0, window)
        while True:
            if random.random()<0.5:
                yield next(pos_gen)
            else:
                yield next(neg_gen)
    
class MyMixedDatasetTrain(IterableDataset):
    def __init__(self, positive_files, negative_files, window):
        self.mixed = mixed_generator(positive_files, negative_files, window)
    
    def __iter__(self):
        return self.mixed
    
class MyMixedDatasetValid(IterableDataset):
    def __init__(self, positive_files, negative_files, window, limit, valid_files_count):
        self.mixed = mixed_generator_valid(positive_files, negative_files, window, limit, valid_files_count)
    
    def __iter__(self):
        return self.mixed
    
                    
                    
def get_my_dataset(window=1000, pos_files = 'pos_2022', neg_files='neg_2022', valid_limit=1000, valid_files_count=3, valid_select_seed=42):
    pos_files = sorted(experiment_files[pos_files])
    neg_files = sorted(experiment_files[neg_files])
    
    #TODO previous validation = last 3 files
    seed = valid_select_seed
    deterministic_random = random.Random(seed)
    deterministic_random.shuffle(pos_files)
    deterministic_random.shuffle(neg_files)
    
    train_pos_files = pos_files[:-valid_files_count]
    train_neg_files = neg_files[:-valid_files_count]
    valid_pos_files = pos_files[-valid_files_count:]
    valid_neg_files = neg_files[-valid_files_count:]
    #always shuffling training data
    random.shuffle(train_pos_files)
    random.shuffle(train_neg_files)
    
    print('valid files indicies')
    for files in [valid_pos_files, valid_neg_files]:
        print(sorted([int(Path(x).stem.split('_')[-1]) for x in files]))
    
    # Check for deterministic valid selection and random training shuffling
    # def f(files, message):
    #     indicies = [int(Path(x).stem.split('_')[-1]) for x in files]
    #     print(message)
    #     print(indicies)
    # f(train_pos_files, 'train_positives')
    # f(train_neg_files, 'train_negatives')
    # f(valid_pos_files, 'valid_positives')
    # f(valid_neg_files, 'valid_negatives')
    
    train = MyMixedDatasetTrain(train_pos_files, train_neg_files, window=window)
    valid = MyMixedDatasetValid(valid_pos_files, valid_neg_files, window=window, limit=valid_limit, valid_files_count=valid_files_count)
    
    #TODO resolve stopIterationException after iterating throught the whole training dset
    #TODO resolve so i can use generator istead
    class MyMixedDatasetValidStatic(Dataset):
        def __init__(self,generator,limit):
            self.datapoints = []
            for i in range(limit):
                #TODO sample randomly, not from the beginning?
                self.datapoints.append(next(iter(generator)))
                
        def __getitem__(self,idx):
            return self.datapoints[idx]
        def __len__(self):
            return len(self.datapoints)
    
    return train, MyMixedDatasetValidStatic(valid, valid_limit)
    
    