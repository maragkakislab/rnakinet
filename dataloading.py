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
import torch
import math

def process_fast5_read(read, window, skip=1000, zscore=True, smartskip = True):
    """ Normalizes and extracts specified region from raw signal """

    s = read.get_raw_data(scale=True)  # Expensive
    
    if zscore:
        s = stats.zscore(s)
    
    #TODO trying out bonito skipping
    if(smartskip):
        # skip, _ = trim(s[:8000])
        
        #Using custom trim arguments according to Explore notebook
        skip, _ = my_trim(signal=s[:26000])
        
    last_start_index = len(s)-window
    if(last_start_index < skip):
        # print('SKIP too long', skip, last_start_index)
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
                


def mixed_generator_valid(positive_files, negative_files, window, limit):
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
    def __init__(self, positive_files, negative_files, window, limit):
        self.mixed = mixed_generator_valid(positive_files, negative_files, window, limit)
    
    def __iter__(self):
        return self.mixed


def get_default_split(pos_files, neg_files):
    valid_select_seed = 42
    valid_files_count = 10
    
    sortkey = lambda x: int(Path(x).stem.split('_')[-1])
    pos_files = sorted(experiment_files[pos_files], key=sortkey)
    neg_files = sorted(experiment_files[neg_files], key=sortkey)
    
    seed = valid_select_seed
    deterministic_random = random.Random(seed)
    deterministic_random.shuffle(pos_files)
    deterministic_random.shuffle(neg_files)
    
    train_pos_files = pos_files[:-valid_files_count]
    train_neg_files = neg_files[:-valid_files_count]
    valid_pos_files = pos_files[-valid_files_count:]
    valid_neg_files = neg_files[-valid_files_count:]
    
    return {
        'train_pos_files':train_pos_files,
        'train_neg_files':train_neg_files,
        'valid_pos_files':valid_pos_files,
        'valid_neg_files':valid_neg_files,
    }


def get_kfold_split_func(total_k, current_k, shuffle=True):
    def f(pos_files, neg_files):
        sortkey = lambda x: int(Path(x).stem.split('_')[-1])
        pos_files = sorted(experiment_files[pos_files], key=sortkey)
        neg_files = sorted(experiment_files[neg_files], key=sortkey)
        
        if(shuffle):
            seed = 42
            deterministic_random = random.Random(seed)
            deterministic_random.shuffle(pos_files)
            deterministic_random.shuffle(neg_files)
        
        pos_k_size = len(pos_files)//total_k
        neg_k_size = len(neg_files)//total_k
        
        valid_pos_files = pos_files[pos_k_size*current_k:pos_k_size*(current_k+1)]
        valid_neg_files = neg_files[neg_k_size*current_k:neg_k_size*(current_k+1)]
        
        train_pos_files = pos_files[:pos_k_size*current_k] + pos_files[pos_k_size*(current_k+1):]
        train_neg_files = neg_files[:neg_k_size*current_k] + neg_files[neg_k_size*(current_k+1):]
        
        return {
            'train_pos_files':train_pos_files,
            'train_neg_files':train_neg_files,
            'valid_pos_files':valid_pos_files,
            'valid_neg_files':valid_neg_files,
        }
        
        
    return f
        
def get_my_dataset(window=1000, pos_files = 'pos_2022', neg_files='neg_2022', valid_limit=1000, split_method=get_default_split):
    split = split_method(pos_files=pos_files, neg_files=neg_files)
    
    train_pos_files = split['train_pos_files']
    train_neg_files = split['train_neg_files']
    valid_pos_files = split['valid_pos_files']
    valid_neg_files = split['valid_neg_files']
    
    #always shuffling training data
    random.shuffle(train_pos_files)
    random.shuffle(train_neg_files)
    
    print('valid files indicies')
    for files in [valid_pos_files, valid_neg_files]:
        print(sorted([int(Path(x).stem.split('_')[-1]) for x in files]))
        
    print('train files indicies')
    for files in [train_pos_files, train_neg_files]:
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
    valid = MyMixedDatasetValid(valid_pos_files, valid_neg_files, window=window, limit=valid_limit)
    
    #TODO resolve stopIterationException after iterating throught the whole training dset
    #TODO resolve so i can use generator istead
    class MyMixedDatasetValidStatic(Dataset):
        def __init__(self,generator,limit):
            self.datapoints = []
            gen = iter(generator)
            for i in range(limit):
                #TODO sample randomly, not from the beginning?
                self.datapoints.append(next(gen))
                
        def __getitem__(self,idx):
            return self.datapoints[idx]
        def __len__(self):
            return len(self.datapoints)
    
    return train, MyMixedDatasetValidStatic(valid, valid_limit)
    

def get_my_valid_dataset_unlimited(window=1000, pos_files = 'pos_2022', neg_files='neg_2022', split_method=get_default_split):
    #IN PROGRESS
    split = split_method(pos_files=pos_files, neg_files=neg_files)
    
    train_pos_files = split['train_pos_files']
    train_neg_files = split['train_neg_files']
    valid_pos_files = split['valid_pos_files']
    valid_neg_files = split['valid_neg_files']
    
    print('valid files indicies')
    for files in [valid_pos_files, valid_neg_files]:
        print(sorted([int(Path(x).stem.split('_')[-1]) for x in files]))
    
    
    def process_fast5_read_full(read, window):
        """ Normalizes and extracts specified region from raw signal """

        s = read.get_raw_data(scale=True)  # Expensive
        s = stats.zscore(s)

        #Using custom trim arguments according to Explore notebook
        skip, _ = my_trim(signal=s[:26000])

        last_start_index = len(s)-window
        if(last_start_index < skip):
            # if sequence is not long enough, last #window signals is taken, ignoring the skip index
            skip = last_start_index
        s = s[skip:]
        
        return s
    
    def myite_full(files, label, window):
        for fast5 in files:
            # yield (Path(fast5).stem + f'_lab_{label}')
            # print(Path(fast5).stem,'starting')
            with get_fast5_file(fast5, mode='r') as f5:
                for i, read in enumerate(f5.get_reads()):
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
                        
                        yield x[start:stop].reshape(-1,1).swapaxes(0,1), np.array([y], dtype=np.float32), identifier
                        
    
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

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    total_workers = worker_info.num_workers
    current_worker = worker_id
    
    pos_per_worker = len(dataset.positive_files)//total_workers
    neg_per_worker = len(dataset.negative_files)//total_workers
    
    if(current_worker == total_workers -1): #Last worker
        # print('LAST WORKER')
        dataset.positive_files = dataset.positive_files[pos_per_worker*current_worker:]
        dataset.negative_files = dataset.negative_files[neg_per_worker*current_worker:]
    else:
        dataset.positive_files = dataset.positive_files[pos_per_worker*current_worker: pos_per_worker*(current_worker+1)]
        dataset.negative_files = dataset.negative_files[neg_per_worker*current_worker: neg_per_worker*(current_worker+1)] 
       
    # print(sorted([int(Path(x).stem.split('_')[-1]) for x in dataset.positive_files]))
    # print(sorted([int(Path(x).stem.split('_')[-1]) for x in dataset.negative_files]))
    
    assert(len(dataset.positive_files)>0), f'{pos_per_worker*current_worker}: {pos_per_worker*(current_worker+1)}'
    assert(len(dataset.negative_files)>0), f'{neg_per_worker*current_worker}: {neg_per_worker*(current_worker+1)}'


                         
def my_trim(signal, window_size=200, threshold=1.9, min_elements=25):

    min_trim = 10
    signal = signal[min_trim:]
    num_windows = len(signal) // window_size

    seen_peak = False
    for pos in range(num_windows):
        start = pos * window_size
        end = start + window_size
        window = signal[start:end]
        if len(window[window > threshold]) > min_elements or seen_peak:
            seen_peak = True
            if window[-1] > threshold:
                continue
            return min(end + min_trim, len(signal)), len(signal)
    return min_trim, len(signal)