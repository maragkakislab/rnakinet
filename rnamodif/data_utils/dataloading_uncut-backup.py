import random
import pytorch_lightning as pl
import numpy as np
from ont_fast5_api.fast5_interface import get_fast5_file
from torch.utils.data import IterableDataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from rnamodif.data_utils.generators import uniform_gen, ratio_gen
from rnamodif.data_utils.read_utils import process_read
from rnamodif.data_utils.workers import worker_init_fn


class TrainingDatamodule(pl.LightningDataModule):
    def __init__(
            self,
            train_pos_files,
            train_neg_files,
            valid_exp_to_files_pos,
            valid_exp_to_files_neg,
            valid_auroc_tuples,
            batch_size,
            valid_per_dset_read_limit,
            shuffle_valid,
            workers,
            max_len,
            skip,
            multiexp_generator_type,
            min_len,
            preprocess='rodan',
    ):

        super().__init__()
        self.train_pos_files = train_pos_files
        self.train_neg_files = train_neg_files

        self.valid_exp_to_files_pos = valid_exp_to_files_pos
        self.valid_exp_to_files_neg = valid_exp_to_files_neg

        check_leakage(
            train_pos_files = self.train_pos_files, 
            train_neg_files = self.train_neg_files, 
            valid_exp_to_files_pos = self.valid_exp_to_files_pos, 
            valid_exp_to_files_neg = self.valid_exp_to_files_neg,
        )
        
        self.batch_size = batch_size
        self.valid_per_dset_read_limit = valid_per_dset_read_limit
        self.shuffle_valid = shuffle_valid
        self.workers = workers

        self.train_dataset = None
        self.valid_dataset = None
        self.max_len = max_len
        self.skip = skip
        self.valid_auroc_tuples = valid_auroc_tuples
        self.min_len = min_len
        self.preprocess=preprocess
        
        self.multiexp_generator_type = multiexp_generator_type

    def setup(self, stage=None):
        if (stage == 'fit' or stage is None):
            self.train_dataset = UnlimitedReadsTrainingDataset(
                pos_files=self.train_pos_files,
                neg_files=self.train_neg_files,
                max_len = self.max_len,
                skip=self.skip,
                min_len=self.min_len,
                preprocess=self.preprocess,
                multiexp_generator_type=self.multiexp_generator_type,
            )
            #TODO len limit/min_len ?
            self.valid_dataset = UnlimitedReadsValidDataset(
                valid_exp_to_files_pos=self.valid_exp_to_files_pos,
                valid_exp_to_files_neg=self.valid_exp_to_files_neg,
                valid_per_dset_read_limit=self.valid_per_dset_read_limit,
                shuffle=self.shuffle_valid,
                skip=self.skip,
                preprocess=self.preprocess,
                max_len=self.max_len,
                min_len=self.min_len,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.workers)#, worker_init_fn=worker_init_fn)

    def val_dataloader(self):
        return  DataLoader(self.valid_dataset, batch_size=self.batch_size)


    
class UnlimitedReadsTrainingDataset(IterableDataset):
    """
    Iterable Dataset that contains all reads
    """
    def __init__(self, pos_files, neg_files, max_len, preprocess, skip, min_len, multiexp_generator_type):
        self.positive_files = pos_files
        self.negative_files = neg_files
        self.max_len = max_len
        self.min_len = min_len
        self.skip = skip
        self.preprocess = preprocess
        self.multiexp_generator_type = multiexp_generator_type

    def process_files(self, files, label, exp):
        while True:
            fast5 = random.choice(files)
            try:
                with get_fast5_file(fast5, mode='r') as f5:
                    for read in f5.get_reads():
                        x = process_read(read, window=None, skip=self.skip, preprocess=self.preprocess)
                        y = np.array(label)
                        # Skip if the read is too short
                        if (len(x) > self.max_len or len(x) < self.min_len):
                            continue
                        yield x.reshape(-1, 1).swapaxes(0, 1), np.array([y], dtype=np.float32), exp
            except OSError as error:
                print(error)
                continue

    def get_stream(self):
        pos_gens = []
        pos_sizes = []
        for pos_files_instance in self.positive_files:
            assert len(pos_files_instance) > 0, pos_files_instance
            pos_gens.append(self.process_files(files=pos_files_instance, label=1, exp='pos'))
            pos_sizes.append(len(pos_files_instance))
        neg_gens = []
        neg_sizes = []
        for neg_files_instance in self.negative_files:
            assert len(neg_files_instance) > 0, neg_files_instance
            neg_gens.append(self.process_files(files=neg_files_instance, label=0, exp='neg'))
            neg_sizes.append(len(neg_files_instance))
            
        if(self.multiexp_generator_type  == 'uniform'):
            global_pos_gen = uniform_gen(pos_gens)
            global_neg_gen = uniform_gen(neg_gens)
        
        if(self.multiexp_generator_type == 'ratio'):
            pos_ratios = np.array(pos_sizes)/np.sum(pos_sizes)
            neg_ratios = np.array(neg_sizes)/np.sum(neg_sizes)
            # print('Train positives ratios', pos_ratios)
            # print('Train negatives ratios', neg_ratios)
            global_pos_gen = ratio_gen(pos_gens, pos_ratios)
            global_neg_gen = ratio_gen(neg_gens, neg_ratios)
        
        gen = uniform_gen([global_pos_gen, global_neg_gen])
        
        while True:
            yield next(gen)

    def __iter__(self):
        return self.get_stream()

class UnlimitedReadsInferenceDataset(IterableDataset):
    """
    Iterable Dataset that contains all reads
    """

    def __init__(self, files, max_len, skip, min_len, preprocess='rodan'):
        self.files = files
        self.max_len = max_len
        self.skip = skip
        self.min_len = min_len
        self.preprocess = preprocess

    def process_files_fully(self, files):
        for fast5 in files:
            try:
                with get_fast5_file(fast5, mode='r') as f5:
                    for i, read in enumerate(f5.get_reads()):
                        x = process_read(read, window=None, skip=self.skip, preprocess=self.preprocess)
                        start = 0
                        stop = len(x)
                        if(len(x) > self.max_len or len(x) < self.min_len):
                            #TODO how to resolve?
                            continue
                        identifier = {
                            'file': str(fast5),
                            'readid': read.read_id,
                            'read_index_in_file': 0,
                            'start': start,
                            'stop': stop,
                        }
                        yield x.reshape(-1, 1).swapaxes(0, 1), identifier
            except OSError as error:
                print(error)
                continue

    def __iter__(self):
        return self.process_files_fully(self.files)
    
    
    
class UnlimitedReadsValidDataset(Dataset):
    def __init__(self, valid_exp_to_files_pos, valid_exp_to_files_neg, valid_per_dset_read_limit, shuffle, preprocess, max_len, skip, min_len):
        self.skip = skip
        self.min_len = min_len
        self.max_len = max_len
        self.preprocess = preprocess
        
        pos_gens = []
        for exp, files in valid_exp_to_files_pos.items():
            pos_gens.append(self.process_files_fully(
                files, exp, label=1, shuffle=shuffle))
        neg_gens = []
        for exp, files in valid_exp_to_files_neg.items():
            neg_gens.append(self.process_files_fully(
                files, exp, label=0, shuffle=shuffle))

        items = []
        print('Generating valid dataset')
        for gen in tqdm(pos_gens+neg_gens):
            reads_processed = 0
            last_read = None
            while True:
                x, y, identifier = next(gen)

                current_read = identifier['readid']
                if (last_read and last_read != current_read):
                    reads_processed += 1
                    if (reads_processed >= valid_per_dset_read_limit):
                        print(identifier['exp'], reads_processed)
                        break
                last_read = current_read
                items.append((x, y, identifier))

        self.items = items

    def process_files_fully(self, files, exp, label, shuffle):
        if (shuffle):
            random.shuffle(files)
        for fast5 in files:
            with get_fast5_file(fast5, mode='r') as f5:
                for i, read in enumerate(f5.get_reads()):
                    x = process_read(read, window=None, skip=self.skip, preprocess=self.preprocess)
                    if(len(x) > self.max_len or len(x) < self.min_len):
                        print('skipping too long')
                        continue
                    y = np.array(label)
                    identifier = {'file': str(fast5),
                                  'readid': read.read_id,
                                  'read_index_in_file': i,
                                  'label': label,
                                  'exp': exp,
                                  }

                    yield x.reshape(-1, 1).swapaxes(0, 1), np.array([y], dtype=np.float32), identifier

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
    

    
    
# CHECK DATA LEAKAGE TEST
def check_leakage(train_pos_files, train_neg_files, valid_exp_to_files_pos, valid_exp_to_files_neg):
    train_sets = []
    for file_array in train_neg_files+train_pos_files:
        train_sets.append(set(file_array))

    assert(len(set.intersection(*train_sets))==0)


    test_sets = []
    for file_array in list(valid_exp_to_files_pos.values())+list(valid_exp_to_files_neg.values()):
        test_sets.append(set(file_array))

    assert(len(set.intersection(*test_sets))==0)

    train_set = set.union(*train_sets)
    test_set = set.union(*test_sets)

    assert(len(set.intersection(train_set, test_set))==0)