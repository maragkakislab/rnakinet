import random
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import IterableDataset, Dataset
from torch.utils.data import DataLoader
from pod5 import Reader

from rnakinet.data_utils.generators import uniform_gen, ratio_gen
from rnakinet.data_utils.read_utils import process_pod5_read, pad_read_with_zeros
from rnakinet.data_utils.workers import worker_init_fn_train
class TrainingDatamodule(pl.LightningDataModule):
    def __init__(
            self,
            train_pos_pod5s, #Assumes 1 pod5 per experiment
            train_neg_pod5s,
            valid_pos_pod5s,
            valid_neg_pod5s,
            batch_size,
            valid_read_limit,
            shuffle_valid,
            workers,
            max_len,
            skip,
            multiexp_generator_type,
            min_len,
    ):

        super().__init__()
        self.train_pos_pod5s = train_pos_pod5s
        self.train_neg_pod5s = train_neg_pod5s
        
        self.valid_pos_pod5s = valid_pos_pod5s
        self.valid_neg_pod5s = valid_neg_pod5s
        
        self.batch_size = batch_size
        self.valid_read_limit = valid_read_limit
        self.shuffle_valid = shuffle_valid
        self.workers = workers

        self.train_dataset = None
        self.valid_dataset = None
        self.max_len = max_len
        self.skip = skip
        self.min_len = min_len
        
        self.multiexp_generator_type = multiexp_generator_type

    def setup(self, stage=None):
        if (stage == 'fit' or stage is None):
            self.train_dataset = UnlimitedReadsTrainingDataset(
                pos_pod5s=self.train_pos_pod5s,
                neg_pod5s=self.train_neg_pod5s,
                max_len = self.max_len,
                skip=self.skip,
                min_len=self.min_len,
                multiexp_generator_type=self.multiexp_generator_type,
            )
            self.valid_dataset = UnlimitedReadsValidDataset(
                pos_pod5s=self.valid_pos_pod5s,
                neg_pod5s=self.valid_neg_pod5s,
                valid_read_limit=self.valid_read_limit,
                skip=self.skip,
                max_len=self.max_len,
                min_len=self.min_len,
                multiexp_generator_type='uniform',
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.workers,
            worker_init_fn=worker_init_fn_train,
        )

    def val_dataloader(self):
        return  DataLoader(self.valid_dataset, batch_size=self.batch_size)

    
class UnlimitedReadsTrainingDataset(IterableDataset):
    """
    Iterable Dataset that contains all reads
    """
    def __init__(self, pos_pod5s, neg_pod5s, max_len, skip, min_len, multiexp_generator_type):
        self.positive_pod5s = pos_pod5s
        self.negative_pod5s = neg_pod5s
        self.max_len = max_len
        self.min_len = min_len
        self.skip = skip
        self.multiexp_generator_type = multiexp_generator_type

        # This will get modified by worker processes, if multiprocessing takes place
        self.pod5_to_readids = {pod5: Reader(pod5).read_ids for pod5 in self.positive_pod5s+self.negative_pod5s}

    def process_pod5(self, pod5_file, label, exp):
        while True:
            with Reader(pod5_file) as reader:
                read_ids = self.pod5_to_readids[pod5_file]
                for read in reader.reads(selection=read_ids):
                    x = process_pod5_read(read, skip=self.skip)
                    y = np.array(label)
                    if len(x) > self.max_len or len(x) < self.min_len:
                        continue
                        
                    yield pad_read_with_zeros(x.reshape(-1, 1).swapaxes(0, 1), max_len=self.max_len), np.array([y], dtype=np.float32), exp

    def get_pod5_size(self, pod5_file):
        return len(self.pod5_to_readids[pod5_file])

    def get_stream(self):
        pos_gens = []
        pos_sizes = []
        for pos_pod5 in self.positive_pod5s:
            pos_gens.append(self.process_pod5(pod5_file=pos_pod5, label=1, exp='pos'))
            pos_sizes.append(self.get_pod5_size(pos_pod5))
        neg_gens = []
        neg_sizes = []
        for neg_pod5 in self.negative_pod5s:
            neg_gens.append(self.process_pod5(pod5_file=neg_pod5, label=0, exp='neg'))
            neg_sizes.append(self.get_pod5_size(neg_pod5))
            
        if(self.multiexp_generator_type  == 'uniform'):
            global_pos_gen = uniform_gen(pos_gens)
            global_neg_gen = uniform_gen(neg_gens)
        
        if(self.multiexp_generator_type == 'ratio'):
            pos_ratios = np.array(pos_sizes)/np.sum(pos_sizes)
            neg_ratios = np.array(neg_sizes)/np.sum(neg_sizes)
            global_pos_gen = ratio_gen(pos_gens, pos_ratios)
            global_neg_gen = ratio_gen(neg_gens, neg_ratios)
        
        gen = uniform_gen([global_pos_gen, global_neg_gen])
        
        while True:
            yield next(gen)

    def __iter__(self):
        return self.get_stream()

  
class UnlimitedReadsValidDataset(Dataset):
    """
    Mapped Dataset that contains validation reads
    """
    def __init__(self, pos_pod5s, neg_pod5s, max_len, skip, min_len, multiexp_generator_type, valid_read_limit):
        self.positive_pod5s = pos_pod5s
        self.negative_pod5s = neg_pod5s
        self.max_len = max_len
        self.min_len = min_len
        self.skip = skip
        self.multiexp_generator_type = multiexp_generator_type
        self.valid_read_limit = valid_read_limit
        
        print('Generating valid dataset')
        self.items = self.generate_data()

    def process_pod5(self, pod5_file, label):
        while True:
            with Reader(pod5_file) as reader:
                read_ids = reader.read_ids
                random.shuffle(read_ids)
                for read in reader.reads(selection=read_ids):
                    x = process_pod5_read(read, skip=self.skip)
                    y = np.array(label)
                    if len(x) > self.max_len or len(x) < self.min_len:
                        continue

                    yield pad_read_with_zeros(x.reshape(-1, 1).swapaxes(0, 1), max_len=self.max_len), np.array([y], dtype=np.float32)
    
    def get_pod5_size(self, pod5_file):
        with Reader(pod5_file) as reader:
            return len(reader.read_ids)

    def generate_data(self):
        pos_gens = []
        pos_sizes = []
        for pos_pod5 in self.positive_pod5s:
            pos_gens.append(self.process_pod5(pos_pod5, label=1))
            pos_sizes.append(self.get_pod5_size(pos_pod5))
        neg_gens = []
        neg_sizes = []
        for neg_pod5 in self.negative_pod5s:
            neg_gens.append(self.process_pod5(neg_pod5, label=0))
            neg_sizes.append(self.get_pod5_size(neg_pod5))
            
        if(self.multiexp_generator_type  == 'uniform'):
            global_pos_gen = uniform_gen(pos_gens)
            global_neg_gen = uniform_gen(neg_gens)
        
        if(self.multiexp_generator_type == 'ratio'):
            pos_ratios = np.array(pos_sizes)/np.sum(pos_sizes)
            neg_ratios = np.array(neg_sizes)/np.sum(neg_sizes)
            global_pos_gen = ratio_gen(pos_gens, pos_ratios)
            global_neg_gen = ratio_gen(neg_gens, neg_ratios)
        
        gen = uniform_gen([global_pos_gen, global_neg_gen])
        
        items = []
        for _ in range(self.valid_read_limit):
            x, y = next(gen)
            items.append((x, y))
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

class InferenceDataset(IterableDataset):
    """
    Iterable Dataset that contains all reads from POD5 files
    """

    def __init__(self, files, max_len, skip, min_len):
        self.files = files
        self.max_len = max_len
        self.skip = skip
        self.min_len = min_len

    def process_files_fully(self, files):
        for pod5_file in files:
            try:
                with Reader(pod5_file) as reader:
                    for read in reader.reads():
                        x = process_pod5_read(read, skip=self.skip)
                        if len(x) > self.max_len or len(x) < self.min_len:
                            continue
                        identifier = {
                            'file': str(pod5_file),
                            'readid': str(read.read_id),
                        }
                        yield pad_read_with_zeros(x.reshape(-1, 1).swapaxes(0, 1), max_len=self.max_len), identifier
                        
            except OSError as error:
                print(error)
                continue

    def __iter__(self):
        return self.process_files_fully(self.files)