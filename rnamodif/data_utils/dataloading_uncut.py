import random
import pytorch_lightning as pl
import numpy as np
from ont_fast5_api.fast5_interface import get_fast5_file
from torch.utils.data import IterableDataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from rnamodif.data_utils.generators import uniform_gen
from rnamodif.data_utils.read_utils import process_read
from rnamodif.data_utils.workers import worker_init_fn


class TrainingDatamodule(pl.LightningDataModule):
    def __init__(
            self,
            train_pos_files,
            train_neg_files,
            valid_exp_to_files_pos,
            valid_exp_to_files_neg,
            batch_size,
            per_dset_read_limit,
            shuffle_valid,
            workers,
            len_limit,):

        super().__init__()
        self.train_pos_files = train_pos_files
        self.train_neg_files = train_neg_files

        self.valid_exp_to_files_pos = valid_exp_to_files_pos
        self.valid_exp_to_files_neg = valid_exp_to_files_neg

        self.batch_size = batch_size
        self.per_dset_read_limit = per_dset_read_limit
        self.shuffle_valid = shuffle_valid
        self.workers = workers

        self.train_dataset = None
        self.valid_dataset = None
        self.len_limit = len_limit

    def setup(self, stage=None):
        if (stage == 'fit' or stage is None):
            self.train_dataset = UnlimitedReadsTrainingDataset(
                pos_files=self.train_pos_files,
                neg_files=self.train_neg_files,
                len_limit = self.len_limit,
            )

            self.valid_dataset = UnlimitedReadsValidDataset(
                valid_exp_to_files_pos=self.valid_exp_to_files_pos,
                valid_exp_to_files_neg=self.valid_exp_to_files_neg,
                per_dset_read_limit=self.per_dset_read_limit,
                shuffle=self.shuffle_valid
            )

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                  num_workers=self.workers)#, worker_init_fn=worker_init_fn)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size)
        return val_loader


    
class UnlimitedReadsTrainingDataset(IterableDataset):
    """
    Iterable Dataset that contains all reads
    """
    def __init__(self, pos_files, neg_files, len_limit):
        self.positive_files = pos_files
        self.negative_files = neg_files
        self.len_limit = len_limit

    def process_files(self, files, label, exp):
        while True:
            fast5 = random.choice(files)
            with get_fast5_file(fast5, mode='r') as f5:
                for read in f5.get_reads():
                    x = process_read(read, window=None)
                    y = np.array(label)
                    # Skip if the read is too short
                    if (len(x) > self.len_limit):
                        continue

                    yield x.reshape(-1, 1).swapaxes(0, 1), np.array([y], dtype=np.float32), exp

    def get_stream(self):
        pos_gens = []
        for pos_files_instance in self.positive_files:
            pos_gens.append(self.process_files(files=pos_files_instance, label=1, exp='pos'))
        neg_gens = []
        for neg_files_instance in self.negative_files:
            neg_gens.append(self.process_files(files=neg_files_instance, label=0, exp='neg'))
        global_pos_gen = uniform_gen(pos_gens)
        global_neg_gen = uniform_gen(neg_gens)
        
        gen = uniform_gen([global_pos_gen, global_neg_gen])
        while True:
            yield next(gen)

    def __iter__(self):
        return self.get_stream()

class UnlimitedReadsInferenceDataset(IterableDataset):
    """
    Iterable Dataset that contains all reads
    """

    def __init__(self, files, len_limit=400000):
        self.files = files
        self.len_limit = len_limit

    def process_files_fully(self, files):
        for fast5 in files:
            try:
                with get_fast5_file(fast5, mode='r') as f5:
                    for i, read in enumerate(f5.get_reads()):
                        x = process_read(read, window=None)
                        start = 0
                        stop = len(x)
                        if(len(x) > self.len_limit):
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
    def __init__(self, valid_exp_to_files_pos, valid_exp_to_files_neg, per_dset_read_limit, shuffle):
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
                    if (reads_processed >= per_dset_read_limit):
                        print(identifier['exp'], reads_processed)
                        break
                last_read = current_read
                items.append((x, y, identifier))

        self.items = items

    def process_files_fully(self, files, exp, label, shuffle):
        for fast5 in files:
            if (shuffle): #TODO shuffle is misplaced, should be above for loop!
                random.shuffle(files)
            with get_fast5_file(fast5, mode='r') as f5:
                for i, read in enumerate(f5.get_reads()):
                    x = process_read(read, window=None)
                    if(len(x) > 1000000):
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
    
