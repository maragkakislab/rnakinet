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
    """
    Pytorch lightning datamodule wrapping training and validation dataloaders

    Keyword arguments:
    train_pos_files: list of fast5 files to be used as positives for training
    train_neg_files: list of fast5 files to be used as negatives for training
    valid_exp_to_files_pos: dict where keys are experiment names 
        and values are lists of files to be used as positives for validation
    valid_exp_to_files_neg: dict where keys are experiment names 
        and values are lists of files to be used as negatives for validation

    batch_size: batch size used for training
    per_dset_read_limit: how many reads from each experiment to use for validation
    shuffle_valid: flag to shuffle experiment files before taking some for validation
    workers: how many workers to use for loading the training data
    """

    def __init__(
            self,
            train_pos_files,
            train_neg_files,
            valid_exp_to_files_pos,
            valid_exp_to_files_neg,
            batch_size, window,
            per_dset_read_limit,
            shuffle_valid,
            workers):

        super().__init__()
        self.train_pos_files = train_pos_files
        self.train_neg_files = train_neg_files

        self.valid_exp_to_files_pos = valid_exp_to_files_pos
        self.valid_exp_to_files_neg = valid_exp_to_files_neg

        self.batch_size = batch_size
        self.window = window
        self.per_dset_read_limit = per_dset_read_limit
        self.shuffle_valid = shuffle_valid
        self.workers = workers

        self.train_dataset = None
        self.valid_dataset = None

    def setup(self, stage=None):
        if (stage == 'fit' or stage is None):
            self.train_dataset = InfiniteSampleDataset(
                pos_files=self.train_pos_files,
                neg_files=self.train_neg_files,
                window=self.window,
            )

            self.valid_dataset = CompleteReadsValidDataset(
                valid_exp_to_files_pos=self.valid_exp_to_files_pos,
                valid_exp_to_files_neg=self.valid_exp_to_files_neg,
                window=self.window,
                stride=self.window,
                per_dset_read_limit=self.per_dset_read_limit,
                shuffle=self.shuffle_valid
            )

    def train_dataloader(self):
        print('NOT USING WORKER INIT')
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                  num_workers=self.workers, pin_memory=True)#, worker_init_fn=worker_init_fn)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size)
        return val_loader


class InfiniteSampleDataset(IterableDataset):
    """
    Iterable Dataset that yields random chunks of random reads
    """

    def __init__(self, pos_files, neg_files, window):
        super().__init__()

        self.positive_files = pos_files
        self.negative_files = neg_files
        self.window = window

    def process_files(self, files, label, window, exp):
        while True:
            fast5 = random.choice(files)
            # TODO consider removing try-except and checking file integrity beforehand
            try:
                with get_fast5_file(fast5, mode='r') as f5:
                    reads = list(f5.get_reads())
                    for _ in range(len(reads)):
                        read = random.choice(reads)
                        x = process_read(read, window)
                        y = np.array(label)
                        # Skip if the read is too short
                        if (len(x) == 0):
                            print('skipping')
                            continue

                        yield x.reshape(-1, 1).swapaxes(0, 1), np.array([y], dtype=np.float32), exp
            except OSError as error:
                print(error)
                continue

    def get_stream(self):
        # TODO remove exp from training loop
        pos_gen = self.process_files(
            files=self.positive_files, label=1, window=self.window, exp='pos')
        neg_gen = self.process_files(
            files=self.negative_files, label=0, window=self.window, exp='neg')
        gen = uniform_gen([pos_gen, neg_gen])
        while True:
            yield next(gen)

    def __iter__(self):
        return self.get_stream()


class CompleteReadsValidDataset(Dataset):
    """
    Mapped (in-memory) Dataset that contains multiple reads, 
        and all of their chunks (except the last incomplete one)
    """

    def __init__(self, valid_exp_to_files_pos, valid_exp_to_files_neg, window, stride, per_dset_read_limit, shuffle):
        pos_gens = []
        for exp, files in valid_exp_to_files_pos.items():
            pos_gens.append(self.process_files_fully(
                files, exp, label=1, window=window, stride=stride, shuffle=shuffle))
        neg_gens = []
        for exp, files in valid_exp_to_files_neg.items():
            neg_gens.append(self.process_files_fully(
                files, exp, label=0, window=window, stride=stride, shuffle=shuffle))

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

    def process_files_fully(self, files, exp, label, window, stride, shuffle):
        for fast5 in files:
            if (shuffle):
                random.shuffle(files)
            with get_fast5_file(fast5, mode='r') as f5:
                for i, read in enumerate(f5.get_reads()):
                    # window = None for returning the whole signal
                    x = process_read(read, window=None)
                    y = np.array(label)
                    # Cutoff the last incomplete signal
                    for start in range(0, len(x), stride)[:-1]:
                        stop = start+window
                        identifier = {'file': str(fast5),
                                      'readid': read.read_id,
                                      'read_index_in_file': i,
                                      'start': start,
                                      'stop': stop,
                                      'label': label,
                                      'exp': exp,
                                      }

                        yield x[start:stop].reshape(-1, 1).swapaxes(0, 1), np.array([y], dtype=np.float32), identifier

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class CompleteReadsInferenceDataset(IterableDataset):
    """
    Iterable Dataset that contains all reads, 
        and all of their chunks (except the last incomplete one)
    """

    def __init__(self, files, window, stride):
        self.files = files
        self.window = window
        self.stride = stride

    def process_files_fully(self, files, window):
        for fast5 in files:
            try:
                with get_fast5_file(fast5, mode='r') as f5:
                    for i, read in enumerate(f5.get_reads()):
                        x = process_read(read, window=None)
                        # TODO trim start of the read?
                        for start in range(0, len(x), self.stride):
                            stop = start+window
                            if (stop >= len(x)):
                                continue
                            identifier = {'file': str(fast5),
                                          'readid': read.read_id,
                                          'read_index_in_file': i,
                                          'start': start,
                                          'stop': stop,
                                          }
                            yield x[start:stop].reshape(-1, 1).swapaxes(0, 1), identifier
            except OSError as error:
                print(error)
                continue

    def __iter__(self):
        return self.process_files_fully(self.files, self.window)
