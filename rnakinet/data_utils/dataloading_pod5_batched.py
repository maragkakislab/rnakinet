import functools
import os
import random

import numpy as np
import lightning.pytorch as pl
from pod5 import Reader
from torch.utils.data import DataLoader, Dataset, IterableDataset

from rnakinet.data_utils.generators import ratio_gen, uniform_gen
from rnakinet.data_utils.read_utils import pad_read_with_zeros, process_pod5_read
from rnakinet.data_utils.workers import worker_init_fn_experiments


def list_pod5_files(path):
    """A POD5 file, or all POD5 files under a directory (sorted)."""
    if os.path.isfile(path):
        return [path]
    files = []
    for root, _, names in os.walk(path, followlinks=True):
        files += [os.path.join(root, n) for n in names
                  if n.lower().endswith('.pod5')]
    return sorted(files)


def read_id_set(path):
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


@functools.lru_cache(maxsize=None)
def pod5_read_ids(pod5_file):
    """All read ids of a POD5 file; cached, since the training and
    validation datasets scan the same raw files."""
    with Reader(pod5_file) as reader:
        return tuple(str(r) for r in reader.read_ids)


@functools.lru_cache(maxsize=None)
def pod5_batch_sizes(pod5_file):
    """Number of reads in each record batch of a POD5 file. Read ids are
    stored in batch order, so these split pod5_read_ids into batches."""
    with Reader(pod5_file) as reader:
        return tuple(reader.get_batch(i).num_reads
                     for i in range(reader.batch_count))


def select_batches(selection):
    """{(pod5 file, record batch index): rows} for a selection ({pod5 file:
    [read ids]}): the selected reads' row numbers within each record batch
    (uint16 arrays), skipping batches with no selected reads. Record batches
    are the unit of shuffling: reading whole batches in random order is as
    fast as reading sequentially, random reads are not. Row numbers instead
    of read id strings keep this small: DataLoader workers each hold a
    copy."""
    batches = {}
    for pod5_file, ids in selection.items():
        wanted = set(ids)
        all_ids = pod5_read_ids(pod5_file)
        start = 0
        for i, size in enumerate(pod5_batch_sizes(pod5_file)):
            rows = [r for r, read_id in enumerate(all_ids[start:start + size])
                    if read_id in wanted]
            start += size
            if rows:
                batches[(pod5_file, i)] = np.array(rows, dtype=np.uint16)
    return batches


def select_reads(pod5_path, read_ids=None):
    """{pod5 file: [read ids]} for the reads of pod5_path (a file or a
    directory), restricted to read_ids (a set) if given."""
    selection = {}
    for pod5_file in list_pod5_files(pod5_path):
        ids = [r for r in pod5_read_ids(pod5_file)
               if read_ids is None or r in read_ids]
        if ids:
            selection[pod5_file] = ids
    return selection


def load_experiment(spec):
    """An experiment is given as '<raw pod5 file or dir>::<read ids file>':
    the reads listed in the ids file, read from the raw POD5 data."""
    pod5_path, sep, ids_path = spec.partition('::')
    if not sep:
        raise ValueError(f"expected '<pod5 path>::<read ids file>': {spec}")
    selection = select_reads(pod5_path, read_id_set(ids_path))
    if not selection:
        raise ValueError(f'none of the reads in {ids_path} are in {pod5_path}')
    return selection


class TrainingDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        # experiments: '<raw pod5 file or dir>::<read ids file>'
        train_pos_experiments,
        train_neg_experiments,
        valid_pos_experiments,
        valid_neg_experiments,
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
        self.train_pos_experiments = train_pos_experiments
        self.train_neg_experiments = train_neg_experiments

        self.valid_pos_experiments = valid_pos_experiments
        self.valid_neg_experiments = valid_neg_experiments

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
        if stage == 'fit' or stage is None:
            self.train_dataset = UnlimitedReadsTrainingDataset(
                pos_experiments=self.train_pos_experiments,
                neg_experiments=self.train_neg_experiments,
                max_len=self.max_len,
                skip=self.skip,
                min_len=self.min_len,
                multiexp_generator_type=self.multiexp_generator_type,
            )
            self.valid_dataset = UnlimitedReadsValidDataset(
                pos_experiments=self.valid_pos_experiments,
                neg_experiments=self.valid_neg_experiments,
                valid_read_limit=self.valid_read_limit,
                skip=self.skip,
                max_len=self.max_len,
                min_len=self.min_len,
                multiexp_generator_type='uniform',
            )
            # the read ids are only needed to build the datasets; workers
            # would otherwise each copy them
            pod5_read_ids.cache_clear()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            worker_init_fn=worker_init_fn_experiments,
        )

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)


class UnlimitedReadsTrainingDataset(IterableDataset):
    """
    Iterable Dataset that streams the selected reads of each experiment
    """

    def __init__(self, pos_experiments, neg_experiments, max_len, skip,
                 min_len, multiexp_generator_type):
        self.max_len = max_len
        self.min_len = min_len
        self.skip = skip
        self.multiexp_generator_type = multiexp_generator_type

        # {(pod5 file, record batch): rows} per experiment. Worker
        # processes keep only their share of each experiment's batches
        # (worker_init_fn_experiments)
        self.positive_experiments = [
            select_batches(load_experiment(e)) for e in pos_experiments]
        self.negative_experiments = [
            select_batches(load_experiment(e)) for e in neg_experiments]
        self.experiments = (self.positive_experiments
                            + self.negative_experiments)

    def process_experiment(self, batches, label, exp):
        order = list(batches)
        while True:
            # a new batch order on every pass; the DataLoader seeds `random`
            # differently in each worker
            random.shuffle(order)
            for pod5_file, batch in order:
                with Reader(pod5_file) as reader:
                    record_batch = reader.get_batch(batch)
                    wanted = np.zeros(record_batch.num_reads, dtype=bool)
                    wanted[batches[(pod5_file, batch)]] = True
                    for row, read in enumerate(record_batch.reads()):
                        if not wanted[row]:
                            continue
                        x = process_pod5_read(read, skip=self.skip)
                        y = np.array(label)
                        if len(x) > self.max_len or len(x) < self.min_len:
                            continue
                        x = pad_read_with_zeros(
                            x.reshape(-1, 1).swapaxes(0, 1),
                            max_len=self.max_len)
                        yield x, np.array([y], dtype=np.float32), exp

    @staticmethod
    def experiment_size(batches):
        return sum(len(ids) for ids in batches.values())

    def get_stream(self):
        pos_gens = [self.process_experiment(s, label=1, exp='pos')
                    for s in self.positive_experiments]
        pos_sizes = [self.experiment_size(s)
                     for s in self.positive_experiments]
        neg_gens = [self.process_experiment(s, label=0, exp='neg')
                    for s in self.negative_experiments]
        neg_sizes = [self.experiment_size(s)
                     for s in self.negative_experiments]

        if self.multiexp_generator_type == 'uniform':
            global_pos_gen = uniform_gen(pos_gens)
            global_neg_gen = uniform_gen(neg_gens)

        if self.multiexp_generator_type == 'ratio':
            pos_ratios = np.array(pos_sizes) / np.sum(pos_sizes)
            neg_ratios = np.array(neg_sizes) / np.sum(neg_sizes)
            global_pos_gen = ratio_gen(pos_gens, pos_ratios)
            global_neg_gen = ratio_gen(neg_gens, neg_ratios)

        gen = uniform_gen([global_pos_gen, global_neg_gen])

        while True:
            yield next(gen)

    def __iter__(self):
        return self.get_stream()


class UnlimitedReadsValidDataset(Dataset):
    """
    Mapped Dataset of valid_read_limit reads, sampled at random from the
    experiments' selected reads
    """

    def __init__(self, pos_experiments, neg_experiments, max_len, skip,
                 min_len, multiexp_generator_type, valid_read_limit):
        self.positive_experiments = [
            load_experiment(e) for e in pos_experiments]
        self.negative_experiments = [
            load_experiment(e) for e in neg_experiments]
        self.max_len = max_len
        self.min_len = min_len
        self.skip = skip
        self.multiexp_generator_type = multiexp_generator_type
        self.valid_read_limit = valid_read_limit

        print('Generating valid dataset')
        self.items = self.generate_data()
        # only the sampled reads are kept
        del self.positive_experiments, self.negative_experiments

    def process_experiment(self, selection, label):
        reads = [(f, r) for f, ids in selection.items() for r in ids]
        while True:
            # a random sample, read file by file; twice the limit leaves room
            # for reads dropped by the length filter
            sample = random.sample(
                reads, min(len(reads), 2 * self.valid_read_limit))
            by_file = {}
            for pod5_file, read_id in sample:
                by_file.setdefault(pod5_file, []).append(read_id)
            for pod5_file, read_ids in by_file.items():
                with Reader(pod5_file) as reader:
                    for read in reader.reads(selection=read_ids):
                        x = process_pod5_read(read, skip=self.skip)
                        y = np.array(label)
                        if len(x) > self.max_len or len(x) < self.min_len:
                            continue

                        x = pad_read_with_zeros(
                            x.reshape(-1, 1).swapaxes(0, 1),
                            max_len=self.max_len)
                        yield x, np.array([y], dtype=np.float32)

    def generate_data(self):
        pos_gens = [self.process_experiment(s, label=1)
                    for s in self.positive_experiments]
        pos_sizes = [sum(len(ids) for ids in s.values())
                     for s in self.positive_experiments]
        neg_gens = [self.process_experiment(s, label=0)
                    for s in self.negative_experiments]
        neg_sizes = [sum(len(ids) for ids in s.values())
                     for s in self.negative_experiments]

        if self.multiexp_generator_type == 'uniform':
            global_pos_gen = uniform_gen(pos_gens)
            global_neg_gen = uniform_gen(neg_gens)

        if self.multiexp_generator_type == 'ratio':
            pos_ratios = np.array(pos_sizes) / np.sum(pos_sizes)
            neg_ratios = np.array(neg_sizes) / np.sum(neg_sizes)
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
    Iterable Dataset that contains all reads from POD5 files, or only the
    reads in read_ids (a set) if given
    """

    def __init__(self, files, max_len, skip, min_len, pad_reads=True,
                 read_ids=None):
        self.files = files
        self.max_len = max_len
        self.skip = skip
        self.min_len = min_len
        self.pad_reads = pad_reads
        self.read_ids = read_ids

    def process_files_fully(self, files):
        for pod5_file in files:
            try:
                with Reader(pod5_file) as reader:
                    selection = None
                    if self.read_ids is not None:
                        selection = [str(r) for r in reader.read_ids
                                     if str(r) in self.read_ids]
                        if not selection:
                            continue
                    for read in reader.reads(selection=selection):
                        x = process_pod5_read(read, skip=self.skip)
                        if len(x) > self.max_len or len(x) < self.min_len:
                            continue
                        identifier = {
                            'file': str(pod5_file),
                            'readid': str(read.read_id),
                        }
                        if self.pad_reads:
                            yield pad_read_with_zeros(x.reshape(-1, 1).swapaxes(0, 1), max_len=self.max_len), identifier
                        else:
                            yield x.reshape(-1, 1).swapaxes(0, 1), identifier
            except OSError as error:
                print(error)
                continue

    def __iter__(self):
        return self.process_files_fully(self.files)
