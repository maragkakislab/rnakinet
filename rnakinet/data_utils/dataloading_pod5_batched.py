import functools
import os
import random

import numpy as np
import lightning.pytorch as pl
import torch
from pod5 import Reader
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data import default_collate

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


def collate_variable(batch):
    """Collate reads of different lengths (input mode 'variable'): pad
    each to the longest read of the batch and pass the lengths. Items are
    (read (1, length), *rest); batches are (reads, lengths, *rest)."""
    lengths = torch.tensor([item[0].shape[-1] for item in batch])
    reads = torch.zeros(len(batch), 1, int(lengths.max()))
    for i, item in enumerate(batch):
        reads[i, 0, :lengths[i]] = torch.as_tensor(item[0][0])
    rest = default_collate([item[1:] for item in batch])
    return (reads, lengths, *rest)


def length_balanced(stream, min_len, max_len, bins=50, slack=10):
    """Pass items of a stream of (x, y, exp, length) so that labeled and
    unlabeled reads end up with the same length distribution: per length
    bin (log-spaced), a read is skipped while its class is more than slack
    reads ahead of the other class in that bin. The label can then not be
    inferred from read length."""
    edges = np.geomspace(min_len, max_len, bins + 1)[1:-1]
    counts = np.zeros((2, bins), dtype=np.int64)
    for item in stream:
        label = int(item[1][0] > 0.5)
        b = int(np.digitize(item[3], edges))
        if counts[label, b] - counts[1 - label, b] >= slack:
            continue
        counts[label, b] += 1
        yield item


def crop_read(x, crop, random_start=False):
    """A read's first `crop` samples, or a window of that length at a random
    position (random_start); shorter reads are kept whole. crop 0: the whole
    read."""
    if not crop or len(x) <= crop:
        return x
    start = random.randint(0, len(x) - crop) if random_start else 0
    return x[start:start + crop]


def shape_read(x, input_mode, max_len):
    """A read (1, length) as the model takes it: zero-padded to max_len
    ('padded') or as is ('unpadded', 'variable')."""
    x = x.reshape(-1, 1).swapaxes(0, 1)
    if input_mode == 'padded':
        return pad_read_with_zeros(x, max_len=max_len)
    return x


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
        input_mode='padded',
        length_balance=False,
        with_runs=False,
        crop=0,
        random_window=False,
        skip_max=0,
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
        self.input_mode = input_mode
        self.length_balance = length_balance
        self.with_runs = with_runs
        self.crop = crop
        self.random_window = random_window
        self.skip_max = skip_max
        self.collate_fn = (collate_variable if input_mode == 'variable'
                           else None)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = UnlimitedReadsTrainingDataset(
                pos_experiments=self.train_pos_experiments,
                neg_experiments=self.train_neg_experiments,
                max_len=self.max_len,
                skip=self.skip,
                min_len=self.min_len,
                multiexp_generator_type=self.multiexp_generator_type,
                input_mode=self.input_mode,
                length_balance=self.length_balance,
                crop=self.crop,
                random_window=self.random_window,
                skip_max=self.skip_max,
            )
            self.valid_dataset = UnlimitedReadsValidDataset(
                pos_experiments=self.valid_pos_experiments,
                neg_experiments=self.valid_neg_experiments,
                valid_read_limit=self.valid_read_limit,
                skip=self.skip,
                max_len=self.max_len,
                min_len=self.min_len,
                multiexp_generator_type='uniform',
                input_mode=self.input_mode,
                with_runs=self.with_runs,
                crop=self.crop,
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
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size,
                          collate_fn=self.collate_fn)


class UnlimitedReadsTrainingDataset(IterableDataset):
    """
    Iterable Dataset that streams the selected reads of each experiment.
    Reads (after skip) outside min_len..max_len are dropped; longer than
    crop, they are cut to their first crop samples, or to a crop-long window
    at a random position (random_window). skip_max > skip: each read skips a
    random number of samples in skip..skip_max.
    """

    def __init__(self, pos_experiments, neg_experiments, max_len, skip,
                 min_len, multiexp_generator_type, input_mode='padded',
                 length_balance=False, crop=0, random_window=False,
                 skip_max=0):
        self.max_len = max_len
        self.min_len = min_len
        self.skip = skip
        self.skip_max = skip_max
        self.crop = crop
        self.random_window = random_window
        self.multiexp_generator_type = multiexp_generator_type
        self.input_mode = input_mode
        self.length_balance = length_balance

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
                        skip = (random.randint(self.skip, self.skip_max)
                                if self.skip_max > self.skip else self.skip)
                        x = process_pod5_read(read, skip=skip)
                        y = np.array(label)
                        if len(x) > self.max_len or len(x) < self.min_len:
                            continue
                        x = crop_read(x, self.crop, self.random_window)
                        # the length the model sees (length balancing)
                        length = len(x)
                        x = shape_read(x, self.input_mode, self.max_len)
                        yield x, np.array([y], dtype=np.float32), exp, length

    @staticmethod
    def experiment_size(batches):
        return sum(len(ids) for ids in batches.values())

    def get_stream(self):
        # each read carries the index of its experiment (run) within its
        # class, for run adversaries; other models ignore it
        pos_gens = [self.process_experiment(s, label=1, exp=i)
                    for i, s in enumerate(self.positive_experiments)]
        pos_sizes = [self.experiment_size(s)
                     for s in self.positive_experiments]
        neg_gens = [self.process_experiment(s, label=0, exp=i)
                    for i, s in enumerate(self.negative_experiments)]
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
        if self.length_balance:
            gen = length_balanced(gen, self.min_len, self.max_len)

        for x, y, exp, _ in gen:
            yield x, y, exp

    def __iter__(self):
        return self.get_stream()


class UnlimitedReadsValidDataset(Dataset):
    """
    Mapped Dataset of valid_read_limit reads, sampled at random from the
    experiments' selected reads; reads longer than crop are cut to their
    first crop samples
    """

    def __init__(self, pos_experiments, neg_experiments, max_len, skip,
                 min_len, multiexp_generator_type, valid_read_limit,
                 input_mode='padded', with_runs=False, crop=0):
        self.positive_experiments = [
            load_experiment(e) for e in pos_experiments]
        self.negative_experiments = [
            load_experiment(e) for e in neg_experiments]
        self.max_len = max_len
        self.min_len = min_len
        self.skip = skip
        self.multiexp_generator_type = multiexp_generator_type
        self.valid_read_limit = valid_read_limit
        self.input_mode = input_mode
        # items also carry the run index within the class (run adversaries)
        self.with_runs = with_runs
        self.crop = crop

        print('Generating valid dataset')
        self.items = self.generate_data()
        # only the sampled reads are kept
        del self.positive_experiments, self.negative_experiments

    def process_experiment(self, selection, label, run=0):
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
                        x = crop_read(x, self.crop)
                        x = shape_read(x, self.input_mode, self.max_len)
                        yield x, np.array([y], dtype=np.float32), run

    def generate_data(self):
        pos_gens = [self.process_experiment(s, label=1, run=i)
                    for i, s in enumerate(self.positive_experiments)]
        pos_sizes = [sum(len(ids) for ids in s.values())
                     for s in self.positive_experiments]
        neg_gens = [self.process_experiment(s, label=0, run=i)
                    for i, s in enumerate(self.negative_experiments)]
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
            x, y, run = next(gen)
            items.append((x, y, run) if self.with_runs else (x, y))
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class InferenceDataset(IterableDataset):
    """
    Iterable Dataset that contains all reads from POD5 files, or only the
    reads in read_ids (a set) if given; reads longer than crop are cut to
    their first crop samples
    """

    def __init__(self, files, max_len, skip, min_len, pad_reads=True,
                 read_ids=None, crop=0):
        self.files = files
        self.crop = crop
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
                        # max_len 0: no upper limit
                        if ((self.max_len and len(x) > self.max_len)
                                or len(x) < self.min_len):
                            continue
                        x = crop_read(x, self.crop)
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
