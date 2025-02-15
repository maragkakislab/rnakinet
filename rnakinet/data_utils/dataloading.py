import random
import numpy as np
from ont_fast5_api.fast5_interface import get_fast5_file
from pod5 import Reader
from torch.utils.data import IterableDataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from rnakinet.data_utils.read_utils import process_fast5_read, process_pod5_read

class UnlimitedReadsInferenceDatasetFast5(IterableDataset):
    """
    Iterable Dataset that contains all reads from FAST5 files
    """

    def __init__(self, files, max_len, skip, min_len):
        self.files = files
        self.max_len = max_len
        self.skip = skip
        self.min_len = min_len

    def process_files_fully(self, files):
        for fast5 in files:
            try:
                with get_fast5_file(fast5, mode='r') as f5:
                    for i, read in enumerate(f5.get_reads()):
                        x = process_fast5_read(read, skip=self.skip)
                        start = 0
                        stop = len(x)
                        if(len(x) > self.max_len or len(x) < self.min_len):
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
    
class UnlimitedReadsInferenceDatasetPod5(IterableDataset):
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
                        start = 0
                        stop = len(x)
                        if len(x) > self.max_len or len(x) < self.min_len:
                            continue
                        identifier = {
                            'file': str(pod5_file),
                            'readid': str(read.read_id),
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