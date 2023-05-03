import torch
import numpy as np
from rnamodif.data_utils.trimming import primer_trim


def process_read(read, window, skip=0):
    """
    Normalizes a single read and selects an area of the signal to return

    Keyword arguments:
    read: Read from a fast5 file
    window: Size of a random chunk of signal to return. If None the whole signal is returned.

    Optional keyword arguments:
    skip: The length of the starting chunk of a signal to skip. Default: 0
    """

    # Standardizing signal according to RODAN preprocessing
    s = read.get_raw_data(scale=True)
    med = np.median(s)
    mad = 1.4826 * np.median(np.absolute(s-med))
    s = (s - med) / mad

    # Returning the whole signal
    if (not window):
        return s[skip:]

    # If the sequence is not long enough, last #window numbers is taken, ignoring the skip index
    last_start_index = len(s)-window
    if (last_start_index < skip):
        skip = last_start_index

    # Selecting a random chunk start of the signal. Using torch rand becasue of pytorch workers implementation
    pos = torch.randint(skip, last_start_index+1, (1,))

    # If the signal is too short, returning empty array to signal error
    if (len(s) < window):
        return []

    # TODO remove reshape?
    # Returning a random chunk of #window size
    return s[pos:pos+window].reshape((window, 1))
