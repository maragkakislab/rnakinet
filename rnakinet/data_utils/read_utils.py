import torch
import numpy as np

def process_read(read, skip=0):
    s = read.get_raw_data(scale=True)
    med = np.median(s)
    mad = 1.4826 * np.median(np.absolute(s-med))
    s = (s - med) / mad
    return s[skip:]



