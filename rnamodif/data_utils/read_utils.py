from scipy import stats
import torch
import numpy as np
from rnamodif.data_utils.trimming import primer_trim

def process_read(read, window):
    s = read.get_raw_data(scale=True)
    med = np.median(s)
    mad = 1.4826 * np.median(np.absolute(s-med))
    s = (s - med) / mad
    
    skip = 0
    if(not window):
        return s[skip:]
    
    last_start_index = len(s)-window
    if(last_start_index < skip):
        # if sequence is not long enough, last #window signals is taken, ignoring the skip index
        skip = last_start_index

    #Using torch rand becasue of multiple workers
    pos = torch.randint(skip, last_start_index+1, (1,))
    if(len(s) < window):
        return []
        
    #TODO remove reshape
    return s[pos:pos+window].reshape((window, 1))

