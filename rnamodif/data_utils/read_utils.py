from scipy import stats
import torch
import numpy as np
from rnamodif.data_utils.trimming import primer_trim

def med_mad(x, factor=1.4826):
    med = np.median(x)
    mad = np.median(np.absolute(x - med)) * factor
    return med, mad

def rodan_normalize(signal):
    med, mad = med_mad(signal)
    signal = (signal - med) / mad
    return signal

norm_dict = {
    'zscore':stats.zscore,
    'rodan':rodan_normalize
}
def process_read(read, window, normalization, trim_primer):
    s = read.get_raw_data(scale=True)
    s = norm_dict[normalization](s)
    
    skip = 0
    if(trim_primer):
        skip = primer_trim(signal=s[:26000]) #TODO remove 26000 limit?
    
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