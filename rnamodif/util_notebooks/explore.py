from rnamodif.data_utils.datamap import experiment_files, experiments_dict
from Fast5Fetch.fast5fetch.fast5data import get_all_fast5s
from ont_fast5_api.fast5_interface import get_fast5_file
from matplotlib import pyplot as plt
from rnamodif.data_utils.sorting import get_experiment_sort
from statistics import mean
from scipy import stats
import numpy as np
from itertools import islice
from bonito_pulled.bonito.reader import trim
import random
from rnamodif.data_utils.trimming import primer_trim
import pickle
import pandas as pd


def get_read_lengths(exp_list):
    exp_lengths = {}
    for k in exp_list:
        file_lengths = {}
        index_func = get_experiment_sort(k)
        for f in sorted(experiment_files[k], key=index_func):
            # print(f.stem)
            lengths = []
            with get_fast5_file(f, mode='r') as f5:
                for i,read in enumerate(f5.get_reads()):
                    x = read.get_raw_data(scale=True)
                    lengths.append(len(x))
            file_lengths[f.stem]=lengths
        exp_lengths[k]= file_lengths
        print(f'EXP {k} done')
    return exp_lengths

def plot_avg_lengths(exp_lengths):
    #TODO add medians
    fig, ax = plt.subplots(figsize=(8,4))
    for j, (exp, file_lengths) in enumerate(exp_lengths.items()):
        means = []
        for i,(k,v) in enumerate(file_lengths.items()):
            means.append(int(mean(v)))
        ax.plot(means, label=exp)

    ax.set_title('READS LENGTHS DIFFERENCES')
    ax.set_ylabel('AVG read length')
    ax.set_xlabel('Experiment file order (suffix number)')
    ax.legend(loc="upper left")
    
def process_fast5_read(read, skip=0, zscore=True, scale=True, smartskip=True, exp_label=None):
    """ Normalizes and extracts specified region from raw signal """
    s = read.get_raw_data(scale=scale)  
    if zscore:
        s = stats.zscore(s)
        
    if(smartskip):
        skip = primer_trim(signal=s[:26000])
        
    return s, skip

def myite(files, read_index, file_index, zscore, scale, smartskip, exp_label):
    while True:
        if(not file_index):
            fast5 = random.choice(files)
        else:
            fast5 = files[file_index]
        with get_fast5_file(fast5, mode='r') as f5:
            read = next(islice(f5.get_reads(), read_index, None))
            x, cutoff = process_fast5_read(read, zscore=zscore, scale=scale, smartskip=smartskip, exp_label=exp_label)
            yield x.reshape(-1,1).swapaxes(0,1), cutoff
            

def plot_reads(exp_list, zscore=True, scale=True, total_limit=None, read_index=0, file_index=None, y_axis_lim=None):
    smartskip = True
    
    seqs = {}
    cutoffs = {}
    for exp in exp_list:
        ite = myite(experiment_files[exp], read_index=read_index, file_index=file_index, zscore=zscore, scale=scale, smartskip=smartskip, exp_label=exp) # picking from the first 100 reads
        sample = next(ite)
        seqs[exp] = sample[0][0]
        cutoffs[exp] = sample[1]
    
    fig, axs = plt.subplots(len(exp_list), sharey=True, sharex=True, figsize=(10,len(exp_list)*2.5))
    max_x = max([len(seq) for seq in seqs.values()])
    
    for i,exp in enumerate(exp_list):
        axs[i].plot(seqs[exp])
        if(y_axis_lim):
            axs[i].set_ylim(y_axis_lim)
        axs[i].hlines(y=[-2,2], xmin=0, xmax=max_x, colors='red')
        axs[i].vlines(x=cutoffs[exp], color='purple',ymin=-5, ymax=5)
        axs[i].set_title(exp)
        

def precompute_lengths():
    exp_lengths = {}
    for k in experiment_files.keys():
        file_lengths = {}
        
        for f in experiment_files[k]:
            # print(f.stem)
            lengths = []
            with get_fast5_file(f, mode='r') as f5:
                for i,read in enumerate(f5.get_reads()):
                    x = read.get_raw_data(scale=True)
                    lengths.append(len(x))
            file_lengths[f.stem]=lengths
        exp_lengths[k]= file_lengths
        print(f'EXP {k} done')


    with open('saved_lengths.pkl', 'wb') as f:
        pickle.dump(exp_lengths, f)
        
def get_lengths_df():
    data = []
    with open('saved_lengths.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
        for exp in loaded_dict.keys():
            # print('#'*10, exp)
            subdict = loaded_dict[exp]
            for fast5file in subdict.keys():
                lengths_list = subdict[fast5file]
                for length in lengths_list:
                    datapoint = {
                        'exp':exp,
                        'file':fast5file,
                        'len':length,

                    }
                    data.append(datapoint)
                # print(mean(lengths_list))
    df = pd.DataFrame(data)
    return df

def plot_len_dist(exp_list, df, log=True):
    fig, axs = plt.subplots(len(exp_list), sharex=True, sharey=True, figsize=(10,len(exp_list)*2.5))
    plt.ticklabel_format(useOffset=False,style='plain')
    if(log):
        plt.xticks(np.arange(0, 1e7, 5e5))
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    for i,exp in enumerate(exp_list):
        sub_df = df[df['exp']==exp]
        lengths = sub_df['len'].values
        axs[i].hist(lengths, bins=50, cumulative=False, log=log, density=False)
        axs[i].set_title(exp)
        print(f'{exp} max length',max(lengths))