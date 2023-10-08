import pysam
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import argparse
import seaborn as sns
from plot_helpers import setup_palette
from sklearn import metrics
import matplotlib as mpl
import json

import pysam
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def extract_reads_info(bam_file):
    # Open the BAM file
    with pysam.AlignmentFile(bam_file, 'rb') as file:
        num_reads = 0
        read_lengths = []

        for read in file:
            num_reads += 1
            read_lengths.append(len(read.query_sequence))
    return num_reads, read_lengths

def main(args):
    exp_to_size = {}
    exp_to_lengths = {}
    
    
    for name,bam in zip(args.experiment_names, args.bam_files):
        num_reads, read_lengths = extract_reads_info(bam)
        exp_to_size[name] = num_reads
        exp_to_lengths[name] = read_lengths
        
    df_sizes = pd.DataFrame(list(exp_to_size.items()), columns=['Experiment_name', 'Number_of_reads'])
    # df_lengths = pd.DataFrame(list(exp_to_lengths.items()), columns=['Experiment_name', 'Lengths'])
    df_lengths = pd.DataFrame()
    to_add = []
    
    for exp, lengths in exp_to_lengths.items():
        for length in lengths:
            to_add.append({'exp':exp,'Lengths':length})
    df_lengths = df_lengths.append(to_add, ignore_index=True)
    
    df_sizes.to_csv(args.sizes_output)
    
    plt.figure(figsize=(1.5,1.5))
    mpl.rc('font',family='Arial')
    fontsize=8
    palette = setup_palette()
    sns.histplot(df_lengths, x='Lengths', hue='exp', fill=False, common_norm=False, element='step')
    
    plt.title(args.group, fontsize=fontsize)
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    plt.xlim(0,6000)
    sns.set_style('whitegrid')
    sns.despine()
    
    plt.savefig(args.lengths_output, bbox_inches='tight')
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bam_files', nargs='+', type=str, required=True, help='Bam files for computing stats')
    parser.add_argument('--experiment_names', nargs='+', type=str, required=True, help='Experiment names in the order of bam files')
    parser.add_argument('--sizes_output', type=str, required=True, help='Path to the output dataframe with sizes.')
    parser.add_argument('--lengths_output', type=str, required=True, help='Path to the output plot file.')
    parser.add_argument('--group', type=str, required=True)
    
    
    args = parser.parse_args()
    main(args)