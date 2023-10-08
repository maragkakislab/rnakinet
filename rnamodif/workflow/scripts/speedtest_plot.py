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

def main(args):
    logs = []
    for json_path in args.jsons:
        with open(json_path, 'r') as handle:
            log = json.load(handle)
            logs.append(log)
    
    df = pd.DataFrame(logs)
    
    assert df['GPU type'].nunique() == 1
    
    output_file = args.output
    mpl.rc('font',family='Arial')
    fontsize=8
    plt.figure(figsize=(1.5,1.5))
    palette = setup_palette()
    
    df['total_elapsed_minutes'] = df['total_elapsed']/60
    
#     plt.plot(fpr,tpr, linewidth=1, color=palette[i], label=f'{thresholds[i]} <= {key}  < {thresholds[i+1]} (AUROC {auroc:.2f})')
    sns.lineplot(data=df, x='using_reads', y='total_elapsed_minutes', marker='.')

    plt.xlabel('Number of reads', fontsize=fontsize)
    plt.ylabel('Runtime in minutes', fontsize=fontsize)
    plt.xticks([0,1000000], ['0', '1e6'], fontsize=fontsize-2)
    plt.yticks([0,50,100], fontsize=fontsize-2)
    plt.title(df['GPU type'].values[0], fontsize=fontsize)
    sns.set_style('whitegrid')
    sns.despine()
#     plt.legend(loc='lower right', fontsize=fontsize-2, frameon=False)
    plt.savefig(output_file, bbox_inches='tight')
                                               
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsons', nargs='+', type=str, required=True, help='Json files for speedtest stats')
    parser.add_argument('--output', type=str, required=True, help='Path to the output plot file.')
    args = parser.parse_args()
    main(args)