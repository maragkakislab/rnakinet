import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
from plot_helpers import setup_palette
import seaborn as sns

def plot_fig(df, column, save_path):
    FC = df['log2FoldChange']
    padj = df[column]
    log_col = -np.log10(padj)

    
    fontsize=8
    mpl.rc('font',family='Arial')
    plt.figure(figsize=(1.5,1.5))
    palette = setup_palette()
    
    plt.scatter(FC, log_col, s=.25, alpha=0.3)
    plt.xlabel('log2FC', fontsize=fontsize)
    plt.ylabel(f'-log10({column})', fontsize=fontsize)
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    sns.set_style('whitegrid')
    sns.despine()
    plt.savefig(save_path, bbox_inches='tight')


def main(args):
    df = pd.read_csv(args.table_path, sep='\t', header=0)
    plot_fig(df, args.column, args.save_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make volcano plot from differential expression analysis data.")
    parser.add_argument("--table-path", help="tab file with deseq data. stdin if -")
    parser.add_argument("--save-path", help="filename to save volcano plot")
    parser.add_argument("--column", type=str, help="which column to plot on the y axis")
    
    args = parser.parse_args()
    main(args)