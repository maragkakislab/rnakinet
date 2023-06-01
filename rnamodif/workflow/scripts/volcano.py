import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def plot_fig(df, column, save_path):
    FC = df['log2FoldChange']
    padj = df[column]
    log_col = -np.log10(padj)

    plt.scatter(FC, log_col, s=.25)
    plt.xlabel('log2FC')
    # plt.ylim(0, 0.5)
    plt.ylabel(f'-log10({column})')
    plt.savefig(save_path)


def main(args):
    df = pd.read_csv(args.table_path, sep='\t', header=0)
    plot_fig(df, args.column, args.save_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make volcano plot from differential expression analysis data.")
    parser.add_argument("--table-path", help="tab file with deseq data. stdin if -")
    parser.add_argument("--save-path", help="filename to save volcano plot")
    parser.add_argument("--column", help="which column to plot on the y axis")
    
    args = parser.parse_args()
    main(args)