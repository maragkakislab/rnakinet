import argparse
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
from plot_helpers import setup_palette


def main(args):
    gene_preds = pd.read_csv(args.gene_predictions, sep='\t')
    gene_preds = add_predicted_halflifes(gene_preds)
    gene_preds.to_csv(args.output, sep='\t')
    
def add_predicted_halflifes(df):
    tl = args.tl
    col = 'percentage_modified'
    df['pred_t5'] = -tl * np.log(2) / np.log(1-df[col]) 
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[~df['pred_t5'].isna()]
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prediction on FAST5 files and save results in a pickle file.')
    parser.add_argument('--gene-predictions', type=str, required=True)
    parser.add_argument('--tl', type=float, required=True, help='Time parameter for the decay equation')
    parser.add_argument('--output', type=str, help='Path to the output plot.')
    
    args = parser.parse_args()
    main(args)
    
    
    