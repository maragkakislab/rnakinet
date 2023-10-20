import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import argparse
from plot_helpers import setup_palette
from sklearn import metrics
from auroc import plot_auroc
from matplotlib import pyplot as plt
import pandas as pd

def main(args):
    mpl.rc('font',family='Arial')
    plt.figure(figsize=(1.5,1.5))
    
    setup_palette()
    
    neg_dfs = [pd.read_csv(filename, sep='\t') for filename in args.negatives_paths]
    pos_dfs = [pd.read_csv(filename, sep='\t') for filename in args.positives_paths]

    allneg_df = pd.concat(neg_dfs)
    allpos_df = pd.concat(pos_dfs)

    neg_preds = allneg_df['pred'].values.tolist()
    pos_preds = allpos_df['pred'].values.tolist()

    plot_auroc(pos_preds, neg_preds, pos_name='Positives', neg_name='Negatives')
        
    plt.savefig(args.output, bbox_inches='tight')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prediction on FAST5 files and save results in a pickle file.')
    parser.add_argument(
        '--positives-paths', 
        type=str, 
        required=True, 
        nargs='+', 
        help='Paths to the files containing predictions on positives.'
    )
    parser.add_argument(
        '--negatives-paths', 
        type=str, 
        required=True, 
        nargs='+', 
        help='Paths to the files containing predictions on negatives.'
    )
    parser.add_argument('--output', type=str, required=True, help='Plot AUROC plot for the nanoid predictions')
    args = parser.parse_args()
    main(args)
  
