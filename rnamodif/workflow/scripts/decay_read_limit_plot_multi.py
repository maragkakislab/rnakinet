import argparse
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
from plot_helpers import setup_palette
import matplotlib as mpl

def main(args):
    fontsize=8
    mpl.rc('font',family='Arial')
    plt.figure(figsize=(1.5,1.5))
    palette = setup_palette()
    
    for gene_predictions, gene_halflifes, tl, exp_name in zip(args.gene_predictions_list, args.gene_halflifes_list, args.tl_list, args.exp_name_list):
        plot_read_limit(gene_predictions, gene_halflifes, args.gene_halflifes_gene_column, tl, exp_name)
    
    plt.xlabel('Minimum read requirement',fontsize=fontsize)
    plt.ylabel('Correlation of measured (Eisen)\n and predicted halflife',fontsize=fontsize)
    plt.yticks(fontsize=fontsize-2)
    plt.xticks(fontsize=fontsize-2)
    
    plt.legend(loc='lower right', fontsize=fontsize-2, frameon=False)
    
    sns.set_style('whitegrid')
    sns.despine()
    
    plt.savefig(args.output, bbox_inches = 'tight')

def plot_read_limit(gene_predictions, gene_halflifes, column, tl, exp_name):
    gene_preds = pd.read_csv(gene_predictions, sep='\t')
    gene_halflifes = pd.read_csv(gene_halflifes, sep='\t')
    gene_halflifes = clean_halflifes_df(gene_halflifes, group_col=column)
        
    key_map = {
        'gene': 'Gene stable ID',
        'transcript': 'Transcript stable ID',
    }
    
    gene_join = gene_preds.merge(gene_halflifes, how='left', left_on=key_map[column], right_on=column)
    gene_join = gene_join[gene_join['t5'].notnull()]
    
    gene_join = gene_join[gene_join['t5'] < 5]
    gene_join = add_predicted_halflifes(gene_join, tl)
    
    x = gene_join['t5'].values
    y = gene_join['pred_t5'].values
    
    limits = range(0, gene_join['reads'].max())
    corrs = []
    marks = [10,50]
    last_mark = 0
    min_reads_to_plot = 30
    for read_limit in limits:
        sub_gene_join = gene_join[gene_join['reads']>read_limit]
        if(len(sub_gene_join) < min_reads_to_plot):
            break
        
        corr = np.corrcoef(sub_gene_join['t5'].values, sub_gene_join['pred_t5'].values)[0,1]
        
        corrs.append(corr)
        
    plt.plot(limits[:len(corrs)], corrs, label=exp_name)
        
def add_predicted_halflifes(df, tl):
    col = 'percentage_modified'
    df['pred_t5'] = -tl * np.log(2) / np.log(1-df[col]) 
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[~df['pred_t5'].isna()]
    return df
    
def clean_halflifes_df(df, group_col):
    clean_df = df.groupby(group_col).first().reset_index()
    clean_df = clean_df[clean_df['t5']!='--']
    clean_df['t5'] = pd.to_numeric(clean_df['t5'])
    return clean_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prediction on FAST5 files and save results in a pickle file.')
    parser.add_argument('--gene-predictions-list', nargs='+', type=str, required=True)
    parser.add_argument('--gene-halflifes-list', nargs='+', type=str, required=True)
    parser.add_argument('--gene-halflifes-gene-column', type=str, required=True)
    parser.add_argument('--tl-list', nargs='+', type=float, required=True, help='Time parameter for the decay equation')
    parser.add_argument('--exp-name-list', nargs='+', type=str, required=True)
    
    parser.add_argument('--output', type=str, help='Path to the output plot.')
    
    args = parser.parse_args()
    main(args)
    
    
    