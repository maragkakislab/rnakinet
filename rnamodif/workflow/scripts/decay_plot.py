import argparse
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
from plot_helpers import setup_palette, correlation_plot



def main(args):
    gene_preds = pd.read_csv(args.gene_predictions, sep='\t')
    gene_halflifes = pd.read_csv(args.gene_halflifes)
    gene_halflifes = clean_halflifes_df(gene_halflifes, group_col=args.gene_halflifes_gene_column)
        
    key_map = {
        'gene': 'Gene stable ID',
        'transcript': 'Transcript stable ID',
    }
    
    column = 't5' 
    gene_join = gene_preds.merge(gene_halflifes, how='left', left_on=key_map[args.gene_halflifes_gene_column], right_on=args.gene_halflifes_gene_column)
    gene_join = gene_join[gene_join[column].notnull()]
    
    gene_join = gene_join[gene_join['reads'] > 100] 
    gene_join = gene_join[gene_join[column] < 5] 
    
    gene_join = add_predicted_halflifes(gene_join)
    gene_join['t5_binned'] = pd.cut(gene_join[column], bins=[0, 1, 2, 3, 4, 5], labels=['0-1', '1-2', '2-3', '3-4', '4-5'])
    
    correlation_plot(gene_join, x_column=column,y_column='pred_t5', x_label='t5 measured',y_label='t5 predicted', output=args.output)

def add_predicted_halflifes(df):
    tl = args.tl
    col = 'percentage_modified'
    df['pred_t5'] = -tl * np.log(2) / np.log(1-df[col]) #ORIG
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
    parser.add_argument('--gene-predictions', type=str, required=True)
    parser.add_argument('--gene-halflifes', type=str, required=True)
    parser.add_argument('--gene-halflifes-gene-column', type=str, required=True)
    parser.add_argument('--tl', type=float, required=True, help='Time parameter for the decay equation')
    
    parser.add_argument('--output', type=str, help='Path to the output plot.')
    
    args = parser.parse_args()
    main(args)
    
    
    