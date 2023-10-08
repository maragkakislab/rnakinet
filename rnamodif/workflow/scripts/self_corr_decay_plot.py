import argparse
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
from plot_helpers import setup_palette, correlation_plot


def main(args):
    gene_preds_1 = pd.read_csv(args.gene_predictions_1, sep='\t')
    gene_preds_2 = pd.read_csv(args.gene_predictions_2, sep='\t')
    
    gene_preds_1 = gene_preds_1[gene_preds_1['reads'] > 100] #100
    gene_preds_2 = gene_preds_2[gene_preds_2['reads'] > 100] #100
    
    gene_preds_1 = add_predicted_halflifes(gene_preds_1)
    gene_preds_2 = add_predicted_halflifes(gene_preds_2)
    
    key_map = {
        'gene': 'Gene stable ID',
        'transcript': 'Transcript stable ID',
    }
    
    gene_join = gene_preds_1.merge(gene_preds_2, how='inner', left_on=key_map[args.reference_level], right_on=key_map[args.reference_level])
    
    correlation_plot(gene_join, x_column='pred_t5_x',y_column='pred_t5_y', x_label='t5 predicted (replicate 1)',y_label='t5 predicted (replicate 2)', output=args.output)    
    

def add_predicted_halflifes(df):
    tl = args.tl
    # col = 'average_score'
    col = 'percentage_modified'
    #TODO which statistic?
    # y = gene_join['percentage_modified'].values
    
    df['pred_t5'] = -tl * np.log(2) / np.log(1-df[col]) #ORIG
    # df['pred_t5'] = df[col]
    # df['pred_t5'] = -tl * np.log(2) / np.log(df[col])
    
    # print(df[['t5','pred_t5','percentage_modified','average_score','reads','Gene.stable.ID']].head())
    # df['pred_t5'] = -tl * np.log(2) / np.log(df[col]) #TODO DELETE
    
    # gene_join['pred_t5'] = -tl * np.log(2) / np.log(1-(1/(1+((1-gene_join[col])/gene_join[col]))))
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[~df['pred_t5'].isna()]
    
    # df = df[(df['percentage_modified']>0.15) & (df['percentage_modified']<0.9)] #Matt paper figure recommendation
    
    return df
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prediction on FAST5 files and save results in a pickle file.')
    parser.add_argument('--gene-predictions-1', type=str, required=True)
    parser.add_argument('--gene-predictions-2', type=str, required=True)
    parser.add_argument('--tl', type=float, required=True, help='Time parameter for the decay equation')
    parser.add_argument('--reference-level', type=str, help='Reference level (gene or transcript)')
    
    parser.add_argument('--output', type=str, help='Path to the output plot.')
    
    args = parser.parse_args()
    main(args)
    
    
    