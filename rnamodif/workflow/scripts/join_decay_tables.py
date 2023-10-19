import argparse
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
from plot_helpers import setup_palette, correlation_plot



def main(args):
    gene_preds = pd.read_csv(args.predicted_halflives, sep='\t')
    gene_halflifes = pd.read_csv(args.measured_halflives)
    
    gene_halflifes = clean_halflifes_df(gene_halflifes, group_col=args.reference_level)
        
    #TODO refactor away
    key_map = {
        'gene': 'Gene stable ID',
        'transcript': 'Transcript stable ID',
    }
    column = 't5' 
    gene_join = gene_preds.merge(gene_halflifes, how='left', left_on=key_map[args.reference_level], right_on=args.reference_level)
    gene_join = gene_join[gene_join[column].notnull()]
    
    gene_join = gene_join[gene_join['reads'] > args.min_reads]
    gene_join = gene_join[gene_join[column] < args.max_measured_halflife]
    gene_join.to_csv(args.output, sep='\t')
    
    
def clean_halflifes_df(df, group_col):
    #grouping by gene and taking only genes that have non-conflicting t5 values
    #Multiple transcripts per gene, all have the same halflife measured, dropping duplicates
    clean_df = df.groupby(group_col).first().reset_index()
    clean_df = clean_df[clean_df['t5']!='--']
    clean_df['t5'] = pd.to_numeric(clean_df['t5'])
    return clean_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prediction on FAST5 files and save results in a pickle file.')
    parser.add_argument('--predicted-halflives', type=str, required=True)
    parser.add_argument('--measured-halflives', type=str, required=True)
    parser.add_argument('--min-reads', type=int, required=True)
    parser.add_argument('--max-measured-halflife', type=int, required=True)
    parser.add_argument('--reference-level', type=str, required=True)
    parser.add_argument('--output', type=str, help='Path to the output plot.')
    
    args = parser.parse_args()
    main(args)
    
    
    