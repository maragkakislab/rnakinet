import argparse
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
from plot_helpers import setup_palette



def main(args):
    gene_preds = pd.read_csv(args.table_a, sep='\t')
    gene_halflifes = pd.read_csv(args.table_b, sep='\t')
    
    gene_halflifes = clean_halflifes_df(gene_halflifes, group_col=args.table_b_column, halflife_column=args.halflife_column)
        
    gene_join = gene_preds.merge(gene_halflifes, how='left', left_on=args.table_a_column, right_on=args.table_b_column)
    for col in gene_join.columns:
        gene_join = gene_join[gene_join[col].notnull()]
    
    gene_join.to_csv(args.output, sep='\t')
    
def clean_halflifes_df(df, group_col, halflife_column):
    #grouping by gene and taking only genes that have non-conflicting t5 values
    #Multiple transcripts per gene, all have the same halflife measured, dropping duplicates
    clean_df = df.groupby(group_col).first().reset_index()
    clean_df = clean_df[clean_df[halflife_column]!='--']
    clean_df[halflife_column] = pd.to_numeric(clean_df[halflife_column])
    return clean_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prediction on FAST5 files and save results in a pickle file.')
    parser.add_argument('--table-a', type=str, required=True)
    parser.add_argument('--table-b', type=str, required=True)
    # parser.add_argument('--min-reads', type=int, required=True)
    # parser.add_argument('--max-measured-halflife', type=int, required=True)
    parser.add_argument('--table-a-column', type=str, required=True)
    parser.add_argument('--table-b-column', type=str, required=True)
    parser.add_argument('--halflife-column', type=str, required=True)
    parser.add_argument('--output', type=str, help='Path to the output plot.')
    
    args = parser.parse_args()
    main(args)
    
    
    