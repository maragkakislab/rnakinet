import argparse
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def main(args):
    gene_preds = pd.read_csv(args.gene_predictions, sep='\t')
    gene_halflifes = pd.read_csv(args.gene_halflifes, sep='\t')
    gene_halflifes = clean_halflifes_df(gene_halflifes)
    
    gene_join = gene_preds.merge(gene_halflifes, how='left', left_on='Gene stable ID', right_on='Gene.stable.ID')
    gene_join = gene_join[gene_join['t5'].notnull()]
    
    # TODO filter bad genes
    gene_join = gene_join[gene_join['reads'] > 20]
    gene_join = gene_join[gene_join['t5'] < 6]
    
    gene_join = add_predicted_halflifes(gene_join)
    
    x = gene_join['t5'].values
    y = gene_join['pred_t5'].values
    
    plt.scatter(x,y, s=0.5) 
    plt.title(f'Corr coef {np.corrcoef(x,y)[0,1]}')
    plt.savefig(args.output, bbox_inches = 'tight')
    

def add_predicted_halflifes(df):
    tl = args.tl
    col = 'average_score'
    #TODO which statistic?
    # y = gene_join['percentage_modified'].values
    
    df['pred_t5'] = -tl * np.log(2) / np.log(1-df[col])
    # gene_join['pred_t5'] = -tl * np.log(2) / np.log(1-(1/(1+((1-gene_join[col])/gene_join[col]))))
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[~df['pred_t5'].isna()]
    return df
    

def clean_halflifes_df(df):
    grouped_df = df.groupby('Gene.stable.ID')
    bool_map_of_nonuniques = (grouped_df['t5'].nunique()!=1)
    nonunique_genes = bool_map_of_nonuniques[bool_map_of_nonuniques].index
    
    #dropping genes with various t5 values - TODO just take first?
    unique_df = df[~df['Gene.stable.ID'].isin(nonunique_genes)]
    grouped_unique_df = unique_df.groupby('Gene.stable.ID')
    assert (grouped_unique_df['t5'].nunique()==1).all()
    clean_df = grouped_unique_df.first().reset_index()
    return clean_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prediction on FAST5 files and save results in a pickle file.')
    parser.add_argument('--gene-predictions', type=str, required=True)
    parser.add_argument('--gene-halflifes', type=str, required=True)
    parser.add_argument('--tl', type=float, required=True, help='Time parameter for the decay equation')
    
    parser.add_argument('--output', type=str, help='Path to the output plot.')
    
    args = parser.parse_args()
    main(args)
    
    
    