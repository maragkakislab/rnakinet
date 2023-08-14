import argparse
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path


def main(args):
    gene_preds = pd.read_csv(args.gene_predictions, sep='\t')
    print(gene_preds.head())
    gene_halflifes = pd.read_csv(args.gene_halflifes, sep='\t')
    gene_halflifes = clean_halflifes_df(gene_halflifes)
    
    gene_join = gene_preds.merge(gene_halflifes, how='left', left_on='Gene stable ID', right_on='Gene.stable.ID')
    gene_join = gene_join[gene_join['t5'].notnull()]
    
    # TODO filter bad genes
    # TODO parametrize 
    gene_join = gene_join[gene_join['reads'] > 100] #100
    gene_join = gene_join[gene_join['t5'] < 5] #6 
    
    gene_join = add_predicted_halflifes(gene_join)
    gene_join['t5_binned'] = pd.cut(gene_join['t5'], bins=[0, 1, 2, 3, 4, 5], labels=['0-1', '1-2', '2-3', '3-4', '4-5'])
    
    x = gene_join['t5'].values
    y = gene_join['pred_t5'].values
    
    # Regression plot
    sns.regplot(data=gene_join, x='t5', y='pred_t5', 
            scatter_kws={'alpha':0.6, 's':25, 'color':'blue'}, 
            line_kws={"color": "red", "lw": 2},  # Add regression line color and width
            label=f'r={np.corrcoef(x,y)[0,1]:.2f}',
    )
    
    
    plt.xlabel('t5 Eisen',fontsize=16)
    plt.ylabel('t5 predicted',fontsize=16)
    plt.legend(loc='upper left', fontsize=16, frameon=False)
    
    sns.set_style('whitegrid')
    sns.despine()
    
    plt.savefig(args.output, bbox_inches = 'tight')
    
    # Boxplots
    plt.figure()
    sns.boxplot(data=gene_join, x='t5_binned', y='pred_t5')
    plt.xlabel('t5 Eisen',fontsize=16)
    plt.ylabel('t5 predicted',fontsize=16)
    plt.legend(loc='upper left', fontsize=16, frameon=False)
    
    # sns.set_style('whitegrid')
    plt.grid(axis='y', linestyle='-', which='major', color='white', linewidth=0.5) # Remove or set white color for horizontal grid lines
    
    sns.despine()
    
    path = Path(args.output)
    new_path = path.with_stem(path.stem + "_boxplot")
    plt.savefig(new_path, bbox_inches = 'tight')
    
    

def add_predicted_halflifes(df):
    tl = args.tl
    # col = 'average_score'
    col = 'percentage_modified'
    #TODO which statistic?
    # y = gene_join['percentage_modified'].values
    
    df['pred_t5'] = -tl * np.log(2) / np.log(1-df[col]) #ORIG
    # df['pred_t5'] = df[col]
    # df['pred_t5'] = -tl * np.log(2) / np.log(df[col])
    
    print(df[['t5','pred_t5','percentage_modified','average_score','reads','Gene.stable.ID']].head())
    # df['pred_t5'] = -tl * np.log(2) / np.log(df[col]) #TODO DELETE
    
    # gene_join['pred_t5'] = -tl * np.log(2) / np.log(1-(1/(1+((1-gene_join[col])/gene_join[col]))))
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[~df['pred_t5'].isna()]
    
    # df = df[(df['percentage_modified']>0.15) & (df['percentage_modified']<0.9)] #Matt paper figure recommendation
    
    return df
    

def clean_halflifes_df(df):
    #grouping by gene and taking only genes that have non-conflicting t5 values
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
    
    
    