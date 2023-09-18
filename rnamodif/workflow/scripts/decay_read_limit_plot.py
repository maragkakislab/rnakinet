import argparse
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
from plot_helpers import setup_palette


def main(args):
    gene_preds = pd.read_csv(args.gene_predictions, sep='\t')
    gene_halflifes = pd.read_csv(args.gene_halflifes)
    gene_halflifes = clean_halflifes_df(gene_halflifes, group_col=args.gene_halflifes_gene_column)
        
    #TODO refactor away
    key_map = {
        'gene': 'Gene stable ID',
        'transcript': 'Transcript stable ID',
    }
    
    gene_join = gene_preds.merge(gene_halflifes, how='left', left_on=key_map[args.gene_halflifes_gene_column], right_on=args.gene_halflifes_gene_column)
    gene_join = gene_join[gene_join['t5'].notnull()]
    
    # gene_join = gene_join[gene_join['reads'] > 100] #100
    gene_join = gene_join[gene_join['t5'] < 5] #6 
    gene_join = add_predicted_halflifes(gene_join)
    
    x = gene_join['t5'].values
    y = gene_join['pred_t5'].values
    
    palette = setup_palette()
    
    fontsize=18
    
    limits = range(0, gene_join['reads'].max())
    corrs = []
    marks = [10,50]
    last_mark = 0
    for read_limit in limits:
        sub_gene_join = gene_join[gene_join['reads']>read_limit]
        corr = np.corrcoef(sub_gene_join['t5'].values, sub_gene_join['pred_t5'].values)[0,1]
        
        corrs.append(corr)
        if(len(sub_gene_join) in marks and last_mark!=len(sub_gene_join)):
            last_mark=len(sub_gene_join)
            plt.axvline(x=read_limit, color=palette[1],linestyle='--', linewidth=5)
            plt.text(read_limit, 0.35, f"{len(sub_gene_join)} transcripts left", verticalalignment='center', rotation=-15, fontsize=fontsize-4)
        
    plt.plot(limits, corrs, color=palette[0])
        
    
    # Regression plot
    
    plt.xlabel('Minimum read requirement',fontsize=fontsize+6)
    plt.ylabel('Correlation \nof Eisen t5 and predicted t5\n(Replicate X)',fontsize=fontsize+6)
    # plt.legend(loc='upper left', fontsize=16, frameon=False)
    # plt.text(1, 5, f'r={np.corrcoef(x,y)[0,1]:.2f}', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    
    sns.set_style('whitegrid')
    sns.despine()
    
    plt.savefig(args.output, bbox_inches = 'tight')
    
    # Boxplots
#     plt.figure()
#     sns.boxplot(data=gene_join, x='t5_binned', y='pred_t5')
#     plt.xlabel('t5 Eisen',fontsize=16)
#     plt.ylabel('t5 predicted',fontsize=16)
#     plt.legend(loc='upper left', fontsize=16, frameon=False)
    
#     # sns.set_style('whitegrid')
#     plt.grid(axis='y', linestyle='-', which='major', color='white', linewidth=0.5) # Remove or set white color for horizontal grid lines
    
#     sns.despine()
    
#     path = Path(args.output)
#     new_path = path.with_stem(path.stem + "_boxplot")
#     plt.savefig(new_path, bbox_inches = 'tight')
    
    

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
    

def clean_halflifes_df(df, group_col):
    #grouping by gene and taking only genes that have non-conflicting t5 values
    #Multiple transcripts per gene, all have the same halflife measured, dropping duplicates
    clean_df = df.groupby(group_col).first().reset_index()
    # bool_map_of_nonuniques = (grouped_df['t5'].nunique()!=1)
    # nonunique_genes = bool_map_of_nonuniques[bool_map_of_nonuniques].index
    
    #dropping genes with various t5 values - TODO just take first?
    # unique_df = df[~df[group_col].isin(nonunique_genes)]
    # grouped_unique_df = unique_df.groupby(group_col)
    # assert (grouped_unique_df['t5'].nunique()==1).all()
    # clean_df = grouped_unique_df.first().reset_index()
    
    clean_df = clean_df[clean_df['t5']!='--']
    clean_df['t5'] = pd.to_numeric(clean_df['t5'])
    return clean_df

# import pandas as pd
# exps = ['hsa_dRNA_HeLa_DRB_0h_1','mmu_dRNA_3T3_mion_1', 'mmu_dRNA_3T3_PION_1']
# for exp in exps:
#     df_path = f'../../Hackathon202305/prep/experiments/{exp}/features_v1.csv'
#     df = pd.read_csv(df_path)
#     df = df.rename(columns={'gene':'Gene.stable.ID'})
#     df = df.groupby('Gene.stable.ID').first() #Multiple transcripts per gene, all have the same halflife measured, dropping duplicates
#     #check if not aggregated per gene 
#     print(df.groupby('Gene.stable.ID').size().describe())
#     print(len(df))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prediction on FAST5 files and save results in a pickle file.')
    parser.add_argument('--gene-predictions', type=str, required=True)
    parser.add_argument('--gene-halflifes', type=str, required=True)
    parser.add_argument('--gene-halflifes-gene-column', type=str, required=True)
    parser.add_argument('--tl', type=float, required=True, help='Time parameter for the decay equation')
    
    parser.add_argument('--output', type=str, help='Path to the output plot.')
    
    args = parser.parse_args()
    main(args)
    
    
    