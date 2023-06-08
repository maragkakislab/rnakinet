import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def calc_fc(df, col):
    # #TODO avg is not good for FC calculation?
    #TODO add counts to predicted reads and use Deseq2 to compute FC?
    # df['ctrl_avg'] = df[[f'{col}_ctrl_{i}' for i in range(1,4)]].mean(axis=1)
    # df['cond_avg'] = df[[f'{col}_cond_{i}' for i in range(1,4)]].mean(axis=1)
    # df['Pred_FC'] = np.log2(df['cond_avg']/df['ctrl_avg'])
    # # df['Pred_FC'] = df['NoArs_avg']-df['Ars_avg']
    
    df['ctrl_avg'] = df[[f'{col}_ctrl_{i}' for i in range(1,4)]].mean(axis=1)
    df['cond_avg'] = df[[f'{col}_cond_{i}' for i in range(1,4)]].mean(axis=1)
    for i in range(1,4):
        df[f'FC_{i}']=np.log2(df[f'{col}_cond_{i}']/df[f'{col}_ctrl_{i}'])
    
    df['Pred_FC'] = df[[f'FC_{i}' for i in range(1,4)]].mean(axis=1)
    return df

def main(args):
    ctrl_paths = args.gene_level_preds_control
    cond_paths = args.gene_level_preds_condition
    deseq_path = args.deseq_output
    pred_col = args.pred_col
    target_col = args.target_col
    
    deseq_df = pd.read_csv(deseq_path, sep='\t', index_col=None).reset_index()
    cond_dfs = [pd.read_csv(path, sep='\t', index_col=0) for path in cond_paths]
    ctrl_dfs = [pd.read_csv(path, sep='\t', index_col=0) for path in ctrl_paths]
    
    for i,cond_df in enumerate(cond_dfs,1):
        cond_df.columns = [f'{col}_cond_{i}' if col != 'Gene stable ID' else col for col in cond_df.columns]
    
    for i,ctrl_df in enumerate(ctrl_dfs,1):
        ctrl_df.columns = [f'{col}_ctrl_{i}' if col != 'Gene stable ID' else col for col in ctrl_df.columns]
    
    all_dfs = cond_dfs+ctrl_dfs
    joined_df = all_dfs[0]
    for df in all_dfs[1:]:
        joined_df = joined_df.merge(df, on='Gene stable ID', how='outer')

    joined_df = joined_df.merge(deseq_df, left_on='Gene stable ID', right_on='index', how='right')
    joined_df = calc_fc(joined_df, pred_col)
    
    #OPTIONAL dropping all genes that dont appear in all experiments
    joined_df = joined_df[~joined_df.isna().any(axis=1)]
    
    #OPTIONAL filtering genes that have low amount of reads
    #TODO

    #Filtering where Pred_FC is infinite or nan (after log division when some of the ratios are infinite)
    joined_df = joined_df.replace([np.inf, -np.inf], np.nan)
    joined_df = joined_df[~joined_df['Pred_FC'].isna()]
    
    #OPTIONAL dropping genes that have low padj values
    # joined_df = joined_df[joined_df['padj'] < 0.05]
    
    x = joined_df[target_col].values
    y = joined_df['Pred_FC'].values
    plt.scatter(x,y, s=1, alpha=0.1)
    plt.title(np.corrcoef(x,y)[0,1])
    plt.savefig(args.output)
        
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make FC correlation plot from differential expression analysis data and predictions.")
    parser.add_argument("--gene-level-preds-control",nargs='+', type=str)
    parser.add_argument("--gene-level-preds-condition",nargs='+', type=str)
    parser.add_argument("--deseq-output", type=str)
    parser.add_argument("--pred-col", type=str)
    parser.add_argument("--target-col", type=str)
    
    parser.add_argument("--output", help="filename to save the plot")
    
    args = parser.parse_args()
    main(args)