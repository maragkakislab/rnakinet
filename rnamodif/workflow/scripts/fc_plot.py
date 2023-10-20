import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from plot_helpers import setup_palette, correlation_plot

def calc_fc(df, col, conditions_count, controls_count):
    df['ctrl_avg'] = df[[f'{col}_ctrl_{i}' for i in range(controls_count)]].mean(axis=1)
    df['cond_avg'] = df[[f'{col}_cond_{i}' for i in range(conditions_count)]].mean(axis=1)
    #TODO not Fold Change - rename to someting better
    df['Pred_FC'] = df['cond_avg']/df['ctrl_avg']
    df['Relative modification increase (%)'] = ((df['cond_avg']/df['ctrl_avg'])-1)*100
    return df

def main(args):
    ctrl_paths = args.gene_level_preds_control
    cond_paths = args.gene_level_preds_condition
    deseq_path = args.deseq_output
    pred_col = args.pred_col
    target_col = args.target_col
    min_reads = args.min_reads
    
    deseq_df = pd.read_csv(deseq_path, sep='\t', index_col=None).reset_index()
    deseq_df = deseq_df[~deseq_df['log2FoldChange'].isna()] #Dropping genes that dont contain fold change data
    
    cond_dfs = [pd.read_csv(path, sep='\t', index_col=0) for path in cond_paths]
    ctrl_dfs = [pd.read_csv(path, sep='\t', index_col=0) for path in ctrl_paths]
    
    cond_dfs = [df[df['reads']>=min_reads] for df in cond_dfs]
    ctrl_dfs = [df[df['reads']>=min_reads] for df in ctrl_dfs]
    
    for i,cond_df in enumerate(cond_dfs):
        cond_df.columns = [f'{col}_cond_{i}' if col != 'Gene stable ID' else col for col in cond_df.columns]
    
    for i,ctrl_df in enumerate(ctrl_dfs):
        ctrl_df.columns = [f'{col}_ctrl_{i}' if col != 'Gene stable ID' else col for col in ctrl_df.columns]
    
    all_dfs = cond_dfs+ctrl_dfs
    joined_df = all_dfs[0]
    for df in all_dfs[1:]:
        joined_df = joined_df.merge(df, on='Gene stable ID', how='outer')

    joined_df = joined_df.merge(deseq_df, left_on='Gene stable ID', right_on='index', how='right')
    joined_df = calc_fc(joined_df, pred_col, conditions_count = len(cond_dfs), controls_count = len(ctrl_dfs))
    
    # dropping all genes that dont appear in all experiments
    joined_df = joined_df[~joined_df.isna().any(axis=1)]
    
    #Filtering where Pred_FC is infinite or nan (after log division when some of the ratios are infinite)
    joined_df = joined_df.replace([np.inf, -np.inf], np.nan)
    joined_df = joined_df[~joined_df['Pred_FC'].isna()]
    
    joined_df['Pred_log2FoldChange'] = np.log2(joined_df['Pred_FC'])
    joined_df = joined_df[~joined_df['Pred_log2FoldChange'].isna()]
    joined_df['FC'] = 2**joined_df['log2FoldChange']
    
    correlation_plot(joined_df, x_column='log2FoldChange',y_column='Relative modification increase (%)', x_label='Expression fold change (log2)\nHeat shock vs control', y_label='Relative modification increase (%)', output=args.output)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make FC correlation plot from differential expression analysis data and predictions.")
    parser.add_argument("--gene-level-preds-control",nargs='+', type=str)
    parser.add_argument("--gene-level-preds-condition",nargs='+', type=str)
    parser.add_argument("--deseq-output", type=str)
    parser.add_argument("--pred-col", type=str)
    parser.add_argument("--target-col", type=str)
    parser.add_argument("--min-reads", type=int)
    
    parser.add_argument("--output", help="filename to save the plot")
    
    args = parser.parse_args()
    main(args)