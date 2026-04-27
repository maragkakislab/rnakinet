from matplotlib import pyplot as plt
import numpy as np
import argparse
from plot_helpers import plot_and_save, parse_plotting_args
from sklearn import metrics
import seaborn as sns
import pandas as pd
import math

def plot_pr_curve(pos_preds, neg_preds, pos_name, neg_name):
    predictions = np.concatenate((pos_preds,neg_preds))
    labels = np.concatenate((np.repeat(1, len(pos_preds)),np.repeat(0, len(neg_preds))))
    
    #Upsampling minority class to be balanced for PR curve
    pos_df = pd.DataFrame({'predictions':pos_preds, 'labels':np.repeat(1, len(pos_preds))})
    neg_df = pd.DataFrame({'predictions':neg_preds, 'labels':np.repeat(0, len(neg_preds))}) 
    balance_ratios = np.linspace(0.01, 1.0, 100)
    auc_pr_values = []
    
    total_preds = len(pos_df)+len(neg_df)
    for i,balance_ratio in enumerate(balance_ratios):
        current_pos_ratio = len(pos_df)/total_preds
        current_neg_ratio = len(neg_df)/total_preds

        wanted_pos_ratio = balance_ratio
        wanted_neg_ratio = 1-balance_ratio
        
        #downsample only
        if(current_pos_ratio >= wanted_pos_ratio):
            pos_samples_needed = (wanted_pos_ratio * len(neg_df))/(1-wanted_pos_ratio)
            balanced_pos_df = pos_df[:int(pos_samples_needed)]
            balanced_neg_df = neg_df
        if(current_neg_ratio >= wanted_neg_ratio):
            neg_samples_needed = (wanted_neg_ratio * len(pos_df))/(1-wanted_neg_ratio)
            balanced_pos_df = pos_df
            balanced_neg_df = neg_df[:int(neg_samples_needed)]
        
        balanced_df = pd.concat([balanced_pos_df, balanced_neg_df])
        total_balanced_preds = len(balanced_df)
        current_pos_ratio = len(balanced_pos_df)/total_balanced_preds
        current_neg_ratio = len(balanced_neg_df)/total_balanced_preds
        assert(math.isclose(current_pos_ratio, wanted_pos_ratio, rel_tol=1e-2)),f'{current_pos_ratio}, {wanted_pos_ratio}'
        assert(math.isclose(current_neg_ratio, wanted_neg_ratio, rel_tol=1e-2)),f'{current_neg_ratio}, {wanted_neg_ratio}'
        predictions = balanced_df['predictions'].values
        labels = balanced_df['labels'].values

        precision, recall, thresholds = metrics.precision_recall_curve(labels, predictions)
        aucpr = metrics.auc(recall, precision)
        auc_pr_values.append(aucpr)
        
        
    plt.plot(balance_ratios,auc_pr_values,  label=f'{pos_name}\n{neg_name}\nAUPRC')
    
    fontsize=8
    plt.xlabel('Ratio', fontsize=fontsize)
    plt.ylabel('AUC-PR', fontsize=fontsize)
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    sns.set_style('whitegrid')
    sns.despine()
    plt.legend(loc='lower left', fontsize=fontsize-2, frameon=False)
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    
        
def main(args):
    plot_and_save(args, plot_pr_curve)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot a Precision-Recall curve')
    parser = parse_plotting_args(parser)
    
    args = parser.parse_args()
    main(args)