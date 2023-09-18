from matplotlib import pyplot as plt
import numpy as np
import argparse
from plot_helpers import plot_and_save, parse_plotting_args
from sklearn import metrics
import seaborn as sns

def plot_pr_curve(pos_preds, neg_preds, pos_name, neg_name):
    predictions = np.concatenate((pos_preds,neg_preds))
    labels = np.concatenate((np.repeat(1, len(pos_preds)),np.repeat(0, len(neg_preds))))
    precision, recall, thresholds = metrics.precision_recall_curve(labels, predictions)

    auroc = metrics.auc(recall, precision)
    
    
    # plt.plot(recall, precision, marker='.', label=f'{pos_name}\n{neg_name}\nAUPRC {auroc:.2f}')
    plt.plot(recall, precision, label=f'{pos_name}\n{neg_name}\nAUPRC {auroc:.2f}')
    
    fontsize=20
    plt.xlabel('Recall', fontsize=fontsize+6)
    plt.ylabel('Precision', fontsize=fontsize+6)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    sns.set_style('whitegrid')
    sns.despine()
    plt.legend(loc='lower left', fontsize=fontsize-4, frameon=False)
    plt.xlim([-0.05, 1])
    plt.ylim([-0.05, 1])
    
        
def main(args):
    # fig, ax = plt.subplots(figsize=(5,5))
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    
    plot_and_save(args, plot_pr_curve)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prediction on FAST5 files and save results in a pickle file.')
    parser = parse_plotting_args(parser)
    
    args = parser.parse_args()
    main(args)