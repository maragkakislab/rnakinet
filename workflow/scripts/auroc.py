import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import argparse
from plot_helpers import plot_and_save, parse_plotting_args
from sklearn import metrics


def plot_auroc(pos_preds, neg_preds, pos_name, neg_name):
    predictions = np.concatenate((pos_preds,neg_preds))
    labels = np.concatenate((np.repeat(1, len(pos_preds)),np.repeat(0, len(neg_preds))))
    
    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions)

    cutoff_1 = thresholds[np.argmax(tpr-fpr)]
    cutoff_1_tpr = tpr[np.argmax(tpr-fpr)]

    auroc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{pos_name}\n{neg_name}\nAUROC {auroc:.2f}', linestyle='-')
    
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    
    fontsize=8
    plt.ylabel('True Positive Rate', fontsize=fontsize)
    plt.xlabel('False Positive Rate', fontsize=fontsize)

    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    
    sns.set_style('whitegrid')
    sns.despine()

    plt.legend(loc='lower right', fontsize=fontsize-2, frameon=False)

def main(args):
    mpl.rc('font',family='Arial')
    plot_and_save(args, plot_auroc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot the AUROC plot')
    parser = parse_plotting_args(parser)

    args = parser.parse_args()
    main(args)
  
