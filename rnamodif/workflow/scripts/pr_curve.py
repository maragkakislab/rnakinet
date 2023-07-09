from matplotlib import pyplot as plt
import numpy as np
import argparse
from plot_helpers import plot_and_save, parse_plotting_args
from sklearn import metrics

def plot_pr_curve(pos_preds, neg_preds, pos_name, neg_name):
    predictions = np.concatenate((pos_preds,neg_preds))
    labels = np.concatenate((np.repeat(1, len(pos_preds)),np.repeat(0, len(neg_preds))))
    precision, recall, thresholds = metrics.precision_recall_curve(labels, predictions)

    plt.plot(recall, precision, marker='.', label=f'{pos_name}\n{neg_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='center left', bbox_to_anchor=(1,0.5), prop={'size':10})
        
def main(args):
    plot_and_save(args, plot_pr_curve)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prediction on FAST5 files and save results in a pickle file.')
    parser = parse_plotting_args(parser)
    
    args = parser.parse_args()
    main(args)