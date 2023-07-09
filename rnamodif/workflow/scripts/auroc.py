import seaborn as sns
import matplotlib.pyplot as plt
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

    # create a new figure with a specific size (in inches)
    # fig, ax = plt.subplots(figsize=(10, 8))

    # plot ROC curve with a solid line and a professional color
    # plt.plot(fpr, tpr, label=f'{pos_name}\n{neg_name}\nAUROC {auroc:.2f}, thr {cutoff_1:.2f} tpr {cutoff_1_tpr:.2f}', linestyle='-')
    plt.plot(fpr, tpr, label=f'{pos_name}\n{neg_name}\nAUROC {auroc:.2f}', linestyle='-')
    
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    # plt.title('Receiver Operating Characteristic Curve', fontsize=16)

    # increase ticks size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # add a minor grid
    # plt.grid(which='minor', alpha=0.2)
    # plt.grid(which='major', alpha=0.5)

    # use seaborn to add a minor grid and remove the top and right walls
    sns.set_style('whitegrid')
    sns.despine()

    plt.legend(loc='lower right', fontsize=12, frameon=False)

def main(args):
    plot_and_save(args, plot_auroc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prediction on FAST5 files and save results in a pickle file.')
    parser = parse_plotting_args(parser)

    args = parser.parse_args()
    main(args)



# from matplotlib import pyplot as plt
# import numpy as np
# import argparse
# from plot_helpers import plot_and_save, parse_plotting_args
# from sklearn import metrics


# def plot_auroc(pos_preds, neg_preds, pos_name, neg_name):
#     predictions = np.concatenate((pos_preds,neg_preds))
#     labels = np.concatenate((np.repeat(1, len(pos_preds)),np.repeat(0, len(neg_preds))))
#     fpr, tpr, thresholds = metrics.roc_curve(labels, predictions)

#     cutoff_1 = thresholds[np.argmax(tpr-fpr)]
#     cutoff_1_tpr = tpr[np.argmax(tpr-fpr)]

#     auroc = metrics.auc(fpr, tpr)
    
#     plt.plot(fpr, tpr, label=f'{pos_name}\n{neg_name}\nAUROC {auroc:.2f}, thr {cutoff_1:.2f} tpr {cutoff_1_tpr:.2f}')
#     plt.ylabel('True Positive Rate')
#     plt.xlabel('False Positive Rate')
#     plt.legend(loc='center left', bbox_to_anchor=(1,0.5), prop={'size':10})

# def main(args):
#     plot_and_save(args, plot_auroc)
    
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Run prediction on FAST5 files and save results in a pickle file.')
#     parser = parse_plotting_args(parser)
    
#     args = parser.parse_args()
#     main(args)
    
    
