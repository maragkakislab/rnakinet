from matplotlib import pyplot as plt
import numpy as np
import argparse
from plot_helpers import plot_and_save, parse_plotting_args
import seaborn as sns
from plot_helpers import setup_palette
import matplotlib as mpl

def plot_thresholds(pos_preds, neg_preds, pos_name, neg_name):
    thresholds = np.arange(0,1,0.01)
    #Balanced accuracy
    accs = [np.mean([np.mean(neg_preds<t),np.mean(pos_preds>t)]) for t in thresholds]
    
    mpl.rc('font',family='Arial')
    fontsize=8
    plt.plot(thresholds, accs, label=f'{pos_name}\n{neg_name}', linewidth=2)
    plt.xlabel('Threshold', fontsize=fontsize)
    plt.ylabel('Balanced Accuracy', fontsize=fontsize)
    
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    # plt.legend(loc='center left', bbox_to_anchor=(1,0.5), prop={'size':10})
    sns.set_style('whitegrid')
    sns.despine()
    plt.legend(loc='lower right', frameon=False, fontsize=fontsize-2)
    
def get_threshold_callback(args):
    def threshold_line_callback():
        palette = setup_palette()
        color = palette[4]
        plt.axvline(x=args.chosen_threshold, color=color, linestyle='--')
    return threshold_line_callback
    

def main(args):
    # plt.rcParams['axes.prop_cycle'] = plt.cycler(color=sns.color_palette('colorblind'))
    # fig, ax = plt.subplots(figsize=(5,5))
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    
    plot_and_save(args, plot_thresholds, [get_threshold_callback(args)])
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prediction on FAST5 files and save results in a pickle file.')
    parser = parse_plotting_args(parser)
    args = parser.parse_args()
    main(args)