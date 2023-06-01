from matplotlib import pyplot as plt
import numpy as np
import argparse
from plot_helpers import plot_and_save, parse_plotting_args


def plot_thresholds(pos_preds, neg_preds, pos_name, neg_name):
    thresholds = np.arange(0,1,0.01)
    #Balanced accuracy
    accs = [np.mean([np.mean(neg_preds<t),np.mean(pos_preds>t)]) for t in thresholds]
    plt.plot(thresholds, accs, label=f'{pos_name}\n{neg_name}')
    plt.legend(loc='center left', bbox_to_anchor=(1,0.5), prop={'size':10})
    

def main(args):
    plot_and_save(args, plot_thresholds)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prediction on FAST5 files and save results in a pickle file.')
    parser = parse_plotting_args(parser)
    args = parser.parse_args()
    main(args)