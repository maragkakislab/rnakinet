import pickle
import argparse
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path

def main(args):
    y_size = len(args.files)
    x_size = 1  # len(models)
    fig, ax = plt.subplots(x_size, y_size, figsize=(5*y_size,10*x_size), sharex=True, squeeze=False)
    model_idx = 0
    model_name = args.model_name
    for i,(exp_name,file) in enumerate(zip(args.exp_names,args.files)):
        with open(file,'rb') as f:
            preds = pickle.load(f)

        data_df = pd.DataFrame({"readid":preds.keys(),"score":preds.values()})
        boxplot = ax[model_idx][i].boxplot(data_df['score'], vert=True, widths=0.6)
        # for patch in boxplot['boxes']:
            # patch.set_facecolor('blue')

        ax[model_idx][i].set_yticks([])
        if(i == 0):
            ax[model_idx][i].set_ylabel(model_name)
            # ax[model_idx][i].yaxis.set_label_position('top')
        if(model_idx == 0):
            ax[model_idx][i].set_xlabel(exp_name, rotation=45, labelpad=0)
    plt.savefig(args.output, bbox_inches = 'tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prediction on FAST5 files and save results in a pickle file.')
    parser.add_argument('--files', type=str, required=True, nargs='+', help='Paths to the files containing predictions.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output plot.')
    parser.add_argument('--model-name', type=str, required=True, help='Name of the model to plot')
    parser.add_argument('--exp-names', type=str, nargs='+', required=True, help='Names of the experiments')
    
    args = parser.parse_args()
    main(args)
