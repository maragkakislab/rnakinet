import pickle
import argparse
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path

def main(args):
    x_size = len(args.files)
    y_size = 1#len(models)
    #2.5*x_size
    fig, ax = plt.subplots(x_size, y_size, figsize=(5*y_size,1*x_size), sharex=True, squeeze=False)
    model_idx = 0
    model_name = args.model_name
    for i,(exp_name,file) in enumerate(zip(args.exp_names,args.files)):
        with open(file,'rb') as f:
            preds = pickle.load(f)

        data_df = pd.DataFrame({"readid":preds.keys(),"score":preds.values()})
        v = ax[i][model_idx].violinplot(data_df['score'], showmeans=True, vert=False)
        for v in v['bodies']:
            v.set_facecolor('red')
            v.set_edgecolor('red')
        # ax[model_idx][preds_idx].set_ylabel(k.stem[27:], rotation=30, labelpad=40)
        ax[i][model_idx].set_yticks([])
        if(i == 0):
            ax[i][model_idx].set_xlabel(model_name)
            ax[i][model_idx].xaxis.set_label_position('top')
        if(model_idx == 0):
            ax[i][model_idx].set_ylabel(exp_name, rotation=0, labelpad=110)
    plt.savefig(args.output, bbox_inches = 'tight')

    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prediction on FAST5 files and save results in a pickle file.')
    parser.add_argument('--files', type=str, required=True, nargs='+', help='Paths to the files containing predictions.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output plot.')
    parser.add_argument('--model-name', type=str, required=True, help='Name of the model to plot')
    parser.add_argument('--exp-names', type=str, nargs='+', required=True, help='Names of the experiments')
    
    args = parser.parse_args()
    main(args)