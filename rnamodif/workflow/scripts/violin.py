import pickle
import argparse
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches
from plot_helpers import setup_palette
import matplotlib as mpl

def main(args):
    df = pd.DataFrame({exp_name: pd.Series(get_preds_from_file(preds_file)) for exp_name,preds_file in zip(args.exp_names, args.files)})
    x_axis_name = 'Read modification predicted probability'
    y_axis_name = 'Experiment'
    df = df.melt(var_name=y_axis_name, value_name=x_axis_name)
    df = df.dropna()
    palette_colors = setup_palette()
    palette = {exp:get_exp_color(exp, palette_colors) for exp in args.exp_names}
    
    mpl.rc('font',family='Arial')
    fontsize=8
    plt.figure(figsize=(1.5, 0.5*len(args.files)+0.25))
    sns.violinplot(
        x=x_axis_name, 
        y=y_axis_name, 
        data=df, 
        orient='h', 
        inner='box', 
        palette=palette,
        linewidth=0.75,
        scale='area',
    )
    plt.xlabel(x_axis_name, fontsize=fontsize)
    plt.ylabel(y_axis_name, fontsize=fontsize)
    
    plt.yticks(fontsize=fontsize-2)
    plt.xticks(fontsize=fontsize-2)
    
    red_patch = mpatches.Patch(color=palette_colors[0], label='Control')
    green_patch = mpatches.Patch(color=palette_colors[1], label='5EU')
    sns.set_style('whitegrid')
    sns.despine()
    plt.legend(handles=[green_patch, red_patch], loc='upper center', fontsize=fontsize-2, frameon=False, ncol=2, bbox_to_anchor=(0.5,1.2))
    plt.savefig(args.output, bbox_inches = 'tight')

def get_exp_color(exp_name, palette_colors):
    if 'unlabeled' in exp_name or 'dmso' in exp_name or 'DMSO' in exp_name:
        return palette_colors[0]
    return palette_colors[1]
    

def get_preds_from_file(file):
    with open(file,'rb') as f:
        preds = pickle.load(f)
    return preds.values()
      
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prediction on FAST5 files and save results in a pickle file.')
    parser.add_argument('--files', type=str, required=True, nargs='+', help='Paths to the files containing predictions.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output plot.')
    parser.add_argument('--model-name', type=str, required=True, help='Name of the model to use for predictions')
    parser.add_argument('--exp-names', type=str, nargs='+', required=True, help='Names of the experiments')
    
    args = parser.parse_args()
    main(args)