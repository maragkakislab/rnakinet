import numpy as np
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

palette = ['#B22222','#4682B4', '#FFC107','#004D40', '#6F8480']
def setup_palette():
    plt.gca().set_prop_cycle('color', palette)
    return palette

def check_for_nans(predictions):
    if(np.isnan(predictions).any()):
        print('Warning: Nan found in predictions, setting to 0')
        print(np.where(np.isnan(predictions)))
        print(len(np.where(np.isnan(predictions))))
        for nan_idx in np.where(np.isnan(predictions))[0]:
            predictions[nan_idx] = np.float16(0.0)
            

def plot_and_save(args, plot_fn, callbacks=[]):
    pos_preds = args.positives_in_order
    neg_preds = args.negatives_in_order
    pos_names = args.positives_names_in_order
    neg_names = args.negatives_names_in_order
    pos_groups = args.positives_groups_in_order
    neg_groups = args.negatives_groups_in_order
    output_file = args.output
    
    plt.figure(figsize=(1.5,1.5))
    setup_palette()
    
    #TODO remove pos_names, neg_names
    positives_total = []
    negatives_total = []
    group_to_preds = {}
    for group in pos_groups+neg_groups:
        group_to_preds[group]={'positives':[],'negatives':[]}
    
    for pos_file, pos_name, pos_group in zip(pos_preds, pos_names, pos_groups):
        with open(pos_file,'rb') as f:
            positives = list(pickle.load(f).values())
        check_for_nans(positives)
        group_to_preds[pos_group]['positives']+=positives

    for neg_file, neg_name, neg_group in zip(neg_preds, neg_names, neg_groups):
        with open(neg_file,'rb') as f:
            negatives = list(pickle.load(f).values())
        check_for_nans(negatives)
        group_to_preds[neg_group]['negatives']+=negatives
        
    for group, subdict in group_to_preds.items():
        plot_fn(subdict['positives'], subdict['negatives'],  f'{group}_positives', f'{group}_negatives')
    
    for callback in callbacks:
        callback()
    
    plt.savefig(output_file, bbox_inches='tight')

    
#TODO refactor away, has its own script now
def correlation_plot(df, x_column, y_column, x_label, y_label, output):
    plt.figure(figsize=(1.5,1.5))
    palette = setup_palette()
    sns.regplot(data=df, x=x_column, y=y_column, 
            scatter_kws={'alpha':0.6, 's':7, 'color':palette[0]}, 
            line_kws={"color": palette[1], "lw": 2},  
    )
    
    x = df[x_column].values
    y = df[y_column].values
    
    fontsize=8
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    
    spearman = spearmanr(x,y).statistic
    pearson = pearsonr(x,y).statistic
    
    plt.text(0.1, 0.95, f'r={pearson:.2f} ρ={spearman:.2f}', fontsize=fontsize-2, transform=plt.gca().transAxes, verticalalignment='top')
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    
    sns.set_style('whitegrid')
    sns.despine()
    
    plt.savefig(output, bbox_inches='tight')     
    
    
    
def parse_plotting_args(parser):
    parser.add_argument(
        '--positives-in-order', 
        type=str, 
        required=True, 
        nargs='+', 
        help='Paths to the files containing predictions on positives.'
    )
    parser.add_argument(
        '--negatives-in-order', 
        type=str, 
        required=True, 
        nargs='+', 
        help='Paths to the files containing predictions on negatives.'
    )
    parser.add_argument(
        '--positives-names-in-order', 
        type=str, 
        required=True, 
        nargs='+', 
        help='Names of the experiments containing predictions on positives.'
    )
    parser.add_argument(
        '--negatives-names-in-order', 
        type=str, 
        required=True, 
        nargs='+', 
        help='Names of the experiments containing predictions on negatives.'
    )
    parser.add_argument(
        '--positives-groups-in-order', 
        type=str, 
        required=True, 
        nargs='+', 
        help='Names of the experiment group of positives.'
    )
    parser.add_argument(
        '--negatives-groups-in-order', 
        type=str, 
        required=True, 
        nargs='+', 
        help='Names of the experiment group of negatives.'
    )
    parser.add_argument('--model-name', type=str, required=True, help='Name of the model to plot')
    parser.add_argument('--chosen_threshold', type=float, required=False, help='Chosen threshold of the model')
    parser.add_argument('--output', type=str, required=True, help='Path to the output plot file.')
    return parser