import numpy as np
import pickle
from matplotlib import pyplot as plt

def check_for_nans(predictions):
    if(np.isnan(predictions).any()):
        print('Warning: Nan found in predictions, setting to 0')
        print(np.where(np.isnan(predictions)))
        for nan_idx in np.where(np.isnan(predictions))[0]:
            predictions[nan_idx] = 0.0
            

def plot_and_save(args, plot_fn):
    pos_preds = args.positives_in_order
    neg_preds = args.negatives_in_order
    pos_names = args.positives_names_in_order
    neg_names = args.negatives_names_in_order
    output_file = args.output
    
    for pos_file, neg_file, pos_name, neg_name in zip(pos_preds, neg_preds, pos_names, neg_names):
        with open(pos_file,'rb') as f:
            positives = list(pickle.load(f).values())
        check_for_nans(positives)
        
        with open(neg_file,'rb') as f:
            negatives = list(pickle.load(f).values())
        check_for_nans(negatives)
        
        plot_fn(positives, negatives, pos_name, neg_name)
        
    plt.savefig(output_file, bbox_inches='tight')
    
    
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
    parser.add_argument('--model-name', type=str, required=True, help='Name of the model to plot')
    parser.add_argument('--output', type=str, required=True, help='Path to the output plot file.')
    return parser