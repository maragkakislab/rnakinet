import pickle
import argparse
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn import metrics


def check_for_nans(predictions):
    if(np.isnan(predictions).any()):
        print('Warning: Nan found in predictions, setting to 0')
        print(np.where(np.isnan(predictions)))
        for nan_idx in np.where(np.isnan(predictions))[0]:
            predictions[nan_idx] = 0.0

def main(args):
    pos_preds = args.positives_in_order
    neg_preds = args.negatives_in_order
    pos_names = args.positives_names_in_order
    neg_names = args.negatives_names_in_order
    model_name = args.model_name
    output_file = args.output
    
    for pos_file, neg_file, pos_name, neg_name in zip(pos_preds, neg_preds, pos_names, neg_names):
        with open(pos_file,'rb') as f:
            positives = list(pickle.load(f).values())
        check_for_nans(positives)
        
        with open(neg_file,'rb') as f:
            negatives = list(pickle.load(f).values())
        check_for_nans(negatives)
        
        predictions = np.concatenate((positives,negatives))
        labels = np.concatenate((np.repeat(1, len(positives)),np.repeat(0, len(negatives))))
        fpr, tpr, thresholds = metrics.roc_curve(labels, predictions)
        
        cutoff_1 = thresholds[np.argmax(tpr-fpr)]
        cutoff_1_tpr = tpr[np.argmax(tpr-fpr)]
        
        plt.plot(fpr, tpr, label=f'{pos_name}\n{neg_name}')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc='center left', bbox_to_anchor=(1,0.5), prop={'size':10})
        
        
    plt.savefig(output_file, bbox_inches='tight')
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prediction on FAST5 files and save results in a pickle file.')
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
    
    args = parser.parse_args()
    main(args)