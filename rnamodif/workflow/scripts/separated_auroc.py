import pysam
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import argparse
import seaborn as sns
from plot_helpers import setup_palette
from sklearn import metrics
import matplotlib as mpl


def read_predictions(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)

def get_readid_to_info_dict(bam_file, prediction_dict):
    read_info_dict = {}
    samfile = pysam.AlignmentFile(bam_file, 'rb')
    for read in samfile.fetch():
        if read.query_name in prediction_dict:
            read_sequence = read.seq
            read_length = len(read_sequence)
            
            percent_U = (read_sequence.count('T') / read_length)*100
            
            read_info_dict[read.query_name] = {'read_length':read_length, 'percent_U':percent_U}
            
    samfile.close()
    return read_info_dict
    
def main(args):
    pos_bams = args.positives_bams
    neg_bams = args.negatives_bams
    pos_preds = args.positives_predictions
    neg_preds = args.negatives_predictions
    
    for pos_bam, pos_pred in zip(pos_bams, pos_preds):
        pos_predictions = read_predictions(pos_pred)
        pos_readid_to_info = get_readid_to_info_dict(pos_bam, pos_predictions)
        
    for neg_bam, neg_pred in zip(neg_bams, neg_preds):
        neg_predictions = read_predictions(neg_pred)
        neg_readid_to_info = get_readid_to_info_dict(neg_bam, neg_predictions)
        
    if(args.plot_type == 'Uperc'):
        attribute_auroc_plot(pos_readid_to_info, neg_readid_to_info, pos_predictions, neg_predictions, args, key='percent_U', thresholds=[0,20,30,100])
    #TODO plot auroc
    if(args.plot_type == 'length'):
        attribute_auroc_plot(pos_readid_to_info, neg_readid_to_info, pos_predictions, neg_predictions, args, key='read_length', thresholds=[0,1000,3000,5000,1000000])
        
def attribute_auroc_plot(pos_readid_to_info, neg_readid_to_info, pos_preds, neg_preds, args, key, thresholds):
    output_file = args.output
    mpl.rc('font',family='Arial')
    fontsize=8
    plt.figure(figsize=(1.5,1.5))
    palette = setup_palette()
    
    for i in range(len(thresholds)-1):
        pos_preds_threshold = []
        neg_preds_threshold = []
        for read_id, info in pos_readid_to_info.items():
            if(info[key] >= thresholds[i] and info[key] < thresholds[i+1]):
                pos_preds_threshold.append(pos_preds[read_id])
        for read_id, info in neg_readid_to_info.items():
            if(info[key] >= thresholds[i] and info[key] < thresholds[i+1]):
                neg_preds_threshold.append(neg_preds[read_id])
        
        pos_labels = np.repeat(1, len(pos_preds_threshold)).tolist()
        neg_labels = np.repeat(0, len(neg_preds_threshold)).tolist()
                
        preds = pos_preds_threshold+neg_preds_threshold
        labels = pos_labels+neg_labels
        
        print(thresholds[i], thresholds[i+1], len(labels))
        if(len(labels)<=0):
            continue
    
        fpr, tpr, _ = metrics.roc_curve(labels, preds)
        auroc = metrics.auc(fpr, tpr)   
        plt.plot(fpr,tpr, linewidth=1, color=palette[i], label=f'{thresholds[i]} <= {key}  < {thresholds[i+1]} (AUROC {auroc:.2f})')
                                               
                                               
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=fontsize)
    plt.ylabel('True Positive Rate', fontsize=fontsize)
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    sns.set_style('whitegrid')
    sns.despine()
    plt.legend(loc='lower right', fontsize=fontsize-2, frameon=False)
    plt.savefig(output_file, bbox_inches='tight')
                                               
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prediction on FAST5 files and save results in a pickle file.')
    parser.add_argument('--positives_bams', nargs='+', type=str, required=True, help='Bam files for positive reads')
    parser.add_argument('--negatives_bams', nargs='+', type=str, required=True, help='Bam files for negative reads')
    parser.add_argument('--positives_predictions', nargs='+', type=str, required=True, help='Prediction files for positive reads')
    parser.add_argument('--negatives_predictions', nargs='+', type=str, required=True, help='Prediction files for negative reads')
    parser.add_argument('--plot_type', type=str, required=True, help='Type of the plot (auroc or f1)')
    parser.add_argument('--chosen_threshold', type=float, required=True, help='Chosen threshold for classification')
    
    parser.add_argument('--output', type=str, required=True, help='Path to the output plot file.')
    args = parser.parse_args()
    main(args)