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

def get_readid_to_chr_dict(bam_file, prediction_dict):
    chromosome_dict = {}
    samfile = pysam.AlignmentFile(bam_file, 'rb')
    for read in samfile.fetch():
        if read.query_name in prediction_dict:
            chromosome_dict[read.query_name] = samfile.get_reference_name(read.reference_id)
    samfile.close()
    return chromosome_dict

def get_chr_to_preds_dict(chromosome_dict, prediction_dict, args):
    chr_to_preds = defaultdict(list)
    for read_id, chromosome in chromosome_dict.items():
        if(chromosome not in args.train_chrs+args.valid_chrs+args.test_chrs):
            continue
        chr_to_preds[chromosome].append(prediction_dict[read_id])
    
    return chr_to_preds
    
def main(args):
    pos_bams = args.positives_bams
    neg_bams = args.negatives_bams
    pos_preds = args.positives_predictions
    neg_preds = args.negatives_predictions
    
    all_pos_predictions = {}
    all_neg_predictions = {}
    all_pos_readid_to_chr = {}
    all_neg_readid_to_chr = {}
    
    for pos_bam, pos_pred in zip(pos_bams, pos_preds):
        pos_predictions = read_predictions(pos_pred)
        pos_readid_to_chr = get_readid_to_chr_dict(pos_bam, pos_predictions)
        all_pos_predictions.update(pos_predictions)
        all_pos_readid_to_chr.update(pos_readid_to_chr)
        
    for neg_bam, neg_pred in zip(neg_bams, neg_preds):
        neg_predictions = read_predictions(neg_pred)
        neg_readid_to_chr = get_readid_to_chr_dict(neg_bam, neg_predictions)
        all_neg_predictions.update(neg_predictions)
        all_neg_readid_to_chr.update(neg_readid_to_chr)
        
    pos_chr_to_preds = get_chr_to_preds_dict(all_pos_readid_to_chr, all_pos_predictions, args)
    neg_chr_to_preds = get_chr_to_preds_dict(all_neg_readid_to_chr, all_neg_predictions, args)
             
    if(args.plot_type == 'auroc'):
        auroc_plot(pos_chr_to_preds, neg_chr_to_preds, args)
        
    if(args.plot_type == 'f1'):
        f1_plot(pos_chr_to_preds, neg_chr_to_preds, args)
    
def f1_plot(pos_chr_to_preds, neg_chr_to_preds, args):
    output_file = args.output
    mpl.rc('font',family='Arial')
    fontsize=8
    plt.figure(figsize=(1.5,1.5))
    palette = setup_palette()
    train_color = palette[4]
    valid_color = palette[1]
    test_color = palette[0]
    
    split_to_chrs = {'train':args.train_chrs, 'valid':args.valid_chrs, 'test':args.test_chrs}
    split_to_f1 = {}
    for split, chrs in split_to_chrs.items():
        split_preds = []
        split_labels = []
        for chr_name in chrs:
            pos_preds = pos_chr_to_preds[chr_name]
            neg_preds = neg_chr_to_preds[chr_name]


            #balancing positives and negatives for f1 score
            max_len = max(len(pos_preds), len(neg_preds))
            pos_preds = np.resize(pos_preds, max_len)
            neg_preds = np.resize(neg_preds, max_len)
            
            pos_labels = np.repeat(1, len(pos_preds))
            neg_labels = np.repeat(0, len(neg_preds))
            
            preds = np.concatenate([pos_preds,neg_preds])
            labels = np.concatenate([pos_labels,neg_labels])
            
            split_preds+=preds.tolist()
            split_labels+=labels.tolist()
        # print(args.chosen_threshold)
        split_preds_classes = [1 if pred >= args.chosen_threshold else 0 for pred in split_preds]
        # print(len(split_preds_classes), sum(split_preds_classes))
        f1 = metrics.f1_score(split_labels, split_preds_classes)
        split_to_f1[split]=f1
    
    # print(split_to_f1)
    df = pd.DataFrame(list(split_to_f1.items()), columns=['Chromosomes','F1'])
    b = sns.barplot(x='Chromosomes',y='F1', data=df, palette=palette)
    b.set_ylim(0, args.ylim)
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    # b.set_yticklabels(b.get_yticklabels(), size = fontsize-2)
    # b.set_xticklabels(b.get_xticklabels(), size = fontsize-2)
    plt.xlabel('Chromosomes', fontsize=fontsize)
    plt.ylabel('F1 Score', fontsize=fontsize)
    plt.legend(loc='lower left', frameon=False, fontsize=fontsize-2)
    sns.despine()
    # plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
        
    
    

def auroc_plot(pos_chr_to_preds, neg_chr_to_preds, args):
    output_file = args.output
    mpl.rc('font',family='Arial')
    fontsize=8
    plt.figure(figsize=(1.5,1.5))
    palette = setup_palette()
    train_color = palette[4]
    valid_color = palette[1]
    test_color = palette[0]
    
    for chr_name in args.train_chrs+args.valid_chrs+args.test_chrs:
        pos_preds = pos_chr_to_preds[chr_name]
        neg_preds = neg_chr_to_preds[chr_name]
        
        pos_labels = np.repeat(1, len(pos_preds)).tolist()
        neg_labels = np.repeat(0, len(neg_preds)).tolist()
        
        preds = pos_preds+neg_preds
        labels = pos_labels+neg_labels
        
        if(len(labels)<=0):
            continue
            
        # print(chr_name, len(pos_labels), len(neg_labels))
        
        fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
        auroc = metrics.auc(fpr, tpr)
        
        if(chr_name in args.train_chrs):
            kwargs={'color':train_color, 'alpha':0.7, 'label':'Train chrs', 'zorder':0}
            if(chr_name != args.train_chrs[0]):
                kwargs['label'] = '_nolegend_'
        elif(chr_name in args.valid_chrs):
            kwargs={'color':valid_color, 'alpha':1.0, 'linewidth':2.5, 'label':'Valid chrs', 'zorder':1}
            if(chr_name != args.valid_chrs[0]):
                kwargs['label'] = '_nolegend_'
        elif(chr_name in args.test_chrs):
            kwargs={'color':test_color, 'alpha':1.0, 'linewidth':2, 'label':'Test chrs', 'zorder':2}
            if(chr_name != args.test_chrs[0]):
                kwargs['label'] = '_nolegend_'
        else:
            raise Exception('Chr not recognized')
            
        plt.plot(fpr, tpr, **kwargs)  
             
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
    parser.add_argument('--train_chrs', nargs='+', type=str, required=True, help='List of chromosomes used for training')
    parser.add_argument('--valid_chrs', nargs='+', type=str, required=True, help='List of chromosomes used for validation')
    parser.add_argument('--test_chrs', nargs='+', type=str, required=True, help='List of chromosomes used for testing')
    parser.add_argument('--plot_type', type=str, required=True, help='Type of the plot (auroc or f1)')
    parser.add_argument('--chosen_threshold', type=float, required=True, help='Chosen threshold for classification')
    parser.add_argument('--ylim', type=float, required=True, help='The limit for the y axis')
    
    parser.add_argument('--output', type=str, required=True, help='Path to the output plot file.')
    args = parser.parse_args()
    main(args)