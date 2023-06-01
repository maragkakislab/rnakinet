import pysam
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import argparse

def read_predictions(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)

def get_chromosome_dictionary(bam_file, prediction_dict):
    chromosome_dict = {}
    samfile = pysam.AlignmentFile(bam_file, 'rb')
    for read in samfile.fetch():
        if read.query_name in prediction_dict:
            chromosome_dict[read.query_name] = samfile.get_reference_name(read.reference_id)
    samfile.close()
    return chromosome_dict

def calculate_chromosome_stats(chromosome_dict, prediction_dict, label):
    acceptable_chromosomes = [str(i) for i in range(0,23)]+['X','Y','MT']
    chr_to_preds = defaultdict(list)
    for read_id, chromosome in chromosome_dict.items():
        if(chromosome not in acceptable_chromosomes):
            # print(chromosome, '-skip')
            continue
        chr_to_preds[chromosome].append(prediction_dict[read_id])
    
    chr_stats = defaultdict(list)
    for chromosome, preds in chr_to_preds.items():
        chr_stats[chromosome] = {'accuracy':np.mean([round(p) == label for p in preds]),'count':len(preds)}
        
    # Sort dictionary keys numerically and alphabetically
    sorted_dict_keys = sorted(chr_stats, key=lambda k: (int(k) if k.isdigit() else float('inf'), k))
    chr_stats = {k: chr_stats[k] for k in sorted_dict_keys}
    return chr_stats
                                                    

def process_dict(input_dict):
    df = pd.DataFrame(input_dict).T.reset_index()
    df.columns = ['Chromosome', 'Accuracy', 'Count']
    return df

def plot_accuracies(positive_dict, negative_dict):
    pos_df = process_dict(positive_dict)
    neg_df = process_dict(negative_dict)
    
    # Merging two dataframes based on 'Chromosome'
    final_df = pd.merge(pos_df, neg_df, on='Chromosome', suffixes=('_positive', '_negative'))
    final_df['Balanced_accuracy'] = (final_df['Accuracy_positive']+final_df['Accuracy_negative'])/2
    # Setting up the figure and axis
    fig, ax = plt.subplots(figsize=(10,5))
    bar_width = 0.35
    index = np.arange(len(final_df['Chromosome']))

    bar1 = ax.bar(index - bar_width/2, final_df['Accuracy_positive'], bar_width, label='Positive')
    bar2 = ax.bar(index + bar_width/2, final_df['Accuracy_negative'], bar_width, label='Negative')

    for i in range(len(final_df['Chromosome'])):
        
        ax.hlines(final_df['Balanced_accuracy'][i], i - bar_width/1.2, 
                  i + bar_width/1.2, colors='black', linestyles='-', lw=2, 
                  label='Balanced accuracy' if i==0 else None)

    ax.set_xlabel('Chromosome')
    ax.set_ylabel('Accuracy')
    ax.set_title('Positive and Negative Accuracies by Chromosome')
    ax.set_xticks(index)
    ax.set_xticklabels(final_df['Chromosome'], rotation=45)
    ax.legend(loc='center left', bbox_to_anchor=(1,0.5), prop={'size':10})


def plot_chr_accuracies(pos_bam, neg_bam, pos_pred, neg_pred):
    pos_predictions = read_predictions(pos_pred)
    neg_predictions = read_predictions(neg_pred)

    pos_chromosome_dict = get_chromosome_dictionary(pos_bam, pos_predictions)
    neg_chromosome_dict = get_chromosome_dictionary(neg_bam, neg_predictions)

    pos_stats = calculate_chromosome_stats(pos_chromosome_dict, pos_predictions, 1)
    neg_stats = calculate_chromosome_stats(neg_chromosome_dict, neg_predictions, 0)
                                                    
    plot_accuracies(pos_stats, neg_stats)


def main(args):
    pos_bam = args.positives_bam
    neg_bam = args.negatives_bam
    pos_pred = args.positives_predictions
    neg_pred = args.negatives_predictions
    output_file = args.output
    
    plot_chr_accuracies(pos_bam, neg_bam, pos_pred, neg_pred)
    plt.savefig(output_file, bbox_inches='tight')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prediction on FAST5 files and save results in a pickle file.')
    parser.add_argument('--positives_bam', type=str, required=True, help='Bam file for positive reads')
    parser.add_argument('--negatives_bam', type=str, required=True, help='Bam file for negative reads')
    parser.add_argument('--positives_predictions', type=str, required=True, help='Prediction file for positive reads')
    parser.add_argument('--negatives_predictions', type=str, required=True, help='Prediction file for negative reads')
    parser.add_argument('--output', type=str, required=True, help='Path to the output plot file.')
    args = parser.parse_args()
    main(args)