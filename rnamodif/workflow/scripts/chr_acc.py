import pysam
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import argparse
import seaborn as sns

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

def calculate_chromosome_stats(chromosome_dict, prediction_dict, label, threshold):
    acceptable_chromosomes = [str(i) for i in range(0,23)]+['X','Y','MT']
    chr_to_preds = defaultdict(list)
    for read_id, chromosome in chromosome_dict.items():
        if(chromosome not in acceptable_chromosomes):
            # print(chromosome, '-skip')
            continue
        chr_to_preds[chromosome].append(prediction_dict[read_id])
    
    chr_stats = defaultdict(list)
    for chromosome, preds in chr_to_preds.items():
        # chr_stats[chromosome] = {'accuracy':np.mean([round(p) == label for p in preds]),'count':len(preds)}
        chr_stats[chromosome] = {'accuracy':np.mean(np.where(np.array(preds)>threshold, 1, 0) == label),'count':len(preds)}
        
        
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
    final_df = final_df.rename(columns={"Accuracy_positive":'True positive rate','Accuracy_negative':'True negative rate'})
    final_df['Balanced_accuracy'] = (final_df['True positive rate'] + final_df['True negative rate']) / 2

    # Preparing data in 'tidy' format for seaborn
    tidy_df = pd.melt(final_df, id_vars=['Chromosome', 'Balanced_accuracy'], 
                      value_vars=['True positive rate', 'True negative rate'], 
                      var_name='Accuracy_type', value_name='Accuracy')

    # Setting up the figure and axis
    plt.figure(figsize=(10, 5))

    # Colorblind-friendly color palette
    colors = sns.color_palette("colorblind")

    # Seaborn barplot
    sns.barplot(x='Chromosome', y='Accuracy', hue='Accuracy_type', data=tidy_df, palette=colors)

    # Balanced accuracy lines
    for i, acc in enumerate(final_df['Balanced_accuracy']):
        plt.hlines(acc, i - 0.2, i + 0.2, colors='black', linestyles='-', lw=2, 
                   label='Balanced accuracy' if i==0 else None)

    plt.xlabel('Chromosome', fontsize=16)
    plt.ylabel('TPR and TNR', fontsize=16)

    # Hide the top and right spines
    sns.despine()

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 14})
    plt.tight_layout()
    # plt.show()


#TODO rename
def plot_chr_accuracies(group_to_data, threshold):
    group_to_df = {}
    for group, data in group_to_data.items():
        pos_bams = data['positives']['bams']
        pos_preds = data['positives']['preds']
        
        neg_bams = data['negatives']['bams']
        neg_preds = data['negatives']['preds']
        
        pos_chromosome_dict_merged = {}
        pos_prediction_dict_merged = {}
        
        for pos_bam, pos_pred in zip(pos_bams, pos_preds):
            pos_predictions = read_predictions(pos_pred)
            pos_chromosome_dict = get_chromosome_dictionary(pos_bam, pos_predictions)
            pos_chromosome_dict_merged.update(pos_chromosome_dict)
            pos_prediction_dict_merged.update(pos_predictions)

        neg_chromosome_dict_merged = {}
        neg_prediction_dict_merged = {}
        for neg_bam, neg_pred in zip(neg_bams, neg_preds):
            neg_predictions = read_predictions(neg_pred)
            neg_chromosome_dict = get_chromosome_dictionary(neg_bam, neg_predictions)
            neg_chromosome_dict_merged.update(neg_chromosome_dict)
            neg_prediction_dict_merged.update(neg_predictions)

        pos_stats = calculate_chromosome_stats(pos_chromosome_dict_merged, pos_prediction_dict_merged, 1, threshold)
        neg_stats = calculate_chromosome_stats(neg_chromosome_dict_merged, neg_prediction_dict_merged, 0, threshold)
        pos_df = process_dict(pos_stats)
        neg_df = process_dict(neg_stats)  
        final_df = pd.merge(pos_df, neg_df, on='Chromosome', suffixes=('_positive', '_negative'))
        final_df = final_df.rename(columns={"Accuracy_positive":'True positive rate','Accuracy_negative':'True negative rate'})
        final_df['Balanced_accuracy'] = (final_df['True positive rate'] + final_df['True negative rate']) / 2
        final_df['F1'] = (2 * final_df['True positive rate'] * final_df['True negative rate'])/(final_df['True positive rate'] + final_df['True negative rate'])
        final_df['group'] = group
        #Chromosomes which != 1 still have a small amount of reads - check why! bad alignment filtering?
        # print(final_df)
        
        group_to_df[group] = final_df
        
        
    # plot_accuracies(pos_stats, neg_stats)
    plot_f1_scores(group_to_df)


def plot_f1_scores(group_to_df):
    # Create an empty DataFrame to store the F1 Scores
    final_df = pd.DataFrame(columns=['Chromosome', 'F1', 'Group'])
    # Iterate through the dictionary and append the data to the final DataFrame
    for group, df in group_to_df.items():
        temp_df = df[['Chromosome', 'F1']].copy()
        temp_df['Group'] = group
        final_df = final_df.append(temp_df, ignore_index=True)

    # Create the plot
    plt.figure(figsize=(15, 6))
    sns.barplot(x='Chromosome', y='F1', hue='Group', data=final_df, palette='viridis')
    plt.xlabel('Chromosome', fontsize=16)
    plt.ylabel('F1 Score', fontsize=16)
    plt.legend(fontsize='16', loc='lower left')
    sns.despine()
    plt.tight_layout()
    
    
def main(args):
    pos_bams = args.positives_bams
    neg_bams = args.negatives_bams
    pos_preds = args.positives_predictions
    neg_preds = args.negatives_predictions
    pos_groups = args.positives_groups_in_order
    neg_groups = args.negatives_groups_in_order
    output_file = args.output
    threshold = args.threshold
    
    group_to_data = {}
    
    for group in pos_groups+neg_groups:
        group_to_data[group]={'positives':{'preds':[],'bams':[]},'negatives':{'preds':[],'bams':[]}}
    
    for pos_file, pos_bam, pos_group in zip(pos_preds, pos_bams, pos_groups):
        group_to_data[pos_group]['positives']['preds'].append(pos_file)
        group_to_data[pos_group]['positives']['bams'].append(pos_bam)

    for neg_file, neg_bam, neg_group in zip(neg_preds, neg_bams, neg_groups):
        group_to_data[neg_group]['negatives']['preds'].append(neg_file)
        group_to_data[neg_group]['negatives']['bams'].append(neg_bam)
    
    plot_chr_accuracies(group_to_data, threshold)
    plt.savefig(output_file, bbox_inches='tight')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prediction on FAST5 files and save results in a pickle file.')
    parser.add_argument('--positives_bams', nargs='+', type=str, required=True, help='Bam files for positive reads')
    parser.add_argument('--negatives_bams', nargs='+', type=str, required=True, help='Bam files for negative reads')
    parser.add_argument('--positives_predictions', nargs='+', type=str, required=True, help='Prediction files for positive reads')
    parser.add_argument('--negatives_predictions', nargs='+', type=str, required=True, help='Prediction files for negative reads')
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
    parser.add_argument('--threshold', type=float, required=True, help='Threshold to decide positive/negative class')
    parser.add_argument('--output', type=str, required=True, help='Path to the output plot file.')
    args = parser.parse_args()
    main(args)