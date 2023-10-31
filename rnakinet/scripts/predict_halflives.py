import argparse
import pandas as pd
import pysam
import pickle
import numpy as np
import sys

def run(args):
    bamfile = pysam.AlignmentFile(args.transcriptome_bam, 'rb')
    read_to_transcript_data = [
        {'read_id':read.query_name, 'transcript':read.reference_name} 
        for read in bamfile 
        if (not read.is_unmapped and not read.is_supplementary)
    ]
    bamfile.close()
    read_to_transcript = pd.DataFrame(read_to_transcript_data)
    
    read_to_score = pd.read_csv(args.predictions)

    read_transcript_score = read_to_score.merge(read_to_transcript, how='left', on='read_id')
    read_transcript_score = read_transcript_score.dropna()
    transcript_groups = read_transcript_score.groupby('transcript')
    transcript_agg = transcript_groups['5eu_modified_prediction'].mean().reset_index()
    transcript_agg = transcript_agg.rename(columns={'5eu_modified_prediction':'percentage_modified'})
    transcript_agg['reads'] = transcript_groups.size().values
    assert len(transcript_agg)>0, 'Empty transcript level prediction file, check if read ids in predictions also appear in the bam file'
    
    transcript_agg = add_predicted_halflifes(transcript_agg, tl=args.tl)
    
    transcript_agg.to_csv(args.output)
    
def add_predicted_halflifes(df, tl):
    col = 'percentage_modified'
    df['pred_t5'] = -tl * np.log(2) / np.log(1-df[col]) 
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[~df['pred_t5'].isna()]
    return df
    
def main():
    parser = argparse.ArgumentParser(description="Create a table of genes and their modification predictions stats")
    parser.add_argument("--transcriptome-bam", help="transcriptome bam file path")
    parser.add_argument("--predictions", help="predictions for individual reads")
    parser.add_argument('--tl', type=float, required=True, help='Time parameter for the decay equation')
    parser.add_argument("--output", help="path to the transcript output table")
    
    args = parser.parse_args(sys.argv[1:])
    run(args)
    