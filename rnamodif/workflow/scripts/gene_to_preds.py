import argparse
import pandas as pd
import pysam
import pickle


def main(args):
    bamfile = pysam.AlignmentFile(args.transcriptome_bam, 'rb')
    read_to_transcript_data = [
        {'read_id':read.query_name, 'transcript':read.reference_name} 
        for read in bamfile 
        if (not read.is_unmapped and not read.is_supplementary)
    ]
    bamfile.close()
    read_to_transcript = pd.DataFrame(read_to_transcript_data)
    transcript_to_gene = pd.read_csv(args.transcript_to_gene_table, sep='\t')    
    read_to_gene = read_to_transcript.merge(
        transcript_to_gene, 
        left_on='transcript', 
        right_on='Transcript stable ID version', 
        how='left'
    )
    
    with open(args.predictions, 'rb') as file:
        preds = pickle.load(file)

    read_to_score = pd.DataFrame(preds.items(), columns=['read_id', 'score'])
    
    read_gene_score = read_to_score.merge(read_to_gene, how='left', on='read_id')
    read_gene_score['predicted_modified'] = (read_gene_score['score'] > args.threshold).astype(int)
    print(read_gene_score['Gene stable ID'].isnull().sum(), 'reads not paired with a gene - discarded')
    
    
    gene_groups = read_gene_score.groupby('Gene stable ID')
    
    gene_agg = gene_groups['score'].mean().reset_index()
    gene_agg = gene_agg.rename(columns={'score':'average_score'})
    
    gene_agg['percentage_modified'] = gene_groups['predicted_modified'].mean().values
    gene_agg['reads'] = gene_groups.size().values
    
    assert len(gene_agg)>0, 'Empty gene level prediction file'
    gene_agg.to_csv(args.output, sep='\t')
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a table of genes and their modification predictions stats")
    parser.add_argument("--transcriptome-bam", help="transcriptome bam file path")
    parser.add_argument("--transcript-to-gene-table", help="table mapping transcripts to genes")
    parser.add_argument("--predictions", help="predictions for individual reads")
    parser.add_argument("--threshold", type=float, help="threshold to consider a read modified")
    parser.add_argument("--output", help="path to the output table")
    
    args = parser.parse_args()
    main(args)
    