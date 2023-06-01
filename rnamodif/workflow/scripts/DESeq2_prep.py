import pandas as pd
import argparse

def get_condition(sample):
    tags = ['_ctrl_','_NoArs_', '_Ars_']
    for tag in tags:
        if(tag in sample):
            return tag.strip('_')
        
def get_replicate(sample):
    replicates = ['5P_1','5P_2','5P_3']
    for replicate in replicates:
        if(replicate in sample):
            return replicate.strip('5P_')

def main(input_path, output_path, output_meta, id_col, sample_col, count_col):
    df = pd.read_csv(input_path, sep='\t')
    pivot_df = df.pivot(index=id_col, columns=sample_col, values=count_col)
    pivot_df.to_csv(output_path, sep='\t')

    meta_df = pd.DataFrame([{'sample':sample, 'condition':get_condition(sample), 'replicate':get_replicate(sample)} for sample in df[sample_col].unique()])
    meta_df.to_csv(output_meta, sep='\t', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process count data.')
    parser.add_argument('--input-path', help='Input file path')
    parser.add_argument('--output-path', help='Output file path for pivoted data')
    parser.add_argument('--output-meta', help='Output file path for metadata')
    parser.add_argument('--id-col', default='Gene stable ID', help='Column name for gene IDs')
    parser.add_argument('--sample-col', default='sample', help='Column name for sample IDs')
    parser.add_argument('--count-col', default='count_sum', help='Column name for count data')

    args = parser.parse_args()
    main(args.input_path, args.output_path, args.output_meta, args.id_col, args.sample_col, args.count_col)
