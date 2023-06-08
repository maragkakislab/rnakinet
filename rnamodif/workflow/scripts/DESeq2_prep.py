import pandas as pd
import argparse

def get_condition(sample, controls, conditions):
    if(sample in controls):
        return 'CTRL'
    if(sample in conditions):
        return 'COND'
    raise Exception('Experiment missing from the cond/ctrl list')
        
def get_replicate(sample, controls, conditions):
    for i,ctrl_exp in enumerate(controls):
        if(sample == ctrl_exp):
            return i
    for i,cond_exp in enumerate(conditions):
        if(sample == cond_exp):
            return i
    raise Exception('Experiment missing from the cond/ctrl list')

def main(input_path, controls, conditions, output_path, output_meta, id_col, sample_col, count_col):
    df = pd.read_csv(input_path, sep='\t')
    pivot_df = df.pivot(index=id_col, columns=sample_col, values=count_col)
    pivot_df.to_csv(output_path, sep='\t')
    meta_df = pd.DataFrame([
        {
            'sample':sample, 
            'condition':get_condition(sample, controls, conditions), 
            'replicate':get_replicate(sample, controls, conditions)
        } for sample in df[sample_col].unique()
    ])
    meta_df.to_csv(output_meta, sep='\t', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process count data.')
    parser.add_argument('--input-path', help='Input file path')
    parser.add_argument('--controls', nargs='+', help='list of control experiments')
    parser.add_argument('--conditions', nargs='+', help='list of condition experiments')
    parser.add_argument('--output-path', help='Output file path for pivoted data')
    parser.add_argument('--output-meta', help='Output file path for metadata')
    parser.add_argument('--id-col', default='Gene stable ID', help='Column name for gene IDs')
    parser.add_argument('--sample-col', default='sample', help='Column name for sample IDs')
    parser.add_argument('--count-col', default='count_sum', help='Column name for count data')

    args = parser.parse_args()
    main(args.input_path, args.controls, args.conditions, args.output_path, args.output_meta, args.id_col, args.sample_col, args.count_col)
