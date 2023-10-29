import argparse
import pandas as pd

def main(args):
    table = pd.read_csv(args.table, sep='\t')
    for read_col in args.read_cols:
        table = table[table[read_col] > args.min_reads]
    if(args.max_measured_halflife!="None"):
        table = table[table['t5'] < float(args.max_measured_halflife)]
    table.to_csv(args.output, sep='\t')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prediction on FAST5 files and save results in a pickle file.')
    parser.add_argument('--table', type=str, required=True)
    parser.add_argument('--min-reads', type=int, required=True)
    parser.add_argument('--max-measured-halflife', type=str, required=True)
    parser.add_argument('--read-cols', type=str, required=True, nargs='+')
    parser.add_argument('--output', type=str, help='Path to the output plot.')
    args = parser.parse_args()
    main(args)
    
    
    
