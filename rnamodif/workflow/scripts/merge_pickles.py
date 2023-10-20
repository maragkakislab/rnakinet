import pickle
import argparse
from collections import defaultdict

def merge_dicts(pickle_files):
    merged_dict = {}
    
    for pkl_file in pickle_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                if not isinstance(data, dict):
                    raise ValueError(f"File {pkl_file} does not contain a dictionary.")
                merged_dict.update(data)
        except Exception as e:
            print(f"Error reading {pkl_file}: {str(e)}")
            
    return merged_dict

def save_dict(data, output_file):
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Error writing output file {output_file}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Merge dictionaries from pickle files.")
    parser.add_argument("--pickles", nargs='+', help="Paths of the pickle files to merge.")
    parser.add_argument("--output", required=True, help="Path to save the merged pickle file.")
    args = parser.parse_args()

    merged_data = merge_dicts(args.pickles)
    save_dict(merged_data, args.output)
    print(f"Merged data saved to {args.output}")

if __name__ == "__main__":
    main()
