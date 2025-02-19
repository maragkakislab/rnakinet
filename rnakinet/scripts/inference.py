import argparse
from pathlib import Path
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
import sys
import os
from tqdm import tqdm

from rnakinet.data_utils.dataloading import UnlimitedReadsInferenceDatasetFast5, UnlimitedReadsInferenceDatasetPod5
from rnakinet.models.model import RNAkinet
from rnakinet.models.model_experimental import RNAkinet_LastOnly

from rnakinet.data_utils.workers import worker_init_fn_inference

def run(args):
    print('CUDA', torch.cuda.is_available())
    reads_format = args.format
    files = list(Path(args.path).rglob(f'*.{reads_format}'))
    print(f'Number of {reads_format} files found: {len(files)}')
    if(len(files)==0):
        raise Exception(f'No {reads_format} files found')
        
    kit_architecture = {
        'r9':RNAkinet, 
        'r10':RNAkinet_LastOnly
    }
    base_dir = os.path.dirname(os.path.dirname(__file__))
    kit_checkpoint = {
        'r9':os.path.join(base_dir, 'models', 'rnakinet_r9.ckpt'), 
        'r10':os.path.join(base_dir, 'models', 'rnakinet_r10.ckpt')
    }
    print('Using checkpoint', kit_checkpoint[args.kit])
    
    model = kit_architecture[args.kit]()
    
    model.load_state_dict(torch.load(kit_checkpoint[args.kit], map_location='cpu')['state_dict'])
    model.eval()
    
    if torch.cuda.is_available() and not args.use_cpu:
        model.cuda()
    else:
        model.cpu()
        
    format_dataset_map = {
        'fast5': UnlimitedReadsInferenceDatasetFast5,
        'pod5': UnlimitedReadsInferenceDatasetPod5,
    }
    
    dset = format_dataset_map[reads_format](files=files, max_len=args.max_len, min_len=args.min_len, skip=args.skip)
    
    dataloader = DataLoader(
        dset, 
        batch_size=args.batch_size, 
        num_workers=min([args.max_workers, len(dset.files)]), 
        pin_memory=False, 
        worker_init_fn=worker_init_fn_inference
    )

    id_to_pred = {}
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs, ids = batch
            if torch.cuda.is_available() and not args.use_cpu:
                inputs = inputs.cuda()
            else:
                inputs = inputs.cpu()
                
            outputs = model(inputs)
            
            readid_probs = zip(ids['readid'], outputs.cpu().numpy())
            for readid, probab in readid_probs:
                assert len(probab) == 1
                id_to_pred[readid] = probab[0]
                
    with open(args.output, 'wb') as handle:
        df = pd.DataFrame.from_dict(id_to_pred, orient='index').reset_index()
        df.columns = ['read_id', '5eu_mod_score']
        df['5eu_modified_prediction'] = df['5eu_mod_score'] > args.threshold
        df.to_csv(handle, index=False)
        
def main():
    
    parser = argparse.ArgumentParser(description='Run prediction on FAST5 files')
    parser.add_argument('--path', type=str, required=True, help='Path to the folder containing FAST5 files.')
    parser.add_argument('--kit', type=str, default='r9', help='Sequencing kit used to produce FAST5 files. Model will be selected based on this.', choices=['r9','r10'])
    parser.add_argument('--output', type=str, required=True, help='Path to the output csv file for pooled predictions.')
    parser.add_argument('--max-workers', type=int, default=16, help='Maximum number of workers for data loading')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for data loading')
    parser.add_argument('--max-len', type=int, default=400000, help='Maximum length of the signal sequence to process')
    parser.add_argument('--min-len', type=int, default=5000, help='Minimum length of the signal sequence to process')
    parser.add_argument('--skip', type=int, default=5000, help='How many signal steps to skip at the beginning of each sequence (trimming)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for the predictions to be considered positives')
    parser.add_argument('--use-cpu', action='store_true', help='Use CPU for computation instead of GPU')
    parser.add_argument('--format', type=str, default='fast5', help='Format of read files', choices=['fast5','pod5'])

    
    args = parser.parse_args(sys.argv[1:])
    run(args)

if __name__ == "__main__":
    main()