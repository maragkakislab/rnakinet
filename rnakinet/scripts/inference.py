import argparse
from pathlib import Path
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pickle
from collections import defaultdict
import numpy as np
import torch
import pandas as pd
import sys
import os

from rnakinet.data_utils.dataloading import UnlimitedReadsInferenceDataset
from rnakinet.models.model import RNAkinet
from rnakinet.data_utils.workers import worker_init_fn_inference

def run(args):
    print('CUDA', torch.cuda.is_available())
    files = list(Path(args.path).rglob('*.fast5'))
    print('Number of fast5 files found:', len(files))
    if(len(files)==0):
        raise Exception('No fast5 files found')
    
    arch = RNAkinet
    model = arch.load_from_checkpoint(args.checkpoint)
    dset = UnlimitedReadsInferenceDataset(files=files, max_len=args.max_len, min_len=args.min_len, skip=args.skip)
    
    dataloader = DataLoader(
        dset, 
        batch_size=args.batch_size, 
        num_workers=min([args.max_workers, len(dset.files)]), 
        pin_memory=False, 
        worker_init_fn=worker_init_fn_inference
    )

    trainer = pl.Trainer(accelerator='gpu', precision=16)
    predictions = trainer.predict(model, dataloader)
    
    id_to_pred = {}
    for pr, ids in predictions:
        readid_probs = zip(ids['readid'], pr.numpy())
        for readid, probab in readid_probs:
            assert len(probab) == 1
            id_to_pred[readid] = probab[0]
            
    with open(args.output, 'wb') as handle:
        df = pd.DataFrame.from_dict(id_to_pred, orient='index').reset_index()
        df.columns = ['read_id', '5eu_mod_score']
        df['5eu_modified_prediction'] = df['5eu_mod_score'] > args.threshold
        df.to_csv(handle, index=False)
        
        
def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    default_checkpoint = os.path.join(base_dir, 'models', 'rnakinet.ckpt')
    
    parser = argparse.ArgumentParser(description='Run prediction on FAST5 files')
    parser.add_argument('--path', type=str, required=True, help='Path to the folder containing FAST5 files.')
    parser.add_argument('--checkpoint', type=str, default=default_checkpoint, help='Path to the model checkpoint file.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output csv file for pooled predictions.')
    parser.add_argument('--max-workers', type=int, default=16, help='Maximum number of workers for data loading (default: 16).')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for data loading (default: 256).')
    parser.add_argument('--max-len', type=int, default=400000, help='Maximum length of the signal sequence to process')
    parser.add_argument('--min-len', type=int, default=5000, help='Minimum length of the signal sequence to process')
    parser.add_argument('--skip', type=int, default=5000, help='How many signal steps to skip at the beginning of each sequence (trimming)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for the predictions to be considered positives')
    
    args = parser.parse_args(sys.argv[1:])
    run(args)
