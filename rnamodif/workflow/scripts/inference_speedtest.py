import argparse
from pathlib import Path
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pickle
from collections import defaultdict
import numpy as np
import torch
from ont_fast5_api.fast5_interface import get_fast5_file
import time
import json
import subprocess

from rnamodif.data_utils.dataloading_5eu import CompleteReadsInferenceDataset
from rnamodif.data_utils.dataloading_uncut import UnlimitedReadsInferenceDataset
# from rnamodif.models.model import RodanPretrained
# from rnamodif.models.model_uncut import RodanPretrainedUnlimited
from rnamodif.models.model_mine import MyModel
from rnamodif.data_utils.workers import worker_init_fn_inference
from rnamodif.workflow.scripts.helpers import arch_map
        
def main(args):
    stats = {}
    print('CUDA', torch.cuda.is_available())
    files = list(Path(args.path).rglob('*.fast5'))
    print('Number of fast5 files found:', len(files))
    
    fast5_readcounts = []
    files_to_infer = -1
    for i,file in enumerate(files):
        with get_fast5_file(file, mode="r") as f5:
            num_reads = len(f5.get_read_ids())
            fast5_readcounts.append(num_reads)
            if(sum(fast5_readcounts)>args.reads_limit and files_to_infer==-1):
                files_to_infer = i
    
    stats['avg_per_file_reads'] = sum(fast5_readcounts)/len(fast5_readcounts)
    stats['using_reads'] = sum(fast5_readcounts[:files_to_infer])
    stats['arch'] = args.arch
    stats['batch_size'] = args.batch_size
    stats['max_workers'] = args.max_workers
    stats['checkpoint'] = args.checkpoint
    stats['experiment'] = args.exp_name
    stats['smake_threads'] = args.smake_threads
    stats['GPU type'] = subprocess.getoutput("nvidia-smi --query-gpu=name --format=csv,noheader,nounits").strip()
    
    print(f'Using {files_to_infer} files')
    print('Avg per file reads', stats['avg_per_file_reads'])
    print('Using reads', stats['using_reads'])
    files = files[:files_to_infer]
            
    # args.reads_limit
    total_start_time = time.time()
    
    arch = arch_map[args.arch]
    model = arch.load_from_checkpoint(args.checkpoint)
    #TODO add support for windowed models
    dset = UnlimitedReadsInferenceDataset(files=files, max_len=args.max_len, min_len=args.min_len, skip=args.skip)
    
    dataloader = DataLoader(
        dset, 
        batch_size=args.batch_size, 
        num_workers=min([args.max_workers, len(dset.files)]), 
        pin_memory=False, 
        worker_init_fn=worker_init_fn_inference
    )
    
    precision = 16
    stats['precision'] = precision
    
    trainer = pl.Trainer(accelerator='gpu', precision=precision)
    
    inference_start_time = time.time()
    window_preds = trainer.predict(model, dataloader)
    
    total_end_time = time.time()
    inference_end_time = time.time()
    
    total_elapsed = total_end_time - total_start_time
    inference_elapsed = inference_end_time - inference_start_time
    
    stats['total_elapsed'] = total_elapsed
    stats['inference_elapsed'] = inference_elapsed

    with open(args.output, 'w') as handle:
        json.dump(stats, handle)
        
        
def none_or_int(value):
    if value == 'None':
        return None
    return int(value)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prediction on FAST5 files and save results in a pickle file.')
    parser.add_argument('--path', type=str, required=True, help='Path to the folder containing FAST5 files.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint file.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output file with statistics about the run.')
    parser.add_argument('--max_workers', type=int, default=16, help='Maximum number of workers for data loading (default: 16).')
    # parser.add_argument('--weighted', action='store_true', help='Whether to use weighted loss model')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for data loading (default: 256).')
    parser.add_argument('--max-len', type=int, help='Maximum length of the signal sequence to process')
    parser.add_argument('--min-len', type=int, help='Minimum length of the signal sequence to process')
    parser.add_argument('--skip', type=int, help='How many signal steps to skip at the beginning of each sequence (trimming)')
    
    parser.add_argument('--arch', type=str, required=True, help='Type of architecture.')
    parser.add_argument('--exp_name', type=str, required=True, help='Name of the used experiment.')
    
    parser.add_argument('--smake_threads', type=str, required=True, help='Number of provided snakemake threads')
    
    
    parser.add_argument('--reads_limit', type=int, default=None, help='The limit on reads to use for testing the speed')
    
    args = parser.parse_args()
    main(args)
