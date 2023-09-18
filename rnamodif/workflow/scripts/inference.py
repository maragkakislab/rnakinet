import argparse
from pathlib import Path
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pickle
from collections import defaultdict
import numpy as np

from rnamodif.data_utils.dataloading_5eu import CompleteReadsInferenceDataset
from rnamodif.data_utils.dataloading_uncut import UnlimitedReadsInferenceDataset
# from rnamodif.models.model import RodanPretrained
# from rnamodif.models.model_uncut import RodanPretrainedUnlimited
from rnamodif.models.model_mine import MyModel
from rnamodif.data_utils.workers import worker_init_fn_inference
from rnamodif.workflow.scripts.helpers import arch_map
        
def main(args):
    files = list(Path(args.path).rglob('*.fast5'))
    print('Number of fast5 files found:', len(files))
    if(args.limit):
        files = list(Path(args.path).rglob('*.fast5'))[:args.limit]
        print('Using only', args.limit, 'files')
        #TODO if number of files is less than args.limit, doesnt make any sense
    
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

    trainer = pl.Trainer(accelerator='gpu', precision=16)
    window_preds = trainer.predict(model, dataloader)

    with open(args.output, 'wb') as handle:
        pickle.dump(window_preds, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
        
def none_or_int(value):
    if value == 'None':
        return None
    return int(value)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prediction on FAST5 files and save results in a pickle file.')
    parser.add_argument('--path', type=str, required=True, help='Path to the folder containing FAST5 files.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint file.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output pickle file for window predictions.')
    parser.add_argument('--max_workers', type=int, default=16, help='Maximum number of workers for data loading (default: 16).')
    # parser.add_argument('--weighted', action='store_true', help='Whether to use weighted loss model')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for data loading (default: 256).')
    parser.add_argument('--max-len', type=int, help='Maximum length of the signal sequence to process')
    parser.add_argument('--min-len', type=int, help='Minimum length of the signal sequence to process')
    parser.add_argument('--skip', type=int, help='How many signal steps to skip at the beginning of each sequence (trimming)')
    
    parser.add_argument('--arch', type=str, required=True, help='Type of architecture.')
    
    parser.add_argument('--limit', type=int, default=None, help='Whether to use only subset of data, and how many fast5 files (for faster validation)')
    
    args = parser.parse_args()
    main(args)
