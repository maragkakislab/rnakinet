import argparse
from pathlib import Path
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pickle
from collections import defaultdict
import numpy as np

from rnamodif.data_utils.dataloading_5eu import CompleteReadsInferenceDataset
from rnamodif.data_utils.dataloading_uncut import UnlimitedReadsInferenceDataset
from rnamodif.model import RodanPretrained
from rnamodif.model_uncut import RodanPretrainedUnlimited
from rnamodif.data_utils.workers import worker_init_fn_inference
        
def main(args):
    files = list(Path(args.path).rglob('*.fast5'))
    print('Number of fast5 files found:', len(files))
    if(args.limit):
        files = list(Path(args.path).rglob('*.fast5'))[:args.limit]
        print('Using only', args.limit, 'files')
        #TODO if number of files is less than args.limit, doesnt make any sense
    
    if(not args.window):
        dset = UnlimitedReadsInferenceDataset(files=files)
        #TODO len_limit default value not enough for some reads? Throw away/increase?
        model = RodanPretrainedUnlimited().load_from_checkpoint(args.checkpoint, weighted_loss=args.weighted)
        pin_memory=False
    else:
        dset = CompleteReadsInferenceDataset(files=files, window=args.window, stride=args.window - args.overlap)
        model = RodanPretrained().load_from_checkpoint(args.checkpoint, weighted_loss=args.weighted)
        pin_memory=True

    dataloader = DataLoader(
        dset, 
        batch_size=args.batch_size, 
        num_workers=min([args.max_workers, len(dset.files)]), 
        pin_memory=pin_memory, 
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
    parser.add_argument('--weighted', action='store_true', help='Whether to use weighted loss model')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for data loading (default: 256).')
    parser.add_argument('--window', type=none_or_int, default=4096, help='Window size for data processing (default: 4096).')
    parser.add_argument('--overlap', type=int, default=1024, help='Overlap of neighbouring windows (default: 1024).')
    parser.add_argument('--limit', type=int, default=None, help='Whether to use only subset of data, and how many fast5 files (for faster validation)')
    
    args = parser.parse_args()
    main(args)
