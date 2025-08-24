import argparse
import torch
from torch.utils.data import DataLoader
import pandas as pd
import sys
from tqdm import tqdm

from rnakinet.data_utils.dataloading_pod5_batched import InferenceDataset
from rnakinet.data_utils.workers import worker_init_fn_inference
from rnakinet.scripts.helpers import arch_map

def run(args):
    print('CUDA', torch.cuda.is_available())
    files = args.pod5_files
    print(f'Number of pod5 files found: {len(files)}')
    if(len(files)==0):
        raise Exception(f'No pod5 files found')
        
    model_path = args.model_path
    print('Using checkpoint', model_path)
    
    model = arch_map[args.arch]()
    
    model.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'])
    model.eval()
    
    if torch.cuda.is_available() and not args.use_cpu:
        model.cuda()
    else:
        model.cpu()
        
    dset = InferenceDataset(files=files, max_len=args.max_len, min_len=args.min_len, skip=args.skip)
    
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
    parser = argparse.ArgumentParser(description='Run prediction on POD5 files')
    parser.add_argument('--pod5-files', type=str, required=True, nargs='+', help='Paths to POD5 files.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model weights.')
    parser.add_argument('--arch', type=str, default='rnakinet', help='Architecture of the model')
    parser.add_argument('--output', type=str, required=True, help='Path to the output csv file.')
    parser.add_argument('--max-workers', type=int, default=16, help='Maximum number of workers for data loading')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for data loading')
    parser.add_argument('--max-len', type=int, default=400000, help='Maximum length of the signal sequence to process')
    parser.add_argument('--min-len', type=int, default=5000, help='Minimum length of the signal sequence to process')
    parser.add_argument('--skip', type=int, default=5000, help='How many signal steps to skip at the beginning of each sequence (trimming)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for the predictions to be considered positives')
    parser.add_argument('--use-cpu', action='store_true', help='Use CPU for computation instead of GPU')
    
    args = parser.parse_args(sys.argv[1:])
    run(args)

if __name__ == "__main__":
    main()