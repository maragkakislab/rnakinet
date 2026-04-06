import argparse
import os
import sys
import warnings

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rnakinet.data_utils.dataloading_pod5_batched import InferenceDataset
from rnakinet.data_utils.workers import worker_init_fn_inference
from rnakinet.models.model_loader import arch_map, default_models


def run(args):
    print('CUDA', torch.cuda.is_available(), file=sys.stderr)

    files = []
    for input_path in args.path:
        if os.path.isdir(input_path):
            # If the input path is a directory, search recursively for POD5 files.
            for root, _, filenames in os.walk(input_path):
                for fname in filenames:
                    if fname.lower().endswith('.pod5'):
                        files.append(os.path.join(root, fname))
        elif os.path.isfile(input_path):
            if input_path.lower().endswith('.pod5'):
                files.append(input_path)
        else:
            raise Exception(f'Path {input_path} is not a valid file or directory')

    print(f'Number of pod5 files found: {len(files)}', file=sys.stderr)
    if len(files) == 0:
        raise Exception(f'No pod5 files found')

    pad_reads = True
    # Load model path and architecture based on param choice.
    if args.model_name:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(base_dir, default_models[args.model_name]['path'])
        arch = default_models[args.model_name]['arch']
        pad_reads = default_models[args.model_name]['pad_reads']
        print('Using pretrained model', args.model_name, 'with checkpoint', model_path, file=sys.stderr)
        model = arch_map[arch]()

    if args.model_path:
        model_path = args.model_path
        print('Using checkpoint', model_path, file=sys.stderr)
        model = arch_map[args.arch]()

    model.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'])
    model.eval()

    if torch.cuda.is_available() and not args.use_cpu:
        model.cuda()
    else:
        model.cpu()
        
    dset = InferenceDataset(files=files, max_len=args.max_len, min_len=args.min_len, skip=args.skip, pad_reads=pad_reads)
    
    dataloader = DataLoader(
        dset,
        batch_size=args.batch_size,
        num_workers=min([args.max_workers, len(dset.files)]),
        pin_memory=False,
        worker_init_fn=worker_init_fn_inference,
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
        fraction_positive = (df['5eu_modified_prediction'] == True).mean()
        df.to_csv(handle, index=False)

    if args.log is not None:
        if args.log:
            log_save_path = args.log
        else:
            log_save_path = os.path.join(os.path.dirname(args.output), 'log.txt')

        log_data = {
            'arch': args.arch if args.arch else default_models[args.model_name]['arch'], # log user-specified arch/model name or provide info on pretrained model used
            'model_path': model_path if args.model_path else default_models[args.model_name]['path'],
            'model_name': args.model_name if args.model_name else 'custom',
            'max_workers': args.max_workers,
            'batch_size': args.batch_size,
            'max_len': args.max_len,
            'min_len': args.min_len,
            'skip': args.skip,
            'threshold': args.threshold,
            'pred_frac_modified': fraction_positive,
        }

        os.makedirs(os.path.dirname(log_save_path) or '.', exist_ok=True)
        with open(log_save_path, 'w') as log_out_file:
            for key, value in log_data.items():
                log_out_file.write(f'{key}: {value}\n')


def main():
    parser = argparse.ArgumentParser(
        description='Run prediction on POD5 files',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    
    # input
    parser.add_argument('--pod5-files', type=str, required=False, nargs='+', help=argparse.SUPPRESS)
    parser.add_argument('--path', type=str, required=False, nargs='+', help='Paths to POD5 files or directories containing POD5 files') # TODO change from pod5 files to something that can include fast5 (and add fast5 support)
   
    # model options
    model_group = parser.add_mutually_exclusive_group(required=True)
    helpstring = 'Name of a pretrained model to use. Choices:\n' + '\n'.join(
        f"  {model_name}: {model_info['tip']}" for model_name, model_info in default_models.items())
    model_group.add_argument('--model-name', type=str, choices=list(default_models.keys()), help=helpstring)
    model_group.add_argument('--model-path', type=str, help='Path to model weights. Must be used in conjunction with --arch')
    parser.add_argument('--arch', type=str, help='Architecture of the model. Must be used in conjunction with --model-path')
    model_group.add_argument('--kit', type=str, choices=['r9', 'r10'], help=argparse.SUPPRESS) # deprecated, use --model-name instead
    
    # output
    parser.add_argument('--output', type=str, required=True, help='Path to the output csv file')
    parser.add_argument('--log', nargs='?', const='', default=None, help='log inference params to file. Saves log.txt to output file dir by default or specify path')

    # inference params
    parser.add_argument('--max-workers', type=int, default=16, help='Maximum number of workers for data loading')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for data loading')
    parser.add_argument('--max-len', type=int, default=400000, help='Maximum length of the signal sequence to process')
    parser.add_argument('--min-len', type=int, default=5000, help='Minimum length of the signal sequence to process')
    parser.add_argument('--skip', type=int, default=5000, help='How many signal steps to skip at the beginning of each sequence (trimming)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for the predictions to be considered positives')
    parser.add_argument('--use-cpu', action='store_true', help='Use CPU for computation instead of GPU')
    
    args = parser.parse_args(sys.argv[1:])

    # warnings and checks
    if args.pod5_files:
        warnings.warn("--pod5-files is deprecated and will be removed in a future release. Use --path instead.", FutureWarning, stacklevel=2)
        if not args.path:
            args.path = args.pod5_files
        else:
            args.path.extend(args.pod5_files)

    if not (args.path or args.pod5_files):
        parser.error('--path is required')

    if args.kit:
        warnings.warn("--kit is deprecated and will be removed in a future release. Use --model-name instead.", FutureWarning, stacklevel=2)
        if args.kit == 'r9':
            args.model_name = 'r9_5EU_v1.0'
        elif args.kit == 'r10':
            args.model_name = 'r10_5EU_v2.0'
            
    if args.model_path and not args.arch:
        parser.error('--arch must be explicitly specified when using --model-path')
    if args.arch and not args.model_path:
        parser.error('--model-path is required when using --arch')

    run(args)


if __name__ == '__main__':
    main()
