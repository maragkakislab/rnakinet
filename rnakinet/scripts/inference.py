import argparse
import json
import os
import sys
import warnings
from importlib.metadata import version

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rnakinet.data_utils.dataloading_pod5_batched import (
    InferenceDataset, collate_variable, pod5_read_ids, read_id_set)
from rnakinet.data_utils.workers import worker_init_fn_inference
from rnakinet.models.model_loader import (
    arch_map, default_models, get_arch, inference_lengths, input_mode)


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

    # Load model path and architecture based on param choice.
    if args.model_name:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(base_dir, default_models[args.model_name]['path'])
        arch = default_models[args.model_name]['arch']
        print('Using pretrained model', args.model_name, 'with checkpoint', model_path, file=sys.stderr)
        arch_class = get_arch(arch)

    if args.model_path:
        model_path = args.model_path
        print('Using checkpoint', model_path, file=sys.stderr)
        arch_class = get_arch(args.arch)
    # the architecture's hyperparameters as trained (e.g. layer sizes)
    model = arch_class(**json.loads(args.arch_hyperparams))

    # the architecture sets the preprocessing the model was trained with
    mode = input_mode(arch_class)
    pad_reads = mode == 'padded'
    # length limits not given: the architecture's defaults
    default_max_len, default_crop = inference_lengths(arch_class)
    if args.max_len is None:
        args.max_len = default_max_len
    if args.crop is None:
        args.crop = default_crop
    if pad_reads and not args.max_len:
        raise ValueError(f'{arch_class.__name__} pads reads to --max-len: '
                         f'it cannot be 0')

    model.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'])
    model.eval()

    # unpadded reads keep their own lengths and cannot be stacked into
    # batches larger than one
    if mode == 'unpadded' and args.batch_size > 1:
        warnings.warn(f'reads are not padded; using --batch-size 1 '
                      f'instead of {args.batch_size}')
        args.batch_size = 1

    if torch.cuda.is_available() and not args.use_cpu:
        model.cuda()
    else:
        model.cpu()
        
    read_ids = None
    if args.read_ids:
        read_ids = read_id_set(args.read_ids)
        # report missing reads: a wrong --path or ids file would otherwise
        # silently give fewer predictions
        found = sum(r in read_ids for f in files for r in pod5_read_ids(f))
        print(f'Found {found} of {len(read_ids)} listed reads in the POD5 '
              f'files', file=sys.stderr)
        if found == 0:
            raise Exception(f'None of the reads in {args.read_ids} are in '
                            f'the POD5 files')
    dset = InferenceDataset(files=files, max_len=args.max_len,
                            min_len=args.min_len, skip=args.skip,
                            crop=args.crop,
                            pad_reads=pad_reads, read_ids=read_ids)
    
    dataloader = DataLoader(
        dset,
        batch_size=args.batch_size,
        num_workers=min([args.max_workers, len(dset.files)]),
        pin_memory=False,
        worker_init_fn=worker_init_fn_inference,
        collate_fn=collate_variable if mode == 'variable' else None,
    )

    id_to_pred = {}
    with torch.no_grad():
        for batch in tqdm(dataloader):
            device = ('cuda' if torch.cuda.is_available()
                      and not args.use_cpu else 'cpu')
            if mode == 'variable':
                inputs, lengths, ids = batch
                outputs = model(inputs.to(device), lengths)
            else:
                inputs, ids = batch
                outputs = model(inputs.to(device))

            readid_probs = zip(ids['readid'], outputs.cpu().numpy())
            for readid, probab in readid_probs:
                assert len(probab) == 1
                id_to_pred[readid] = probab[0]

    if not id_to_pred:
        upper = (f'--skip + --max-len = {args.skip + args.max_len}'
                 if args.max_len else 'no upper limit')
        warnings.warn(
            'no reads passed the length filter (between --skip + --min-len '
            f'= {args.skip + args.min_len} and {upper} signal samples); '
            'writing an empty table')
    df = pd.DataFrame({'read_id': list(id_to_pred),
                       'mod_score': list(id_to_pred.values())})
    df['mod_prediction'] = df['mod_score'] > args.threshold
    fraction_positive = (df['mod_prediction'].mean() if len(df)
                         else float('nan'))
    df.to_csv(args.output, index=False)

    if args.log is not None:
        if args.log:
            log_save_path = args.log
        else:
            log_save_path = os.path.join(os.path.dirname(args.output), 'log.txt')

        log_data = {
            'arch': args.arch if args.arch else default_models[args.model_name]['arch'], # log user-specified arch/model name or provide info on pretrained model used
            'model_path': model_path if args.model_path else default_models[args.model_name]['path'],
            'model_name': args.model_name if args.model_name else 'custom',
            'modification': (default_models[args.model_name]['modification']
                             if args.model_name else 'unknown'),
            'read_ids': args.read_ids or 'all',
            'listed_reads_found': (f'{found} of {len(read_ids)}'
                                   if args.read_ids else 'n/a'),
            'max_workers': args.max_workers,
            'batch_size': args.batch_size,
            'pad_reads': pad_reads,
            'max_len': args.max_len,
            'min_len': args.min_len,
            'skip': args.skip,
            'crop': args.crop,
            'threshold': args.threshold,
            'reads_predicted': len(df),
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
    parser.add_argument('--version', action='version', version=f'%(prog)s {version("rnakinet")}')
    
    # input
    parser.add_argument('--pod5-files', type=str, required=False, nargs='+', help=argparse.SUPPRESS)
    parser.add_argument('--path', type=str, required=False, nargs='+', help='Paths to POD5 files or directories containing POD5 files') # TODO change from pod5 files to something that can include fast5 (and add fast5 support)
    parser.add_argument('--read-ids', type=str,
                        help='File with one read id per line: predict only '
                             'these reads')
   
    # model options
    model_group = parser.add_mutually_exclusive_group(required=True)
    helpstring = 'Name of a pretrained model to use. Choices:\n' + '\n'.join(
        f"  {model_name}: {model_info['tip']}" for model_name, model_info in default_models.items())
    model_group.add_argument('--model-name', type=str, choices=list(default_models.keys()), help=helpstring)
    model_group.add_argument('--model-path', type=str, help='Path to model weights. Must be used in conjunction with --arch')
    parser.add_argument('--arch-hyperparams', type=str, default='{}',
                        help='Hyperparameters the model was built with, as '
                             'a JSON object (default: the architecture\'s '
                             'defaults)')
    parser.add_argument('--arch', type=str, help=f'Architecture of the model: one of {list(arch_map)}, or <file.py>:<class> for one outside the package. Must be used in conjunction with --model-path')
    model_group.add_argument('--kit', type=str, choices=['r9', 'r10'], help=argparse.SUPPRESS) # deprecated, use --model-name instead
    
    # output
    parser.add_argument('--output', type=str, required=True, help='Path to the output csv file')
    parser.add_argument('--log', nargs='?', const='', default=None, help='log inference params to file. Saves log.txt to output file dir by default or specify path')

    # inference params
    parser.add_argument('--max-workers', type=int, default=16, help='Maximum number of workers for data loading (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for data loading (default: %(default)s)')
    parser.add_argument('--max-len', type=int, default=None,
                        help='Skip reads longer than this many signal '
                             'samples after --skip; 0: no limit (default: '
                             "the architecture's: 0 for v3.0, 400000 for "
                             'v1.0 and v2.0)')
    parser.add_argument('--min-len', type=int, default=5000, help='Minimum length of the signal sequence to process (default: %(default)s)')
    parser.add_argument('--skip', type=int, default=5000, help='How many signal steps to skip at the beginning of each sequence (trimming) (default: %(default)s)')
    parser.add_argument('--crop', type=int, default=None,
                        help='Score only the first CROP samples of longer '
                             "reads; 0: whole reads (default: the "
                             "architecture's: 200000 for v3.0, 0 for v1.0 "
                             'and v2.0)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for the predictions to be considered positives (default: %(default)s)')
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
