import sys
sys.path.append('./RODAN')

from rnamodif.data_utils.dataloading import get_test_dataset
from rnamodif.evaluation.evaluation import run_test
from rnamodif.architectures.rodan_pretrained_MIL import RodanPretrainedMIL
from rnamodif.architectures.rodan_simple import RodanSimple

import numpy as np
import sys
from pathlib import Path
import pandas as pd
import argparse

def predictions_to_read_predictions(predictions):
    agg_preds = []
    read_ids = []
    for preds, ids in predictions:
        agg_preds.append(preds.numpy())
        read_ids.append(ids['readid'])
    read_ids = np.concatenate(read_ids)
    agg_preds = np.concatenate(agg_preds)
    results = {}
    for un_read_id in np.unique(read_ids):
        indicies = np.where(read_ids == un_read_id)
        results[un_read_id] = agg_preds[indicies]
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir")
    parser.add_argument("--workers", default=1)
    parser.add_argument("--batchsize", default=32)
    parser.add_argument("--model", default='m6a_v3') 
    parser.add_argument("--outfile")
    
    models_dict = {'m6a_v1':'rnamodif/checkpoints_pl/m6a_nih_33_deploy/epoch=0-step=443500.ckpt',
                   'm6a_v2':'rnamodif/checkpoints_pl/m6a_nih_mix_deploy/epoch=0-step=557500.ckpt',
                   'm6a_v3':'rnamodif/checkpoints_pl/m6a_all_mix_deploy/last.ckpt',
                   '5eu_v1':'rnamodif/checkpoints_pl/5eu_nih_light_conv/last.ckpt',
                   's4u_v1':'rnamodif/checkpoints_pl/s4u_0v33_big/last.ckpt'}
    model_arch_dict = {
        'v1':RodanPretrainedMIL,
        'v2':RodanPretrainedMIL,
        'v3':RodanPretrainedMIL,
        '5eu_v1':RodanSimple,
        's4u_v1':RodanPretrainedMIL,
    }
    
    max_thresholds = {'v1':0.85,'v2':0.85, 'v3':0.95, '5eu_v1':0.9, 's4u_v1':0.9} #Derived from ROC curves
    
    args = parser.parse_args()
    
    directory = Path(args.datadir)
    result_file_name = args.outfile
    files = list(directory.rglob('*.fast5'))
    if(len(files) == 0 and directory.suffix == '.fast5'):
        files = [directory]
    assert(len(files) > 0)
    
    window = 4096
    stride = 3584
    max_threshold = max_thresholds[args.model]

    workers = min([int(args.workers), len(files)])
    print(f'using {workers} workers')
    checkpoint = models_dict[args.model]
    test_dset = get_test_dataset(files, window=window, normalization='rodan', trim_primer=False, stride=stride)
    predictions = run_test(test_dset,checkpoint=checkpoint, workers=workers, architecture=model_arch_dict[args.model])
    read_predictions = predictions_to_read_predictions(predictions)

    read_label_dict = {}
    for k,v in read_predictions.items():
        read_label_dict[k] = np.max(v) > max_threshold

    res = {'read id':[], 'is read modified':[]}
    for k,v in read_label_dict.items():
        res['read id'].append(k)
        res['is read modified'].append(v)
    
    results_df = pd.DataFrame.from_dict(res)
    results_df.to_csv(result_file_name,index=False)
    
if __name__=='__main__':
    main()