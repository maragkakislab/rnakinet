import sys
sys.path.append('./RODAN')

from rnamodif.data_utils.dataloading import get_test_dataset
from rnamodif.evaluation.evaluation import run_test
from rnamodif.architectures.rodan_pretrained_MIL import RodanPretrainedMIL
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
    parser.add_argument("--model", default='v1') 
    parser.add_argument("--outfile")
    
    models_dict = {'v1':'rnamodif/checkpoints_pl/m6a_nih_33_deploy/epoch=0-step=443500.ckpt',
                   'v2':'rnamodif/checkpoints_pl/m6a_nih_mix_deploy/epoch=0-step=557500.ckpt'}
    max_thresholds = {'v1':0.85,'v2':0.85} #Derived from ROC curves
    
    args = parser.parse_args()
    
    directory = Path(args.datadir)
    result_file_name = args.outfile
    files = list(directory.rglob('*.fast5'))
    if(len(files) == 0 and directory.suffix == '.fast5'):
        files = [directory]
    assert(len(files) > 0)
    
    window = 4096
    stride = 2048
    max_threshold = max_thresholds[args.model]

    workers = min([args.workers, len(files)])
    checkpoint = models_dict[args.model]
    test_dset = get_test_dataset(files, window=window, normalization='rodan', trim_primer=False, stride=stride)
    predictions = run_test(test_dset,checkpoint=checkpoint, workers=workers, architecture=RodanPretrainedMIL)
    read_predictions = predictions_to_read_predictions(predictions)

    read_label_dict = {}
    for k,v in read_predictions.items():
        read_label_dict[k] = np.max(v) > max_threshold

    res = {'read id':[], 'is m6a modified':[]}
    for k,v in read_label_dict.items():
        res['read id'].append(k)
        res['is m6a modified'].append(v)
    
    results_df = pd.DataFrame.from_dict(res)
    results_df.to_csv(result_file_name,index=False)
    
if __name__=='__main__':
    main()