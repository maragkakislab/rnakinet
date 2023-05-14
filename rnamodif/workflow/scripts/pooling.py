import pickle
import argparse
from collections import defaultdict
import numpy as np

def predictions_to_read_predictions(predictions, pooling):
    id_to_preds = defaultdict(list)
    for pr, ids in predictions:
        readid_probs = zip(ids['readid'], pr.numpy())
        for readid, probab in readid_probs:
            id_to_preds[readid].append(probab)
                
    if(pooling == 'max'):
        for k,v in id_to_preds.items():
            id_to_preds[k] = np.array(v).max()
        return id_to_preds
    
    if(pooling == 'mean'):
        for k,v in id_to_preds.items():
            id_to_preds[k] = np.array(v).mean()
        return id_to_preds
    
    if(pooling == 'none'):
        id_to_preds_nopool = {}
        for k,v in id_to_preds.items():
            for i,prob in enumerate(v):
                id_to_preds_nopool[f'{k}_{i}'] = prob
        return id_to_preds_nopool
    
    else:
        raise Exception(f'{pooling} pooling not implemented')
       
    
def main(args):
    with open(args.window_predictions, 'rb') as file:
        preds = pickle.load(file)
        
    read_preds = predictions_to_read_predictions(preds, pooling=args.pooling)
    with open(args.output, 'wb') as handle:
        pickle.dump(read_preds, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prediction on FAST5 files and save results in a pickle file.')
    parser.add_argument('--window_predictions', type=str, required=True, help='Path to the file containing window predictions.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output pickle file for pooled predictions.')
    parser.add_argument('--pooling', type=str, default='mean', help='Type of pooling to use to combine window predictions to read predictions (default: mean).')
    
    
    args = parser.parse_args()
    main(args)