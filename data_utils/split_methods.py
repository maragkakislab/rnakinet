from datamap import experiment_files
from pathlib import Path
import random

def get_kfold_split_func(total_k, current_k, shuffle=True):
    def f(pos_files, neg_files):
        sortkey = lambda x: int(Path(x).stem.split('_')[-1])
        pos_files = sorted(experiment_files[pos_files], key=sortkey)
        neg_files = sorted(experiment_files[neg_files], key=sortkey)
        
        if(shuffle):
            seed = 42
            deterministic_random = random.Random(seed)
            deterministic_random.shuffle(pos_files)
            deterministic_random.shuffle(neg_files)
        
        pos_k_size = len(pos_files)//total_k
        neg_k_size = len(neg_files)//total_k
        
        valid_pos_files = pos_files[pos_k_size*current_k:pos_k_size*(current_k+1)]
        valid_neg_files = neg_files[neg_k_size*current_k:neg_k_size*(current_k+1)]
        
        train_pos_files = pos_files[:pos_k_size*current_k] + pos_files[pos_k_size*(current_k+1):]
        train_neg_files = neg_files[:neg_k_size*current_k] + neg_files[neg_k_size*(current_k+1):]
        
        return {
            'train_pos_files':train_pos_files,
            'train_neg_files':train_neg_files,
            'valid_pos_files':valid_pos_files,
            'valid_neg_files':valid_neg_files,
        }
        
        
    return f


def get_default_split(pos_files, neg_files):
    valid_select_seed = 42
    valid_files_count = 10
    
    sortkey = lambda x: int(Path(x).stem.split('_')[-1])
    pos_files = sorted(experiment_files[pos_files], key=sortkey)
    neg_files = sorted(experiment_files[neg_files], key=sortkey)
    
    seed = valid_select_seed
    deterministic_random = random.Random(seed)
    deterministic_random.shuffle(pos_files)
    deterministic_random.shuffle(neg_files)
    
    train_pos_files = pos_files[:-valid_files_count]
    train_neg_files = neg_files[:-valid_files_count]
    valid_pos_files = pos_files[-valid_files_count:]
    valid_neg_files = neg_files[-valid_files_count:]
    
    return {
        'train_pos_files':train_pos_files,
        'train_neg_files':train_neg_files,
        'valid_pos_files':valid_pos_files,
        'valid_neg_files':valid_neg_files,
    }