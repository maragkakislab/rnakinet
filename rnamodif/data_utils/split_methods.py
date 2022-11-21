from rnamodif.data_utils.datamap import experiment_files
from pathlib import Path
import random

def get_kfold_splits(pos_exps, neg_exps, total_k, current_k, shuffle=True, limit=None, verbose=False):
    splits = []
    split_fn = kfold_split_single(total_k, current_k, shuffle, limit, verbose)
    for exp in pos_exps:
        splits.append(split_fn(exp, 'pos'))
    for exp in neg_exps:
        splits.append(split_fn(exp, 'neg'))
    return splits

def kfold_split_single(total_k, current_k, shuffle=True, limit=None, verbose=False):
    def f(exp, label):
        if label not in ['pos','neg']:
            raise Exception('needs to be pos or neg')
        
        files = sorted(experiment_files[exp])
        
        if(shuffle):
            seed = 42
            deterministic_random = random.Random(seed)
            deterministic_random.shuffle(files)
        
        if(limit):
            files = files[:limit]
        
        k_size = len(files)//total_k
        
        valid_files = files[k_size*current_k:k_size*(current_k+1)]
        train_files = files[:k_size*current_k] + files[k_size*(current_k+1):]
        
        assert len(set(train_files).intersection(valid_files)) == 0
        
        if(verbose):
            print(f'FAST5 {label} files counts')
            print('valid_files', len(valid_files))
            print('train_files', len(train_files))
        
        return {
            f'train_{label}_files':train_files,
            f'valid_{label}_files':valid_files,
        }
        
    return f

def kfold_split(total_k, current_k, shuffle=True, limit=None, verbose=False):
    def f(pos_files, neg_files):
        pos_files = sorted(experiment_files[pos_files])
        neg_files = sorted(experiment_files[neg_files])
        
        if(shuffle):
            seed = 42
            deterministic_random = random.Random(seed)
            deterministic_random.shuffle(pos_files)
            deterministic_random.shuffle(neg_files)
        
        if(limit):
            pos_files = pos_files[:limit]
            neg_files = neg_files[:limit]
        
        pos_k_size = len(pos_files)//total_k
        neg_k_size = len(neg_files)//total_k
        
        #TODO dont throw aways the last samples cutoff
        valid_pos_files = pos_files[pos_k_size*current_k:pos_k_size*(current_k+1)]
        valid_neg_files = neg_files[neg_k_size*current_k:neg_k_size*(current_k+1)]
        
        train_pos_files = pos_files[:pos_k_size*current_k] + pos_files[pos_k_size*(current_k+1):]
        train_neg_files = neg_files[:neg_k_size*current_k] + neg_files[neg_k_size*(current_k+1):]
        
        assert len(set(train_pos_files).intersection(valid_pos_files)) == 0
        assert len(set(train_neg_files).intersection(valid_neg_files)) == 0
        
        assert len(set(train_pos_files).intersection(train_neg_files)) == 0
        assert len(set(valid_pos_files).intersection(valid_neg_files)) == 0
        
        if(verbose):
            print('FAST5 files counts')
            print(pos_files, neg_files)
            print('valid_pos_files', len(valid_pos_files))
            print('valid_neg_files', len(valid_neg_files))
            print('train_pos_files', len(train_pos_files))
            print('train_neg_files', len(train_neg_files))
        
        return {
            'train_pos_files':train_pos_files,
            'train_neg_files':train_neg_files,
            'valid_pos_files':valid_pos_files,
            'valid_neg_files':valid_neg_files,
        }
        
        
    return f



#LEGACY - used for labelcleaning 2022
def get_experiment_sort(exp_name):  
    feu_to_index=lambda x: int(x.stem.split('_')[-1])
    covid_to_index=lambda x: int(x.stem[5:])
    if(exp_name in ['pos_2022','pos_2020','neg_2022','neg_2020']):
        index_func = feu_to_index
    else:
        index_func = covid_to_index
    return index_func

def get_kfold_split_func(total_k, current_k, shuffle=True, limit=None):
    def f(pos_files, neg_files):
        sortkey = lambda x: int(Path(x).stem.split('_')[-1])
        pos_files = sorted(experiment_files[pos_files], key=get_experiment_sort(pos_files))
        neg_files = sorted(experiment_files[neg_files], key=get_experiment_sort(neg_files))
        
        if(shuffle):
            seed = 42
            deterministic_random = random.Random(seed)
            deterministic_random.shuffle(pos_files)
            deterministic_random.shuffle(neg_files)
        
        if(limit):
            pos_files = pos_files[:limit]
            neg_files = neg_files[:limit]
        
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
    pos_files = sorted(experiment_files[pos_files], key=get_experiment_sort(pos_files))
    neg_files = sorted(experiment_files[neg_files], key=get_experiment_sort(neg_files))
    
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


def get_fullvalid_split(limit=None, shuffle=False):
    def fullvalid_split(pos_files, neg_files):
        valid_pos_files = experiment_files[pos_files]
        valid_neg_files = experiment_files[neg_files]
        
        if(shuffle):
            random.shuffle(valid_pos_files)
            random.shuffle(valid_neg_files)
        if(limit):
            valid_pos_files=valid_pos_files[:limit]
            valid_neg_files=valid_neg_files[:limit]
        return {
            'train_pos_files':[],
            'train_neg_files':[],
            'valid_pos_files':valid_pos_files,
            'valid_neg_files':valid_neg_files,
        }
    
    return fullvalid_split