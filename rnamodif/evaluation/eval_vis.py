from rnamodif.data_utils.dataloading import get_test_dataset
from rnamodif.data_utils.datamap import experiment_files
import torch
from rnamodif.evaluation.evaluation import run_test
from rnamodif.architectures.rodan_pretrained_MIL import RodanPretrainedMIL
from rnamodif.data_utils.split_methods import kfold_split_single
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from matplotlib import pyplot as plt

def get_valid_files(exp, label):
    return kfold_split_single(total_k=5, current_k=0)(exp, label=label)[f'valid_{label}_files']

def get_dsets_preds(checkpoint, arch, exp_label_dict, window=4096, stride = 2048, batch_size=64, workers=4, per_worker_read_limit=None, profiler=None, trim_primer=False):
    exp_label = exp_label_dict
    workers=workers
    architecture=arch
    window=window
    stride = stride
    batch_size=batch_size
    read_limit = per_worker_read_limit
    profiler=profiler #None, 'simple'
    
    dsets_preds = []
    label_to_num = {'pos':1, 'neg':0}
    for exp,label in exp_label.items():
        test_dset = get_test_dataset(get_valid_files(exp, label), window=window, normalization='rodan', trim_primer=trim_primer, stride=stride, read_limit=read_limit)
        predictions = run_test(test_dset,checkpoint=checkpoint, workers=workers, architecture=architecture, batch_size=batch_size, profiler=profiler)
        dsets_preds.append((predictions, label_to_num[label], exp))

    return dsets_preds

def filter_dset_preds(preds, exps):
    result = []
    for predictions, label, exp in preds:
        for exp_lookup in exps:
            if exp_lookup in exp:
                result.append((predictions, label, exp))
    return result



def predictions_to_read_predictions(predictions):
    agg_preds = []
    read_ids = []
    read_starts = []
    for preds, ids in predictions:
        agg_preds.append(preds.numpy())
        read_ids.append(ids['readid'])
        read_starts.append(ids['start'])
    read_ids = np.concatenate(read_ids)
    agg_preds = np.concatenate(agg_preds)
    starts = np.concatenate(read_starts)
    results = {}
    for un_read_id in np.unique(read_ids):
        indicies = np.where(read_ids == un_read_id)
        # print(agg_preds[indicies])
        results[un_read_id] = agg_preds[indicies]
        read_starts = starts[indicies]
        assert(all(read_starts[i] <= read_starts[i+1] for i in range(len(read_starts) - 1))) #Check ascending order of starts
    return results
    

def get_metrics(read_predictions, label, mean_threshold=0.5, max_threshold=0.9):
    mean_accs = []
    max_accs = []
    
    for k,v in read_predictions.items():
        if(label == 1):
            mean_accs.append(np.mean(v) > mean_threshold)
            max_accs.append(np.max(v) > max_threshold)
        else:
            mean_accs.append(np.mean(v) <= mean_threshold)
            max_accs.append(np.max(v) <= max_threshold)
    mean_based_acc = np.mean(mean_accs)
    max_based_acc = np.mean(max_accs)
    if(mean_threshold!=0):
        print('mean based', mean_based_acc)
    print('max based', max_based_acc)
    return mean_based_acc, max_based_acc




def test_thresholds(dsets_preds, max_threshold, mean_threshold=0):
    me_t = mean_threshold #0.35
    ma_t = max_threshold #0.75 
    means = []
    maxes = []
    for preds, label, dset in dsets_preds:
        # print('_______')
        print(dset)
        res = predictions_to_read_predictions(preds)
        mean_based, max_based = get_metrics(res, label=label, mean_threshold=me_t, max_threshold=ma_t)
        means.append(mean_based)
        maxes.append(max_based)
    if(mean_threshold!=0):
        print('MEAN BASED ACC', np.mean(means))
    print('MAX BASED ACC', np.mean(maxes))
    print('-------------')
    
    

def get_decision_points(dsets_preds, max_threshold, title=''):
    correct_decisions = []
    wrong_decisions = []
    
    correct_all = []
    wrong_all = []
    for preds, label, dset in dsets_preds:
        res = predictions_to_read_predictions(preds)
        for k,v in res.items():
            #TODO put all >max threshold there, not only maximal
            v = v.flatten()
            predicted_label = np.max(v) > max_threshold
            decision_index = np.argmax(v)/len(v)
            decision_indicies = list(np.nonzero(v > max_threshold)[0]/len(v))
            if(predicted_label ==1): #Only predicted positives
                if(predicted_label == label):
                    correct_decisions.append(decision_index)
                    correct_all += decision_indicies
                else:
                    wrong_decisions.append(decision_index)
                    wrong_all += decision_indicies
                    # print(decision_index)
                    # print(decision_indicies)
                    # print(v)
                    # print(v > max_threshold)
                    # print('___')
    plt.hist([correct_decisions, wrong_decisions], color=['green','red'], stacked=False)
    plt.title(f'{title}\nPredicted positives - decision areas \n Strongest')
    plt.show()
    
    plt.hist([correct_all, wrong_all], color=['green','red'], stacked=False)
    plt.title('Sufficient')
    
    plt.show()



def plot_roc(dsets_preds, pooling='max', title=''):
    predictions = []
    labels = []
    exps = []
    for log in dsets_preds:
        preds, label, exp = log
        exps.append(exp)
        preds = predictions_to_read_predictions(preds)
        mod_probs = []
        for k,v in preds.items():
            if(pooling == 'max'):
                mod_probability = np.max(v)
            elif(pooling == 'mean'):
                mod_probability = np.mean(v)
            else:
                raise Exception('pooling unspecified')
            mod_probs.append(mod_probability)
        predictions = predictions + mod_probs
        labels = labels + [label]*len(mod_probs)

    # print(predictions)
    # print(labels)

    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions)
    cutoff_1 = thresholds[np.argmax(tpr-fpr)]
    cutoff_1_tpr = tpr[np.argmax(tpr-fpr)]
    
    cutoff_2 = thresholds[np.argmin((1-tpr) ** 2 + fpr ** 2)]
    cutoff_2_tpr = tpr[np.argmin((1-tpr) ** 2 + fpr ** 2)]
    
    try:
        auc = metrics.roc_auc_score(labels, predictions)
    except ValueError:
        print('AUC not defined')
        auc=0
    
    exps = str(exps[:len(exps)//2])+'\n'+str(exps[len(exps)//2:]) #For nice legend printing
    plt.plot(fpr, tpr, label = f'{exps} \n AUC %.3f CUTOFFS {str(cutoff_1)[:4]} (tpr {str(cutoff_1_tpr)[:4]}) or {str(cutoff_2)[:4]} (tpr {str(cutoff_2_tpr)[:4]})' % auc)
    plt.title(f'{title} {pooling}')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc='center left', bbox_to_anchor=(1,0.5), prop={'size':10})

def plot_covid_roc(dsets_preds, title, pooling='max'):
    plot_roc(filter_dset_preds(dsets_preds, ['_0_covid','5_covid']), pooling='max', title=title)
    plot_roc(filter_dset_preds(dsets_preds, ['_0_covid','10_covid']), pooling='max', title=title)
    plot_roc(filter_dset_preds(dsets_preds, ['_0_covid','33_covid']), pooling='max', title=title)
    plt.show()

def plot_roc_curves(dsets_preds, title, pooling='max'):
    #Balance data?
    plot_roc(dsets_preds, pooling=pooling)
    plot_roc(filter_dset_preds(dsets_preds, ['covid']), pooling=pooling, title=title)
    plot_roc(filter_dset_preds(dsets_preds, ['novoa']), pooling=pooling, title=title)
    plt.show()

