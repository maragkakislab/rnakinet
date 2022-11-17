import optuna
import itertools
from rnamodif.data_utilsdatamap import experiment_files

optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective_factory(exp1, exp2, aggregate_files=False):
    def objective(trial):
        threshold = trial.suggest_int('x', 0, 1000000)
        pos_sub = df[df['exp'].isin([exp1])]
        neg_sub = df[df['exp'].isin([exp2])]
        
        if(aggregate_files):
            pos_lengths = pos_sub.groupby('file').agg('mean')['len'].values
            neg_lengths = neg_sub.groupby('file').agg('mean')['len'].values
        else:
            pos_lengths = pos_sub['len'].values
            neg_lengths = neg_sub['len'].values
        
        #TODO make possible that threshold goes both ways, not assuming pos<neg
        pos_perc = sum(pos_lengths < threshold)/len(pos_lengths)
        neg_perc = sum(neg_lengths > threshold)/len(neg_lengths)
        return (pos_perc+neg_perc)/2
    return objective

def run_threshold(exp1, exp2, trials=250, agg_files=False):
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_factory(exp1, exp2, agg_files), n_trials=trials)

    trial = study.best_trial
    # print("  Value: ", trial.value)
    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print("    {}: {}".format(key, value))
    return trial.value



def get_length_based_acc(agg_files):
    experiments = experiment_files.keys()
    combs = list(itertools.combinations(experiments, 2))
    results = []
    for exp1, exp2 in combs:
        acc = run_threshold(exp1, exp2, agg_files=agg_files)
        results.append([exp1, exp2, acc])
    return results