feu_to_index=lambda x: int(x.stem.split('_')[-1])
covid_to_index=lambda x: int(x.stem[5:])

def get_experiment_sort(exp_name):    
    if(exp_name in ['pos_2022','pos_2020','neg_2022','neg_2020']):
        index_func = feu_to_index
    else:
        index_func = covid_to_index
    return index_func

