from pathlib import Path

# pth = Path('../../meta/martinekv/store/seq/ont/experiments')
pth = Path('/home/jovyan/local_store/store/seq/ont/experiments')

neg_2022 = pth/'20220520_hsa_dRNA_HeLa_DMSO_1/runs'
pos_2022 = pth/'20220303_hsa_dRNA_HeLa_5EU_polyA_REL5_2/runs'
pos_2020 = pth/'20201016_hsa_dRNASeq_HeLa_5EU_polyA_REL5_short_1/runs'
neg_2020 = pth/'20201016_hsa_dRNASeq_HeLa_dmso_polyA_REL5_short_1/runs'

experiments_dict = {
    '5eu_2022_nia':pos_2022, #old pos_2022
    'UNM_2022_nia':neg_2022, #old neg_2022
    '5eu_2020_nia':pos_2020, #old pos_2020
    'UNM_2020_nia':neg_2020, #old neg_2020
}
experiment_files = {
}
for name, exp in experiments_dict.items():
    dire = list((exp).iterdir())
    assert len(dire) == 1
    experiment_files[name]=list((dire[0]/'fast5').iterdir())

    
covid_experiments = [exp for exp in pth.iterdir() if 'hsa' not in str(exp)]
covid_experiments_dict = {exp.stem:exp for exp in covid_experiments}
covid_experiment_files = {}
for name, exp in covid_experiments_dict.items():
    dire = list((exp/'fast5').glob('./*.fast5'))
    assert len(dire) > 0
    covid_experiment_files[name+"_covid"] = dire

experiment_files.update(covid_experiment_files)



# https://www.nature.com/articles/s41587-021-00915-6#data-availability
novoa_data_path = Path('/home/jovyan/local_store/novoa_data')
m5c_novoa = novoa_data_path/'RNA010220191_m5C_fast5'
m6a_novoa_replicate_1 = novoa_data_path/'RNAAB090763_m6A_fast5'
m6a_novoa_replicate_2 = novoa_data_path/'RNA081120182_m6A_fast5'
unm_novoa_replicate_1 = novoa_data_path/'RNAAB089716_m6A_UNM_fast5'    
unm_novoa_replicate_2 = novoa_data_path/'RNA081120181_m6A_UNM_fast5'
unm_novoa_short = novoa_data_path/'RNAAB063141_00DMS_fast5'

#TODO Uncomment
novoa_experiments_dict = {
    'm5c_novoa':m5c_novoa,
    'm6a_novoa_1':m6a_novoa_replicate_1,
    'm6a_novoa_2':m6a_novoa_replicate_2,
    'UNM_novoa_1':unm_novoa_replicate_1,
    'UNM_novoa_2':unm_novoa_replicate_2,
    'UNM_novoa_short':unm_novoa_short,
}
novoa_experiment_files = {}
for name, exp in novoa_experiments_dict.items():
    files = list(exp.rglob('*.fast5'))
    assert(len(files)>0)
    novoa_experiment_files[name] = files

experiment_files.update(novoa_experiment_files)
    


#TODO download more nanoid data, refactor loading to dict                
#https://data.goettingen-research-online.de/dataset.xhtml?persistentId=doi:10.25625/XNSXV6
# https://genome.cshlp.org/content/30/9/1332.long#sec-39
fiveEU_nanoid_path = Path('/home/jovyan/local_store/nanoid/20180514_1054_K562_5EU_1440_labeled_run')
experiment_files['5eu_nanoid_24h'] = list(fiveEU_nanoid_path.rglob('*.fast5'))

fiveEU_nanoid_path_60 = Path('/home/jovyan/local_store/nanoid/20180226_1208_K562_5EU_60_labeled_run')
experiment_files['5eu_nanoid_1h'] = list(fiveEU_nanoid_path_60.rglob('*.fast5'))

fiveEU_nanoid_path_mix = Path('/home/jovyan/local_store/nanoid/20170912_1101_alternative_run')
experiment_files['5eu_nanoid_mix'] = list(fiveEU_nanoid_path_mix.rglob('*.fast5'))

fiveEU_nanoid_path_neg = Path('/home/jovyan/local_store/nanoid/20180403_1208_K562_5EU_0_unlabeled_III_run')
experiment_files['5eu_nanoid_neg'] = list(fiveEU_nanoid_path_neg.rglob('*.fast5'))

experiment_files['empty'] = []
