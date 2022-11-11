from pathlib import Path

# pth = Path('../../meta/martinekv/store/seq/ont/experiments')
pth = Path('/home/jovyan/local_store/store/seq/ont/experiments')

neg_2022 = pth/'20220520_hsa_dRNA_HeLa_DMSO_1/runs'
pos_2022 = pth/'20220303_hsa_dRNA_HeLa_5EU_polyA_REL5_2/runs'
pos_2020 = pth/'20201016_hsa_dRNASeq_HeLa_5EU_polyA_REL5_short_1/runs'
neg_2020 = pth/'20201016_hsa_dRNASeq_HeLa_dmso_polyA_REL5_short_1/runs'


experiments_dict = {
    'pos_2022':pos_2022,
    'neg_2022':neg_2022,
    'pos_2020':pos_2020,
    'neg_2020':neg_2020,
}
experiment_files = {
}
for name, exp in experiments_dict.items():
    dire = list((exp).iterdir())
    assert len(dire) == 1
    experiment_files[name]=list((dire[0]/'fast5').iterdir())


# covid_pth = Path('../../meta/martinekv/store/seq/ont/experiments')
covid_pth = pth

covid_experiments = [exp for exp in covid_pth.iterdir() if 'hsa' not in str(exp)]
covid_experiments_dict = {exp.stem:exp for exp in covid_experiments}

covid_experiment_files = {}
for name, exp in covid_experiments_dict.items():
    dire = list((exp/'fast5').glob('./*.fast5'))
    assert len(dire) > 0
    covid_experiment_files[name] = dire

experiment_files.update(covid_experiment_files)