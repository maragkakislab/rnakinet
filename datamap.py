from pathlib import Path

pth = Path('../../meta/martinekv/store/seq/ont/experiments')
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
