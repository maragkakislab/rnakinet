from pathlib import Path

name_to_path = {
    'nia_2022_pos':Path('/home/jovyan/preprocessing_RNAModif/splits/20220303_hsa_dRNA_HeLa_5EU_polyA_REL5_2'),
    'nia_2022_neg':Path('/home/jovyan/preprocessing_RNAModif/splits/20220520_hsa_dRNA_HeLa_DMSO_1'),
    
    'nia_2022_pos':Path('/home/jovyan/preprocessing_RNAModif/splits/20220303_hsa_dRNA_HeLa_5EU_polyA_REL5_2'),
    'nia_2022_neg':Path('/home/jovyan/preprocessing_RNAModif/splits/20220520_hsa_dRNA_HeLa_DMSO_1'),

    'nia_2020_pos':Path('/home/jovyan/preprocessing_RNAModif/splits/20201016_hsa_dRNASeq_HeLa_5EU_polyA_REL5_short_1'),
    'nia_2020_neg':Path('/home/jovyan/preprocessing_RNAModif/splits/20201016_hsa_dRNASeq_HeLa_dmso_polyA_REL5_short_1'),

    'nano_pos_1':Path('/home/jovyan/preprocessing_RNAModif/splits/20180514_1054_K562_5EU_1440_labeled_run'),
    'nano_pos_2':Path('/home/jovyan/preprocessing_RNAModif/splits/20180514_1541_K562_5EU_1440_labeled_II_run'),
    'nano_pos_3':Path('/home/jovyan/preprocessing_RNAModif/splits/20180516_1108_K562_5EU_1440_labeled_III_run'),

    'nano_neg_1':Path('/home/jovyan/preprocessing_RNAModif/splits/20180327_1102_K562_5EU_0_unlabeled_run'),
    'nano_neg_2':Path('/home/jovyan/preprocessing_RNAModif/splits/20180403_1102_K562_5EU_0_unlabeled_II_run'),
    'nano_neg_3':Path('/home/jovyan/preprocessing_RNAModif/splits/20180403_1208_K562_5EU_0_unlabeled_III_run'),
}
name_to_files = {}
for name,path in name_to_path.items():
    train_files = list((path/'train').rglob('*fast5'))
    test_files = list((path/'test').rglob('*.fast5'))
    
    name_to_files[name] = {'train':train_files, 'test':test_files}
    
    
    