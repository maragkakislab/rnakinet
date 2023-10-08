from pathlib import Path

base_path = Path('/home/jovyan/RNAModif/rnamodif/workflow/outputs/splits')
name_to_path = {
    'nia_2022_pos':base_path/'20220303_hsa_dRNA_HeLa_5EU_polyA_REL5_2',
    'nia_2022_neg':base_path/'20220520_hsa_dRNA_HeLa_DMSO_1',
    
    'nia_2022_pos':base_path/'20220303_hsa_dRNA_HeLa_5EU_polyA_REL5_2',
    'nia_2022_neg':base_path/'20220520_hsa_dRNA_HeLa_DMSO_1',

    'nia_2020_pos':base_path/'20201016_hsa_dRNASeq_HeLa_5EU_polyA_REL5_short_1',
    'nia_2020_neg':base_path/'20201016_hsa_dRNASeq_HeLa_dmso_polyA_REL5_short_1',

    'nano_pos_1':base_path/'20180514_1054_K562_5EU_1440_labeled_run',
    'nano_pos_2':base_path/'20180514_1541_K562_5EU_1440_labeled_II_run',
    'nano_pos_3':base_path/'20180516_1108_K562_5EU_1440_labeled_III_run',

    'nano_neg_1':base_path/'20180327_1102_K562_5EU_0_unlabeled_run',
    'nano_neg_2':base_path/'20180403_1102_K562_5EU_0_unlabeled_II_run',
    'nano_neg_3':base_path/'20180403_1208_K562_5EU_0_unlabeled_III_run',
    
    'nia_neuron_hek':base_path/'20201001_hsa_dRNA_Hek293T_NoArs_5P_1',
    'nia_neuron_ctrl':base_path/'20201022_hsa_dRNA_Neuron_ctrl_5P_1',
    'nia_neuron_tdp':base_path/'20201022_hsa_dRNA_Neuron_TDP_5P_1',
    
    'nia_2022_pos_march':base_path/'20220303_hsa_dRNA_HeLa_5EU_polyA_REL5_2',
    'nia_2022_pos_may':base_path/'20220520_hsa_dRNA_HeLa_5EU_200_1',
    
    'nia_2022_neg_march':base_path/'20220303_hsa_dRNA_HeLa_DMSO_polyA_REL5_2',
    'nia_2022_neg_may':base_path/'20220520_hsa_dRNA_HeLa_DMSO_1',
    
}

# Experiments that do not require split
name_to_path_extras = {
    'm6A_0': '/home/jovyan/RNAModif/rnamodif/workflow/local_store/store/seq/ont/experiments/m6A_0/fast5',
    '2-OmeATP_0': '/home/jovyan/RNAModif/rnamodif/workflow/local_store/store/seq/ont/experiments/2-OmeATP_0/fast5',
    'ac4C_0': '/home/jovyan/RNAModif/rnamodif/workflow/local_store/store/seq/ont/experiments/ac4C_0/fast5',
    'm5C_0': '/home/jovyan/RNAModif/rnamodif/workflow/local_store/store/seq/ont/experiments/m5C_0/fast5',
    'remdesivir_0': '/home/jovyan/RNAModif/rnamodif/workflow/local_store/store/seq/ont/experiments/remdesivir_0/fast5',
    's4U_0': '/home/jovyan/RNAModif/rnamodif/workflow/local_store/store/seq/ont/experiments/s4U_0/fast5',
}

name_to_expname = {name: path.stem for name, path in name_to_path.items()}
# for name, path in name_to_path_extras.items():
#     name_to_expname[name] = Path(path).parent.stem
    


name_to_files = {}
for name,path in name_to_path.items():
    train_files = list((path/'train').rglob('*.fast5'))
    test_files = list((path/'test').rglob('*.fast5'))
    valid_files = list((path/'validation').rglob('*.fast5'))
    
    name_to_files[name] = {'train':train_files, 'test':test_files, 'validation':valid_files}
    
    
for name,path in name_to_path_extras.items():
    train_files = list(Path(path).rglob('*.fast5'))
    
    name_to_files[name] = {'train':train_files}
    
