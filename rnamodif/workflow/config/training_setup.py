from pathlib import Path


class ExperimentData:
    def __init__(self, 
                 name=None, 
                 path=None, 
                 kit=None, 
                 flowcell=None, 
                 basecalls_path=None,
                 genome=None,
                 transcriptome=None,
                 train_chrs=None,
                 test_chrs=None,
                 valid_chrs=None,
                 halflives_name_to_file={},
                 time=None,
                ):
        self.name = name
        self.path = path
        self.kit = kit
        self.flowcell = flowcell
        self.basecalls_path = basecalls_path
        self.genome = genome
        self.transcriptome = transcriptome
        self.train_chrs = train_chrs
        self.test_chrs = test_chrs
        self.valid_chrs = valid_chrs
        self.halflives_name_to_file = halflives_name_to_file
        # Time the cells have been exposed to 5EU, used for decay calculation
        self.time = time 
        
        if(not self.basecalls_path):
            #Default basecalls path
            self.basecalls_path = f"outputs/basecalling/{self.name}/guppy/reads.fastq.gz"
            
        self.splits_path = Path('outputs/splits')/self.name
        
    def get_name(self):
        return self.name
    def get_path(self):
        return self.path
    def get_kit(self):
        if(not self.kit):
            raise Exception(f'Kit not defined for {self.name}')
        return self.kit
    def get_flowcell(self):
        if(not self.kit):
            raise Exception(f'Flowcell not defined for {self.name}')
        return self.flowcell
    def get_basecalls(self):
        return self.basecalls_path
    def get_genome(self):
        if(not self.genome):
            raise Exception(f'Genome not defined for {self.name}')
        return self.genome
    def get_transcriptome(self):
        if(not self.transcriptome):
            raise Exception(f'Transcriptome not defined for {self.name}')
        return self.transcriptome
    def get_train_chrs(self):
        if(self.train_chrs is None):
            raise Exception(f'Train chromosomes not defined for {self.name}')
        return self.train_chrs
    def get_test_chrs(self):
        if(self.test_chrs is None):
            raise Exception(f'Test chromosomes not defined for {self.name}')
        return self.test_chrs
    def get_valid_chrs(self):
        if(self.valid_chrs is None):
            raise Exception(f'Validation chromosomes not defined for {self.name}')
        return self.valid_chrs
    def get_all_fast5_files(self):
        return list(self.get_path().rglob('*.fast5'))
    def get_train_fast5_files(self):
        return list((self.splits_path/'train').rglob('*.fast5'))
    def get_test_fast5_files(self):
        return list((self.splits_path/'test').rglob('*.fast5'))
    def get_valid_fast5_files(self):
        return list((self.splits_path/'validation').rglob('*.fast5'))
    def get_split_fast5_files(self, split):
        getters = {
            'all':self.get_all_fast5_files,
            'train':self.get_train_fast5_files,
            'test':self.get_test_fast5_files,
            'validation':self.get_valid_fast5_files,
        }
        return getters[split]()
    def get_halflives_name_to_file(self):
        return self.halflives_name_to_file
    def get_time(self):
        if(self.time is None):
            raise Exception(f'Time is not defined for {self.name}')
        return self.time

    
    
# TODO add these to expdata, dont call the file 'train' ? this is for eval only
# TODO how to resolve the joined experiment?
# exp_to_halflife_file = {
#     'ALL_NoArs60':'halflives_data/experiments/hl_drb_renamed.csv',
# }      
# def get_time_from_expname(experiment_name):
#     pattern_to_time = {
#         'ALL_NoArs60': 1.0,
#     }
        
mouse_genome = 'references/Mus_musculus.GRCm39.dna_sm.primary_assembly.fa'
mouse_transcriptome = 'references/Mus_musculus.GRCm39.cdna.all.fa'

human_genome = 'references/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa'
human_transcriptome = 'references/Homo_sapiens.GRCh38.cdna.all.fa'
    
experiments_data = {}
   
#TODO rename
old_exps = [
    '20220520_hsa_dRNA_HeLa_DMSO_1',
    '20220520_hsa_dRNA_HeLa_5EU_200_1',
    
    '20201016_hsa_dRNASeq_HeLa_dmso_polyA_REL5_short_1',
    '20201016_hsa_dRNASeq_HeLa_5EU_polyA_REL5_short_1',
    
    '20220303_hsa_dRNA_HeLa_DMSO_polyA_REL5_2',
    '20220303_hsa_dRNA_HeLa_5EU_polyA_REL5_2',
]

for exp_name in old_exps:
    exp_data = ExperimentData(
        name=exp_name,
        path=f'local_store/store/seq/ont/experiments/{exp_name}',
        kit='SQK-RNA002',
        flowcell='FLO-MIN106',
        genome=human_genome,
        transcriptome=human_transcriptome,
        train_chrs=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,'X','MT'],
        test_chrs=[1],
        valid_chrs=[20],
    )
    experiments_data[exp_name] = exp_data

nanoid_exps = [
    '20180327_1102_K562_5EU_0_unlabeled_run',
    '20180403_1102_K562_5EU_0_unlabeled_II_run',
    '20180403_1208_K562_5EU_0_unlabeled_III_run',
    '20180514_1054_K562_5EU_1440_labeled_run',
    '20180514_1541_K562_5EU_1440_labeled_II_run',
    '20180516_1108_K562_5EU_1440_labeled_III_run',
    
    '20180226_1208_K562_5EU_60_labeled_run',
    '20180227_1206_K562_5EU_60_labeled_II_run',
    '20180228_1655_K562_5EU_60_labeled_III_run',
    '20181206_1038_K562_5EU_60_labeled_IV_run',
    '20190719_1232_K562_5EU_60_labeled_V_run',
    '20190719_1430_K562_5EU_60_labeled_VI_run',
    
    '20180628_1020_K562_5EU_60_labeled_heat_run',
    '20180731_1020_K562_5EU_60_labeled_heat_II_run',
    '20180802_1111_K562_5EU_60_labeled_heat_III_run',
    '20190725_0809_K562_5EU_60_labeled_heat_IV_run',
    '20190725_0812_K562_5EU_60_labeled_heat_V_run',
]
for exp_name in nanoid_exps:
    exp_data = ExperimentData(
        name=exp_name,
        path=f'local_store/nanoid/{exp_name}',
        kit='SQK-RNA001',
        flowcell='FLO-MIN106',
        genome=human_genome,
        transcriptome=human_transcriptome,
        train_chrs=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,'X','MT'],
        test_chrs=[1],
        valid_chrs=[],
    )
    experiments_data[exp_name] = exp_data
    
new_exps = [    
    #TODO are these human
    #TODO remove these and make new exp batch - these dont have specified time
    '20201001_hsa_dRNA_Hek293T_NoArs_5P_1',
    '20201022_hsa_dRNA_Neuron_ctrl_5P_1',
    '20201022_hsa_dRNA_Neuron_TDP_5P_1',
    
    # '20210111_hsa_dRNA_HeLa_5EU_1hr_ctrl_0000m_5P_1',
    # '20210208_hsa_dRNA_HeLa_5EU_1hr_ctrl_0000m_5P_2',
    # '20210208_hsa_dRNA_HeLa_5EU_1hr_ctrl_0000m_5P_3',
    
    # '20210128_hsa_dRNA_HeLa_5EU_2hr_NoArs_0030m_5P_1',
    # '20210128_hsa_dRNA_HeLa_5EU_2hr_NoArs_0030m_5P_2',
    # 'noars_0030_3':,
    
    # '20210128_hsa_dRNA_HeLa_5EU_2hr_Ars_0030m_5P_1',
    # '20210128_hsa_dRNA_HeLa_5EU_2hr_Ars_0030m_5P_2',
    
    '20201215_hsa_dRNA_HeLa_5EU_2hr_NoArs_0060m_5P_1',
    '20210202_hsa_dRNA_HeLa_5EU_2hr_NoArs_0060m_5P_2',
    '20210519_hsa_dRNA_HeLa_5EU_2hr_NoArs_0060m_5P_3',
    
    # '20201215_hsa_dRNA_HeLa_5EU_2hr_Ars_0060m_5P_1',
    # '20210202_hsa_dRNA_HeLa_5EU_2hr_Ars_0060m_5P_2',
    # '20210519_hsa_dRNA_HeLa_5EU_2hr_Ars_0060m_5P_3',
    
    # '20210111_hsa_dRNA_HeLa_5EU_2hr_NoArs_0090m_5P_1',
    # '20210202_hsa_dRNA_HeLa_5EU_2hr_NoArs_0090m_5P_2',
    # '20210519_hsa_dRNA_HeLa_5EU_2hr_NoArs_0090m_5P_3',
    
    # '20201220_hsa_dRNA_HeLa_5EU_2hr_NoArs_0120m_5P_1',
    # '20210205_hsa_dRNA_HeLa_5EU_2hr_NoArs_0120m_5P_2',
    # 'noars_0120_3':,
    
    # '20210114_hsa_dRNA_HeLa_5EU_2hr_NoArs_0180m_5P_1',
    # '20210205_hsa_dRNA_HeLa_5EU_2hr_NoArs_0180m_5P_2',
    # 'noars_0180_3':,
    
    # '20201220_hsa_dRNA_HeLa_5EU_2hr_NoArs_0300m_5P_1',
    # 'noars_0300_2':,
    # 'noars_0300_3':,
]

for exp_name in new_exps:
    exp_data = ExperimentData(
        name=exp_name,
        path=f'local_store/arsenite/raw/{exp_name}',
        kit='SQK-RNA002',
        flowcell='FLO-MIN106',
        genome=human_genome, #TODO neurons are human?
        transcriptome=human_transcriptome,
        halflives_name_to_file={'DRB':'halflives_data/experiments/hl_drb_renamed.csv'}, #TODO tani? rename?
        time=2.0, #TODO correct?
    )
    experiments_data[exp_name] = exp_data

external_mouse = [
    '20211203_mmu_dRNA_3T3_mion_1',
    '20211203_mmu_dRNA_3T3_PION_1',
]
for exp_name in external_mouse:
    exp_data = ExperimentData(
        name=exp_name,
        path=f'local_store/arsenite/raw/{exp_name}',
        genome=mouse_genome,
        transcriptome=mouse_transcriptome,
    )
    experiments_data[exp_name] = exp_data

mouse = [
    '20230706_mmu_dRNA_3T3_5EU_400_1',
    '20230816_mmu_dRNA_3T3_5EU_400_2',
]

for exp_name in mouse:
    root_dir=Path(f'local_store/arsenite/raw/{exp_name}')
    fast5_pass_dirs = [x for x in root_dir.glob("**/fast5_pass") if x.is_dir()]
    assert len(fast5_pass_dirs) == 1, len(fast5_pass_dirs)
    exp_path = fast5_pass_dirs[0]
    exp_data = ExperimentData(
        name=exp_name,
        path=exp_path,
        kit='SQK-RNA002',
        flowcell='FLO-MIN106',
        genome=mouse_genome,
        transcriptome=mouse_transcriptome,
        halflives_name_to_file={'PION':'halflives_data/experiments/mmu_dRNA_3T3_mion_1/features_v1.csv'}, #TODO add PION
        time=2.0,
    )
    experiments_data[exp_name] = exp_data
    
    
extras = [
    'm6A_0',
    '2-OmeATP_0',
    'ac4C_0',
    'm5C_0',
    'remdesivir_0',
    's4U_0',
]
for exp_name in extras:
    exp_data = ExperimentData(
        name=exp_name,
        path=f'local_store/store/seq/ont/experiments/{exp_name}/fast5',
        # kit=,
        # flowcell=,
        # genome=,
        # transcriptome=,
    )
    experiments_data[exp_name] = exp_data


# # 'outputs/splits/20180327_1102_K562_5EU_0_unlabeled_run/test',
#TODO remove _TEST splits!! (now needed for plotting, but refactored code should remove this)
test_splits = [
    
    '20180327_1102_K562_5EU_0_unlabeled_run_TEST',
    '20180514_1054_K562_5EU_1440_labeled_run_TEST',
    
    '20180403_1102_K562_5EU_0_unlabeled_II_run_TEST',
    '20180514_1541_K562_5EU_1440_labeled_II_run_TEST',
    
    '20180403_1208_K562_5EU_0_unlabeled_III_run_TEST',
    '20180516_1108_K562_5EU_1440_labeled_III_run_TEST',
    
    '20220520_hsa_dRNA_HeLa_DMSO_1_TEST',
    '20220303_hsa_dRNA_HeLa_5EU_polyA_REL5_2_TEST',
    

]
for exp_name in test_splits:
    og_exp_name = exp_name[:-5]
    exp_data = ExperimentData(
        name=exp_name,
        path=f'outputs/splits/{og_exp_name}/test',
        kit='SQK-RNA001',
        flowcell='FLO-MIN106',
        # kit=,
        # flowcell=,
        genome=human_genome,
        transcriptome=human_transcriptome,
    )
    experiments_data[exp_name] = exp_data

#TODO remove _TRAIN splits!! (now needed for plotting, but refactored code should remove this)
train_splits = [
    '20180327_1102_K562_5EU_0_unlabeled_run_TRAIN',
    '20180514_1054_K562_5EU_1440_labeled_run_TRAIN',
    
    '20180403_1102_K562_5EU_0_unlabeled_II_run_TRAIN', 
    '20180514_1541_K562_5EU_1440_labeled_II_run_TRAIN', 
    
    '20180403_1208_K562_5EU_0_unlabeled_III_run_TRAIN', 
    '20180516_1108_K562_5EU_1440_labeled_III_run_TRAIN', 
    
    '20220520_hsa_dRNA_HeLa_DMSO_1_TRAIN',
    '20220303_hsa_dRNA_HeLa_5EU_polyA_REL5_2_TRAIN',
]
for exp_name in train_splits:
    og_exp_name = exp_name[:-6]
    exp_data = ExperimentData(
        name=exp_name,
        path=f'outputs/splits/{og_exp_name}/train',
        kit='SQK-RNA001',
        flowcell='FLO-MIN106',
        # kit=,
        # flowcell=,
        genome=human_genome,
        transcriptome=human_transcriptome,
    )
    experiments_data[exp_name] = exp_data
    
# # 'outputs/splits/20180327_1102_K562_5EU_0_unlabeled_run/train',
# train_splits = [
#     '20180327_1102_K562_5EU_0_unlabeled_run_TRAIN',
#     '20180514_1054_K562_5EU_1440_labeled_run_TRAIN',
    
#     '20180403_1102_K562_5EU_0_unlabeled_II_run_TRAIN',
#     '20180514_1541_K562_5EU_1440_labeled_II_run_TRAIN',
    
#     '20180403_1208_K562_5EU_0_unlabeled_III_run_TRAIN',
#     '20180516_1108_K562_5EU_1440_labeled_III_run_TRAIN',
    
#     '20220520_hsa_dRNA_HeLa_DMSO_1_TRAIN',
#     '20220303_hsa_dRNA_HeLa_5EU_polyA_REL5_2_TRAIN',
    
# ]


default_train_positives = [
    '20220303_hsa_dRNA_HeLa_5EU_polyA_REL5_2'
]
all_train_negatives = [
    '20220520_hsa_dRNA_HeLa_DMSO_1',
    '20201016_hsa_dRNASeq_HeLa_dmso_polyA_REL5_short_1',
    '20201001_hsa_dRNA_Hek293T_NoArs_5P_1',
    '20201022_hsa_dRNA_Neuron_ctrl_5P_1',
    '20201022_hsa_dRNA_Neuron_TDP_5P_1',
    'm6A_0',
    '2-OmeATP_0',
    'ac4C_0',
    'm5C_0',
    'remdesivir_0',
    's4U_0'
]
intermediate_train_negatives = [
    '20220520_hsa_dRNA_HeLa_DMSO_1',
    '20201016_hsa_dRNASeq_HeLa_dmso_polyA_REL5_short_1',
    '20201001_hsa_dRNA_Hek293T_NoArs_5P_1',
    '20201022_hsa_dRNA_Neuron_ctrl_5P_1',
    '20201022_hsa_dRNA_Neuron_TDP_5P_1',
]
basic_train_negatives = [
    '20220520_hsa_dRNA_HeLa_DMSO_1',
]
non2020_train_negatives = [
    '20220520_hsa_dRNA_HeLa_DMSO_1',
    '20201001_hsa_dRNA_Hek293T_NoArs_5P_1',
    '20201022_hsa_dRNA_Neuron_ctrl_5P_1',
    '20201022_hsa_dRNA_Neuron_TDP_5P_1',
    'm6A_0',
    '2-OmeATP_0',
    'ac4C_0',
    'm5C_0',
    'remdesivir_0',
    's4U_0'
]

train_positives_may = [
    '20220520_hsa_dRNA_HeLa_5EU_200_1'
]
train_negatives_may = [
    '20220520_hsa_dRNA_HeLa_DMSO_1'
]
train_positives_all2022 = [
    '20220520_hsa_dRNA_HeLa_5EU_200_1',
    '20220303_hsa_dRNA_HeLa_5EU_polyA_REL5_2',
]
train_negatives_all2022 = [
    '20220520_hsa_dRNA_HeLa_DMSO_1',
    '20220303_hsa_dRNA_HeLa_DMSO_polyA_REL5_2',
]

all_train_negatives_new = [
    '20220520_hsa_dRNA_HeLa_DMSO_1',
    '20220303_hsa_dRNA_HeLa_DMSO_polyA_REL5_2',
    '20201016_hsa_dRNASeq_HeLa_dmso_polyA_REL5_short_1',
    '20201001_hsa_dRNA_Hek293T_NoArs_5P_1',
    '20201022_hsa_dRNA_Neuron_ctrl_5P_1',
    '20201022_hsa_dRNA_Neuron_TDP_5P_1',
    'm6A_0',
    '2-OmeATP_0',
    'ac4C_0',
    'm5C_0',
    'remdesivir_0',
    's4U_0'
]

#TODO delete API KEY (comet logging)
api_key = "TEVQbgxxvilM1WdTyqZLJ57ac"

training_configs  = {
    #TODO dont use name_to_files strings, unify paths from snakemake config and use here
    # 'TEST_REFACTOR': {
    #     'training_positives_exps': train_positives_all2022,
    #     'training_negatives_exps': train_negatives_all2022,
    #     'min_len':5000,
    #     'max_len':400000,
    #     'skip':5000,
    #     'workers':32,
    #     'sampler':'ratio',
    #     'lr':1e-3,
    #     'warmup_steps':100,
    #     'pos_weight':1.0,
    #     'wd':0.01,
    #     'arch':'cnn_rnn',
    #     'arch_hyperparams':{
    #         'cnn_depth':5,
    #         'mlp_hidden_size':10,
    #     },
    #     'grad_acc':64,
    #     'early_stopping_patience':200, 
    #     'comet_api_key':api_key,
    #     'comet_project_name':'RNAModif',
    #     'logging_step':500, 
    #     'enable_progress_bar':'no',
    #     'log_to_file':True,
    # },
    'dummy': {
        'training_positives_exps': train_positives_all2022,
        'training_negatives_exps': train_negatives_may,
        'min_len':5000,
        'max_len':400000,
        'skip':5000,
        'workers':32,
        'sampler':'ratio',
        'lr':1e-3,
        'warmup_steps':100,
        'pos_weight':1.0,
        'wd':0.01,
        'arch':'cnn_rnn',
        'arch_hyperparams':{
            'cnn_depth':5,
            'mlp_hidden_size':10,
        },
        'grad_acc':64,
        'early_stopping_patience':200, 
        'comet_api_key':api_key,
        'comet_project_name':'RNAModif',
        'logging_step':500, 
        'enable_progress_bar':'yes',
        'log_to_file':False,
    },
}