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
                 gene_transcript_table=None,
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
        self.gene_transcript_table = gene_transcript_table
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
    def get_gene_transcript_table(self):
        if(self.gene_transcript_table is None):
            raise Exception(f'Gene-transcript table not defined for {self.name}')
        return self.gene_transcript_table

    
    

# transcript-gene.tab downloaded and renamed from 
# http://useast.ensembl.org/biomart/martview/989e4fff050168c3154e5398a6f27dde

mouse_genome = 'references/Mus_musculus.GRCm39.dna_sm.primary_assembly.fa'
mouse_transcriptome = 'references/Mus_musculus.GRCm39.cdna.all.fa'
mouse_gene_transcript_table = 'references/transcript-gene-mouse.tab'

human_genome = 'references/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa'
human_transcriptome = 'references/Homo_sapiens.GRCh38.cdna.all.fa'
human_gene_transcript_table = 'references/transcript-gene-human.tab'
    
experiments_data = {}
   
#TODO remove & rename
inhouse_exps = [
    '20220520_hsa_dRNA_HeLa_DMSO_1', 
    '20220520_hsa_dRNA_HeLa_5EU_200_1',
    
    '20220303_hsa_dRNA_HeLa_DMSO_polyA_REL5_2',
    '20220303_hsa_dRNA_HeLa_5EU_polyA_REL5_2',
]

for exp_name in inhouse_exps:
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
        gene_transcript_table=human_gene_transcript_table,
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
        gene_transcript_table=human_gene_transcript_table,
    )
    experiments_data[exp_name] = exp_data
    
two_hour_5eu_exps = [    
    # TODO resolve joined ALL_NoArs60 experiment for tani halflives plotting
    '20201215_hsa_dRNA_HeLa_5EU_2hr_NoArs_0060m_5P_1',
    '20210202_hsa_dRNA_HeLa_5EU_2hr_NoArs_0060m_5P_2',
    '20210519_hsa_dRNA_HeLa_5EU_2hr_NoArs_0060m_5P_3',
]

for exp_name in two_hour_5eu_exps:
    exp_data = ExperimentData(
        name=exp_name,
        path=f'local_store/arsenite/raw/{exp_name}',
        kit='SQK-RNA002',
        flowcell='FLO-MIN106',
        genome=human_genome,
        transcriptome=human_transcriptome,
        halflives_name_to_file={'DRB':'halflives_data/experiments/hl_drb_renamed.csv'}, #TODO tani halflives rename?
        time=2.0,
        gene_transcript_table=human_gene_transcript_table,
    )
    experiments_data[exp_name] = exp_data


neuron_exps = [    
    '20201001_hsa_dRNA_Hek293T_NoArs_5P_1',
    '20201022_hsa_dRNA_Neuron_ctrl_5P_1',
    '20201022_hsa_dRNA_Neuron_TDP_5P_1',
]
for exp_name in neuron_exps:
    exp_data = ExperimentData(
        name=exp_name,
        path=f'local_store/arsenite/raw/{exp_name}',
        kit='SQK-RNA002',
        flowcell='FLO-MIN106',
        genome=human_genome,
        transcriptome=human_transcriptome,
        gene_transcript_table=human_gene_transcript_table,
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
        gene_transcript_table=mouse_gene_transcript_table,
    )
    experiments_data[exp_name] = exp_data

inhouse_mouse = [
    '20230706_mmu_dRNA_3T3_5EU_400_1',
    '20230816_mmu_dRNA_3T3_5EU_400_2',
]

for exp_name in inhouse_mouse:
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
        halflives_name_to_file={
            'PION':'halflives_data/experiments/mmu_dRNA_3T3_PION_1/features_v1.csv',
            'MION':'halflives_data/experiments/mmu_dRNA_3T3_mion_1/features_v1.csv',
        },
        time=2.0,
        gene_transcript_table=mouse_gene_transcript_table,
    )
    experiments_data[exp_name] = exp_data
    
    
#TODO remove _TEST splits (now needed for plotting, but refactored code should remove this)
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
        gene_transcript_table=human_gene_transcript_table,
    )
    experiments_data[exp_name] = exp_data

#TODO remove _TRAIN splits (now needed for plotting, but refactored code should remove this)
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
        gene_transcript_table=human_gene_transcript_table,
    )
    experiments_data[exp_name] = exp_data
    

default_train_positives = [
    '20220303_hsa_dRNA_HeLa_5EU_polyA_REL5_2'
]
default_train_negatives = [
    '20220520_hsa_dRNA_HeLa_DMSO_1',
    '20201001_hsa_dRNA_Hek293T_NoArs_5P_1',
    '20201022_hsa_dRNA_Neuron_ctrl_5P_1',
    '20201022_hsa_dRNA_Neuron_TDP_5P_1',
]

#TODO delete API KEY (comet logging)
api_key = "TEVQbgxxvilM1WdTyqZLJ57ac"

training_configs  = {
    'rnakinet': {
        'training_positives_exps': default_train_positives,
        'training_negatives_exps': default_train_negatives,
        'min_len':5000,
        'max_len':400000,
        'skip':5000,
        'workers':32,
        'sampler':'ratio',
        'lr':1e-3,
        'warmup_steps':100,
        'pos_weight':1.0,
        'wd':0.01,
        'arch':'cnn_gru',
        'arch_hyperparams':{
            'cnn_depth':5,
            'mlp_hidden_size':10,
        },
        'grad_acc':64,
        'early_stopping_patience':50, 
        'comet_api_key':api_key,
        'comet_project_name':'RNAModif', #TODO rename
        'logging_step':500,
        'enable_progress_bar':'no',
        'log_to_file':True,
    },
    'rnakinet_tl': {
        'training_positives_exps': default_train_positives,
        'training_negatives_exps': default_train_negatives,
        'min_len':5000,
        'max_len':400000,
        'skip':5000,
        'workers':32,
        'sampler':'ratio',
        'lr':1e-3,
        'warmup_steps':100,
        'pos_weight':1.0,
        'wd':0.01,
        'arch':'rodan',
        'arch_hyperparams':{},
        'grad_acc':64,
        'early_stopping_patience':50, 
        'comet_api_key':api_key,
        'comet_project_name':'RNAModif', #TODO rename
        'logging_step':500,
        'enable_progress_bar':'no',
        'log_to_file':True,
    },
}