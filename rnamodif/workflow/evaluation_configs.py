#all = unmapped 
exp_groups = {
    'all_nanoid_positives':[
        '20180514_1054_K562_5EU_1440_labeled_run',
        '20180514_1541_K562_5EU_1440_labeled_II_run',
        '20180516_1108_K562_5EU_1440_labeled_III_run',
    ],
    'all_nanoid_negatives':[
        '20180327_1102_K562_5EU_0_unlabeled_run',
        '20180403_1102_K562_5EU_0_unlabeled_II_run',
        '20180403_1208_K562_5EU_0_unlabeled_III_run',
    ],
    'all_2022_nia_positives':[
        '20220303_hsa_dRNA_HeLa_5EU_polyA_REL5_2'
    ],
    'all_2022_nia_negatives':[
        '20220520_hsa_dRNA_HeLa_DMSO_1'
    ],
    'new_2022_nia_positives':[
        '20220303_hsa_dRNA_HeLa_5EU_polyA_REL5_2',
        '20220520_hsa_dRNA_HeLa_5EU_200_1',
    ],
    'new_2022_nia_negatives':[
        '20220520_hsa_dRNA_HeLa_DMSO_1',
        '20220303_hsa_dRNA_HeLa_DMSO_polyA_REL5_2',
    ],
    'train_nia_positives_old':[
        '20220303_hsa_dRNA_HeLa_5EU_polyA_REL5_2_TRAIN',
    ],
    'train_nia_negatives_old':[
        '20220520_hsa_dRNA_HeLa_DMSO_1_TRAIN',
    ],
    'test_nanoid_positives':[
        '20180514_1054_K562_5EU_1440_labeled_run_TEST',
        '20180514_1541_K562_5EU_1440_labeled_II_run_TEST',
        '20180516_1108_K562_5EU_1440_labeled_III_run_TEST',
    ],
    'test_nanoid_negatives':[
        '20180327_1102_K562_5EU_0_unlabeled_run_TEST',
        '20180403_1102_K562_5EU_0_unlabeled_II_run_TEST',
        '20180403_1208_K562_5EU_0_unlabeled_III_run_TEST',
    ],
    'test_2022_nia_positives': [
        '20220303_hsa_dRNA_HeLa_5EU_polyA_REL5_2_TEST'
    ],
    'test_2022_nia_negatives': [
        '20220520_hsa_dRNA_HeLa_DMSO_1_TEST'
    ],
    'nanoid_shock_controls':[
        '20180226_1208_K562_5EU_60_labeled_run',
        '20180227_1206_K562_5EU_60_labeled_II_run',
        '20180228_1655_K562_5EU_60_labeled_III_run',
        '20181206_1038_K562_5EU_60_labeled_IV_run',
        '20190719_1232_K562_5EU_60_labeled_V_run',
        '20190719_1430_K562_5EU_60_labeled_VI_run',
    ],
    'nanoid_shock_conditions':[
        '20180628_1020_K562_5EU_60_labeled_heat_run',
        '20180731_1020_K562_5EU_60_labeled_heat_II_run',
        '20180802_1111_K562_5EU_60_labeled_heat_III_run',
        '20190725_0809_K562_5EU_60_labeled_heat_IV_run',
        '20190725_0812_K562_5EU_60_labeled_heat_V_run',
    ],
    'stat_exps':[
        '20180514_1054_K562_5EU_1440_labeled_run',
        '20180514_1541_K562_5EU_1440_labeled_II_run',
        '20180516_1108_K562_5EU_1440_labeled_III_run',
        '20180327_1102_K562_5EU_0_unlabeled_run',
        '20180403_1102_K562_5EU_0_unlabeled_II_run',
        '20180403_1208_K562_5EU_0_unlabeled_III_run',
        '20220303_hsa_dRNA_HeLa_5EU_polyA_REL5_2',
        '20220520_hsa_dRNA_HeLa_DMSO_1',
    ],
    'hela_decay_exps':[
        '20201215_hsa_dRNA_HeLa_5EU_2hr_NoArs_0060m_5P_1',
        '20210202_hsa_dRNA_HeLa_5EU_2hr_NoArs_0060m_5P_2',
        '20210519_hsa_dRNA_HeLa_5EU_2hr_NoArs_0060m_5P_3',
    ],
    'mouse_decay_exps':[
        '20230706_mmu_dRNA_3T3_5EU_400_1',
        '20230816_mmu_dRNA_3T3_5EU_400_2',
    ],
}

#Used in violin plots, chromosome-wise plots
pos_neg_pairs = {
    'ALL_NANOID':{
        'positives':exp_groups['all_nanoid_positives'],
        'negatives':exp_groups['all_nanoid_negatives'],
    },
    'TEST_NANOID':{
        'positives':exp_groups['test_nanoid_positives'],
        'negatives':exp_groups['test_nanoid_negatives'],
    },
    'ALL_2022_NIA':{
        'positives':exp_groups['all_2022_nia_positives'],
        'negatives':exp_groups['all_2022_nia_negatives'],    
    },
    'TEST_2022_NIA':{
        'positives':exp_groups['test_2022_nia_positives'],
        'negatives':exp_groups['test_2022_nia_negatives'],
    },
    'TRAIN_2022_NIA_OLD':{
        'positives':exp_groups['train_nia_positives_old'],
        'negatives':exp_groups['train_nia_negatives_old'],
    },
}

#Used for graphs that compare multiple experiments
#Used in AUROC, AUPRC, thresholds
comparison_groups = {
    'ALL':['ALL_2022_NIA','ALL_NANOID'],
    'TEST':['TEST_2022_NIA','TEST_NANOID'],
    'TRAIN_OLD':['TRAIN_2022_NIA_OLD'],
}



# comparisons = {
#     'all_nanoid':{
#         'positives':[
#             '20180514_1054_K562_5EU_1440_labeled_run',
#             '20180514_1541_K562_5EU_1440_labeled_II_run',
#             '20180516_1108_K562_5EU_1440_labeled_III_run',
#         ],
#         'negatives':[
#             '20180327_1102_K562_5EU_0_unlabeled_run',
#             '20180403_1102_K562_5EU_0_unlabeled_II_run',
#             '20180403_1208_K562_5EU_0_unlabeled_III_run',
#         ],
#     },
#     'all_2022_nia':{
#             'positives': ['20220303_hsa_dRNA_HeLa_5EU_polyA_REL5_2'],
#             'negatives': ['20220520_hsa_dRNA_HeLa_DMSO_1'],
#         },
#     # 'test_2020_nia':{
#     #         'positives': ['20201016_hsa_dRNASeq_HeLa_5EU_polyA_REL5_short_1_TEST'],
#     #         'negatives': ['20201016_hsa_dRNASeq_HeLa_dmso_polyA_REL5_short_1_TEST'],
#     # },
# }

# test_comparisons = {
#     'test_nanoid':{
#         'positives':[
#             '20180514_1054_K562_5EU_1440_labeled_run_TEST',
#             '20180514_1541_K562_5EU_1440_labeled_II_run_TEST',
#             '20180516_1108_K562_5EU_1440_labeled_III_run_TEST',
#         ],
#         'negatives':[
#             '20180327_1102_K562_5EU_0_unlabeled_run_TEST',
#             '20180403_1102_K562_5EU_0_unlabeled_II_run_TEST',
#             '20180403_1208_K562_5EU_0_unlabeled_III_run_TEST',
#         ],
#     },
#     'test_2022_nia':{
#             'positives': ['20220303_hsa_dRNA_HeLa_5EU_polyA_REL5_2_TEST'],
#             'negatives': ['20220520_hsa_dRNA_HeLa_DMSO_1_TEST'],
#         },
#     # 'test_2020_nia':{
#     #         'positives': ['20201016_hsa_dRNASeq_HeLa_5EU_polyA_REL5_short_1_TEST'],
#     #         'negatives': ['20201016_hsa_dRNASeq_HeLa_dmso_polyA_REL5_short_1_TEST'],
#     # },
# }

#TODO add to exp_groups
time_data = {
    'NANOID_shock': {
        'controls':[
            '20180226_1208_K562_5EU_60_labeled_run',
            '20180227_1206_K562_5EU_60_labeled_II_run',
            '20180228_1655_K562_5EU_60_labeled_III_run',
            '20181206_1038_K562_5EU_60_labeled_IV_run',
            '20190719_1232_K562_5EU_60_labeled_V_run',
            '20190719_1430_K562_5EU_60_labeled_VI_run',
        ],
        'conditions':[
            '20180628_1020_K562_5EU_60_labeled_heat_run',
            '20180731_1020_K562_5EU_60_labeled_heat_II_run',
            '20180802_1111_K562_5EU_60_labeled_heat_III_run',
            '20190725_0809_K562_5EU_60_labeled_heat_IV_run',
            '20190725_0812_K562_5EU_60_labeled_heat_V_run',
        ],
    }
}

hela_decay_exps = [
    '20201215_hsa_dRNA_HeLa_5EU_2hr_NoArs_0060m_5P_1',
    '20210202_hsa_dRNA_HeLa_5EU_2hr_NoArs_0060m_5P_2',
    '20210519_hsa_dRNA_HeLa_5EU_2hr_NoArs_0060m_5P_3',
]
decay_exps = [
    
#         '20210128_hsa_dRNA_HeLa_5EU_2hr_NoArs_0030m_5P_1',
#         '20210128_hsa_dRNA_HeLa_5EU_2hr_NoArs_0030m_5P_2',
    
    
        # '20201215_hsa_dRNA_HeLa_5EU_2hr_Ars_0060m_5P_1',
        # '20210202_hsa_dRNA_HeLa_5EU_2hr_Ars_0060m_5P_2',
        # '20210519_hsa_dRNA_HeLa_5EU_2hr_Ars_0060m_5P_3',
    
        # '20210111_hsa_dRNA_HeLa_5EU_1hr_ctrl_0000m_5P_1',
        # '20210208_hsa_dRNA_HeLa_5EU_1hr_ctrl_0000m_5P_2',
        # '20210208_hsa_dRNA_HeLa_5EU_1hr_ctrl_0000m_5P_3',
        '20230706_mmu_dRNA_3T3_5EU_400_1',
        '20230816_mmu_dRNA_3T3_5EU_400_2',
        
]

# stat_exps = [
#         '20180514_1054_K562_5EU_1440_labeled_run',
#         '20180514_1541_K562_5EU_1440_labeled_II_run',
#         '20180516_1108_K562_5EU_1440_labeled_III_run',
#         '20180327_1102_K562_5EU_0_unlabeled_run',
#         '20180403_1102_K562_5EU_0_unlabeled_II_run',
#         '20180403_1208_K562_5EU_0_unlabeled_III_run',
#         '20220303_hsa_dRNA_HeLa_5EU_polyA_REL5_2',
#         '20220520_hsa_dRNA_HeLa_DMSO_1',
# ]

# prediction_type = 'predictions_limited'
prediction_type = 'predictions'

pooling = [
    # 'mean',
    'max',
]
model_name = [
    'CUSTOM_allneg_maxpool',
    'CNN_RNN_maydata_weighted',
    # 'CNN_MAX_hid100',
    # 'TEST_model',
    # 'unlimited_standard_allneg',
    # 'CNN_MAX_extraratio',
    
#     'CNN_MAX_hid30_min15k_last',
#     'CNN_MAX_hid30_min15k_best',
    # 'rodanlike_max_hid30',
    
    # 'CNN_RNN_basicneg',
    # 'CNN_MAX_hid30_min15k_longtrain_last',
    # 'CNN_RNN_allneg',
    # 'CNN_RNN_allneg_weighted',
]
