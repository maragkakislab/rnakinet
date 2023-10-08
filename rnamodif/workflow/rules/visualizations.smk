#TODO split to multiple files (decay, FC, classification...)

exp_to_halflife_file = {
    '20201215_hsa_dRNA_HeLa_5EU_2hr_NoArs_0060m_5P_1':'halflives_data/experiments/hl_drb_renamed.csv',
    '20210202_hsa_dRNA_HeLa_5EU_2hr_NoArs_0060m_5P_2':'halflives_data/experiments/hl_drb_renamed.csv',
    '20210519_hsa_dRNA_HeLa_5EU_2hr_NoArs_0060m_5P_3':'halflives_data/experiments/hl_drb_renamed.csv',
    'ALL_NoArs60':'halflives_data/experiments/hl_drb_renamed.csv',
    '20230706_mmu_dRNA_3T3_5EU_400_1':'halflives_data/experiments/mmu_dRNA_3T3_mion_1/features_v1.csv',
    '20230816_mmu_dRNA_3T3_5EU_400_2':'halflives_data/experiments/mmu_dRNA_3T3_mion_1/features_v1.csv',
}
def get_halflives(experiment_name):
    return exp_to_halflife_file[experiment_name]
    #TODO
    # transcriptome = get_transcriptome_version(experiment_name) #TODO import explicitly
    # transcriptome_to_file = {
    #     'Mus_musculus.GRCm39.cdna.all': 'halflives_data/experiments/mmu_dRNA_3T3_mion_1/features_v1.csv',
    #     #TODO ADD PION mouse data
    #     'Homo_sapiens.GRCh38.cdna.all': 'halflives_data/experiments/mmu_dRNA_HeLa_DRB_0h_1/features_v1.csv',
    # }
    # # print('using', transcriptome_to_file[transcriptome], 'map file')
    # return transcriptome_to_file[transcriptome]

rule create_distribution_plot:
    input:
        files = lambda wildcards: expand(
            'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling.pickle', 
            experiment_name=pos_neg_pairs[wildcards.pair_name]['negatives']+pos_neg_pairs[wildcards.pair_name]['positives'],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
            prediction_type=wildcards.prediction_type,
        ),
    output:
        'outputs/visual/{prediction_type}/{model_name}/{pair_name}_{pooling}_pooling_{plot_type}.pdf'
    wildcard_constraints:
        plot_type='(boxplot|violin)'
    conda:
        '../envs/visual.yaml'
    params:
        exp_names = lambda wildcards: pos_neg_pairs[wildcards.pair_name]['negatives']+pos_neg_pairs[wildcards.pair_name]['positives']
    shell:
        """
        python3 scripts/{wildcards.plot_type}.py \
            --files {input.files} \
            --output {output} \
            --model-name {wildcards.model_name} \
            --exp-names {params.exp_names} \
        """

#TODO extract to config
nanoid_preds_groups = {
    'nanoid':{
                'negatives':[f'local_store/arsenite/samples/K562_5EU_0_unlabeled{prefix}_run/nanoid/pred.tab.gz' for prefix in ["","_II","_III"]],
                'positives':[f'local_store/arsenite/samples/K562_5EU_1440_labeled{prefix}_run/nanoid/pred.tab.gz' for prefix in ["","_II","_III"]],
             },
}

rule create_nanoid_auroc:
    input:
        positives_paths=lambda wildcards: nanoid_preds_groups[wildcards.group_name]['positives'],
        negatives_paths=lambda wildcards: nanoid_preds_groups[wildcards.group_name]['negatives'],
    output:
        'outputs/visual/nanoid_preds/{group_name}_auroc.pdf'
    conda:
        '../envs/visual.yaml'
    shell:
        """
        python3 scripts/auroc_nanoid.py \
            --positives-paths {input.positives_paths} \
            --negatives-paths {input.negatives_paths} \
            --output {output} \
        """

#TODO test_Comparisons vs comparisons
rule create_classification_plot:
    input:
        neg_files = lambda wildcards: expand(
            'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling.pickle',
            experiment_name=[exp for pair_name in comparison_groups[wildcards.group] for exp in pos_neg_pairs[pair_name]['negatives']],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
            prediction_type=wildcards.prediction_type,),
        pos_files = lambda wildcards: expand(
            'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling.pickle', 
            experiment_name=[exp for pair_name in comparison_groups[wildcards.group] for exp in pos_neg_pairs[pair_name]['positives']],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
            prediction_type=wildcards.prediction_type,
        ),
    output:
        'outputs/visual/{prediction_type}/{model_name}/{group}_{pooling}_pooling_{plot_type}.pdf'
    wildcard_constraints:
        plot_type='(auroc|thresholds|pr_curve)'
    conda:
        '../envs/visual.yaml'
    params:
        neg_experiments = lambda wildcards: [exp for pair_name in comparison_groups[wildcards.group] for exp in pos_neg_pairs[pair_name]['negatives']],
        pos_experiments = lambda wildcards: [exp for pair_name in comparison_groups[wildcards.group] for exp in pos_neg_pairs[pair_name]['positives']],
        neg_group_names = lambda wildcards: [pair_name for pair_name in comparison_groups[wildcards.group] for _ in pos_neg_pairs[pair_name]['negatives']],
        pos_group_names = lambda wildcards: [pair_name for pair_name in comparison_groups[wildcards.group] for _ in pos_neg_pairs[pair_name]['positives']],    
        chosen_threshold = lambda wildcards: config['MODELS'][wildcards.model_name]['threshold'],
    shell:
        """
        python3 scripts/{wildcards.plot_type}.py \
            --positives-in-order {input.pos_files} \
            --negatives-in-order {input.neg_files} \
            --positives-names-in-order {params.pos_experiments} \
            --negatives-names-in-order {params.neg_experiments} \
            --negatives-groups-in-order {params.neg_group_names} \
            --positives-groups-in-order {params.pos_group_names} \
            --output {output} \
            --chosen_threshold {params.chosen_threshold} \
            --model-name {wildcards.model_name} \
        """

#TODO why is f1 score for NIA data train chrs different in test vs all
rule create_chromosome_plot:
    input:
        neg_files = lambda wildcards: expand(
            'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling.pickle', 
            experiment_name=[exp for exp in pos_neg_pairs[wildcards.pair_name]['negatives']],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
            prediction_type=wildcards.prediction_type,
        ),
        pos_files = lambda wildcards: expand(
            'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling.pickle', 
            experiment_name=[exp for exp in pos_neg_pairs[wildcards.pair_name]['positives']],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
            prediction_type=wildcards.prediction_type,
        ),
        neg_bams = lambda wildcards: expand('outputs/alignment/{experiment_name}/reads-align.genome.sorted.bam', experiment_name=[exp for exp in pos_neg_pairs[wildcards.pair_name]['negatives']]),
        pos_bams = lambda wildcards: expand('outputs/alignment/{experiment_name}/reads-align.genome.sorted.bam', experiment_name=[exp for exp in pos_neg_pairs[wildcards.pair_name]['positives']]),
    output:
        'outputs/visual/{prediction_type}/{model_name}/{pair_name}_{pooling}_pooling_chr_{plot_type}.pdf'
    conda:
        '../envs/visual.yaml'
    params:
        train_chrs = [str(i) for i in [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,'X','Y','MT']],
        valid_chrs = [str(i) for i in [20]],
        test_chrs = [str(i) for i in [1]],
        chosen_threshold = lambda wildcards: config['MODELS'][wildcards.model_name]['threshold'],
    wildcard_constraints:
        plot_type='(auroc|f1)'
    shell:
        """
        python3 scripts/chrwise_plot.py \
            --positives_bams {input.pos_bams} \
            --negatives_bams {input.neg_bams} \
            --positives_predictions {input.pos_files} \
            --negatives_predictions {input.neg_files} \
            --train_chrs {params.train_chrs} \
            --valid_chrs {params.valid_chrs} \
            --test_chrs {params.test_chrs} \
            --plot_type {wildcards.plot_type} \
            --chosen_threshold {params.chosen_threshold} \
            --output {output} \
        """
        
rule create_separated_auroc_plot:
    input:
        neg_files = lambda wildcards: expand(
            'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling.pickle', 
            experiment_name=[exp for exp in pos_neg_pairs[wildcards.pair_name]['negatives']],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
            prediction_type=wildcards.prediction_type,
        ),
        pos_files = lambda wildcards: expand(
            'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling.pickle', 
            experiment_name=[exp for exp in pos_neg_pairs[wildcards.pair_name]['positives']],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
            prediction_type=wildcards.prediction_type,
        ),
        neg_bams = lambda wildcards: expand('outputs/alignment/{experiment_name}/reads-align.genome.sorted.bam', experiment_name=[exp for exp in pos_neg_pairs[wildcards.pair_name]['negatives']]),
        pos_bams = lambda wildcards: expand('outputs/alignment/{experiment_name}/reads-align.genome.sorted.bam', experiment_name=[exp for exp in pos_neg_pairs[wildcards.pair_name]['positives']]),
    output:
        'outputs/visual/{prediction_type}/{model_name}/{pair_name}_{pooling}_pooling_{plot_type}.pdf'
    conda:
        '../envs/visual.yaml'
    params:
        chosen_threshold = lambda wildcards: config['MODELS'][wildcards.model_name]['threshold'],
    wildcard_constraints:
        plot_type='(Uperc|length)'
    shell:
        """
        python3 scripts/separated_auroc.py \
            --positives_bams {input.pos_bams} \
            --negatives_bams {input.neg_bams} \
            --positives_predictions {input.pos_files} \
            --negatives_predictions {input.neg_files} \
            --plot_type {wildcards.plot_type} \
            --chosen_threshold {params.chosen_threshold} \
            --output {output} \
        """        

rule create_volcano_plot:
    input:
        "outputs/diff_exp/{time}/DESeq_output.tab"
    output:
        "outputs/visual/diff_exp/{time}/{column}.pdf"
    conda:
        "../envs/visual.yaml"
    shell:
        """
        python3 scripts/volcano.py \
            --table-path {input} \
            --save-path {output} \
            --column {wildcards.column} \
        """

        
def get_time_from_expname(experiment_name):
    pattern_to_time = {
        '_0060m_':2.0, #TODO correct?
        '_0030m_':1.5,
        '_ctrl_':1.0,
        '20230706_mmu_dRNA_3T3_5EU_400_1':2.0, 
        '20230816_mmu_dRNA_3T3_5EU_400_2':2.0,
        'ALL_NoArs60': 1.0,
    }
    for k,v in pattern_to_time.items():
        if(k in experiment_name):
            return v
    raise Exception('Uknown decay time constant')
    
rule create_decay_plot:
    input:
        gene_predictions = 'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling_{reference_level}_level_predictions.tsv',
        gene_halflifes = lambda wildcards: get_halflives(wildcards.experiment_name), #TODO expand to allow multiple
    output:
        'outputs/visual/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling_{reference_level}_decay_plot.pdf'
    conda:
        "../envs/visual.yaml"
    params:
        tl = lambda wildcards: get_time_from_expname(wildcards.experiment_name),
        # halflives_gene_column_name = 'gene', #TODO delete, replaced by reference level - gene or transcript
    #TODO add parameter for column (average_score vs percentage_modified[dependent on my threshold])
    shell:
        """
        python3 scripts/decay_plot.py \
            --gene-predictions {input.gene_predictions} \
            --gene-halflifes {input.gene_halflifes} \
            --gene-halflifes-gene-column {wildcards.reference_level} \
            --tl {params.tl} \
            --output {output} \
        """
        


        
# rule create_decay_read_limit_plot:
#     input:
#         gene_predictions = 'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling_{reference_level}_level_predictions.tsv',
#         gene_halflifes = lambda wildcards: get_halflives(wildcards.experiment_name), #TODO expand to allow multiple
#     output:
#         'outputs/visual/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling_{reference_level}_read_limit_decay_plot.pdf'
#     conda:
#         "../envs/visual.yaml"
#     params:
#         tl = lambda wildcards: get_time_from_expname(wildcards.experiment_name),
#         # halflives_gene_column_name = 'gene', #TODO delete, replaced by reference level - gene or transcript
#     #TODO add parameter for column (average_score vs percentage_modified[dependent on my threshold])
#     shell:
#         """
#         python3 scripts/decay_read_limit_plot.py \
#             --gene-predictions {input.gene_predictions} \
#             --gene-halflifes {input.gene_halflifes} \
#             --gene-halflifes-gene-column {wildcards.reference_level} \
#             --tl {params.tl} \
#             --output {output} \
#         """


rule create_decay_read_limit_plot:
    input:
        gene_predictions_list = lambda wildcards: expand('outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling_{reference_level}_level_predictions.tsv',
                                                        experiment_name=decay_exps, #Using the mouse decay exps
                                                        model_name=wildcards.model_name,
                                                        pooling=wildcards.pooling,
                                                        reference_level=wildcards.reference_level,
                                                        prediction_type=wildcards.prediction_type,),
        gene_halflifes_list = [get_halflives(experiment_name) for experiment_name in decay_exps], #TODO expand to allow multiple halflive files?
    output:
        'outputs/visual/{prediction_type}/{model_name}/decay/{pooling}_pooling_{reference_level}_read_limit_decay_plot.pdf'
    conda:
        "../envs/visual.yaml"
    params:
        tl_list = lambda wildcards: [get_time_from_expname(experiment_name) for experiment_name in decay_exps],
        exp_name_list = decay_exps,
        # halflives_gene_column_name = 'gene', #TODO delete, replaced by reference level - gene or transcript
    #TODO add parameter for column (average_score vs percentage_modified[dependent on my threshold])
    shell:
        """
        python3 scripts/decay_read_limit_plot_multi.py \
            --gene-predictions-list {input.gene_predictions_list} \
            --gene-halflifes-list {input.gene_halflifes_list} \
            --gene-halflifes-gene-column {wildcards.reference_level} \
            --tl-list {params.tl_list} \
            --exp-name-list {params.exp_name_list} \
            --output {output} \
        """

#TODO save halflives prediction and just do a correlation plot rule
self_corr_exp_1 = '20230706_mmu_dRNA_3T3_5EU_400_1'
self_corr_exp_2 = '20230816_mmu_dRNA_3T3_5EU_400_2'
tl = 2
rule create_self_corr_decay_plot:
    input:
        gene_predictions_1 = 'outputs/{prediction_type}/{model_name}/'+self_corr_exp_1+'/{pooling}_pooling_{reference_level}_level_predictions.tsv',
        gene_predictions_2 = 'outputs/{prediction_type}/{model_name}/'+self_corr_exp_2+'/{pooling}_pooling_{reference_level}_level_predictions.tsv',
    output:
        'outputs/visual/{prediction_type}/{model_name}/self_corr_{pooling}_pooling_{reference_level}_decay_plot.pdf'
    conda:
        "../envs/visual.yaml"
    params:
        tl = tl,
    shell:
        """
        python3 scripts/self_corr_decay_plot.py \
            --gene-predictions-1 {input.gene_predictions_1} \
            --gene-predictions-2 {input.gene_predictions_2} \
            --tl {params.tl} \
            --reference-level {wildcards.reference_level} \
            --output {output} \
        """
        
rule create_fc_plot:
    input:
        gene_level_preds_control=lambda wildcards: expand(
            "outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling_gene_level_predictions.tsv", 
            experiment_name=time_data[wildcards.time]['controls'],
            prediction_type=wildcards.prediction_type,
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
        ),
        gene_level_preds_condition=lambda wildcards: expand(
            "outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling_gene_level_predictions.tsv", 
            experiment_name=time_data[wildcards.time]['conditions'],
            prediction_type=wildcards.prediction_type,
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
        ),
        deseq_output = 'outputs/diff_exp/{time}/DESeq_output.tab',
    output:
        "outputs/visual/{prediction_type}/{model_name}/{time}/{pooling}_pooling_fc_{pred_col}.pdf"
    conda:
        "../envs/visual.yaml"
    params:
        target_col = 'log2FoldChange',
        min_reads = 100,
    shell:
        #TODO try to filter low padj values - inside the script
        #TODO MY_FC calculation - only simple average is weird - see how deseq does it! Get counts and let deseq calculate it?
        #TODO deseq FC is amount - im doing a ratio - more reads means more FC but not more of ratio! rethink. Adjust by pvalue?
        """
        python3 scripts/fc_plot.py \
            --gene-level-preds-control {input.gene_level_preds_control} \
            --gene-level-preds-condition {input.gene_level_preds_condition} \
            --deseq-output {input.deseq_output} \
            --pred-col {wildcards.pred_col} \
            --target-col {params.target_col} \
            --min-reads {params.min_reads} \
            --output {output} \
        """
        
speedtest_readlimits = [10000, 100000, 500000, 1000000]
# run with --resources parallel_lock=1 to avoid paralelization and multiple runs utilizing the gpu, slowing the time
rule create_inference_speed_plot:
    input:
        jsons = lambda wildcards: expand('outputs/{prediction_type}/{model_name}/{experiment_name}/speedtest/readlimit_{reads_limit}_threads_{threads}.json',
                experiment_name=wildcards.experiment_name,
                prediction_type=wildcards.prediction_type,
                model_name=wildcards.model_name,
                reads_limit=speedtest_readlimits,
                threads=wildcards.threads,
        )
    output:
        'outputs/visual/{prediction_type}/{model_name}/{experiment_name}/speedtest_threads_{threads}.pdf'
    conda:
        "../envs/visual.yaml"
    shell:
        """
        python3 scripts/speedtest_plot.py \
            --jsons {input.jsons} \
            --output {output} \
        """ 

#TODO use only mapped datasets
datastats_groups = {
    'nanoid':[item for group in ['all_nanoid_positives','all_nanoid_negatives'] for item in exp_groups[group]],
    'nanoid_shock':[item for group in ['nanoid_shock_controls','nanoid_shock_conditions'] for item in exp_groups[group]],
    'nia':[item for group in ['new_2022_nia_positives','new_2022_nia_negatives'] for item in exp_groups[group]],
    'noars60':[item for group in ['hela_decay_exps'] for item in exp_groups[group]],
    '3t3':[item for group in ['mouse_decay_exps'] for item in exp_groups[group]],
}

rule create_datastats:
    input:
        bam_files = lambda wildcards: expand('outputs/alignment/{experiment_name}/reads-align.genome.sorted.bam',
                experiment_name=datastats_groups[wildcards.group],
        ),
    output:
        sizes_output = 'outputs/visual/datastats/{group}_sizes_stats.csv',
        lengths_output = 'outputs/visual/datastats/{group}_lengths_dist.pdf',
    conda:
        "../envs/visual.yaml"
    params:
        experiment_names = lambda wildcards: datastats_groups[wildcards.group]
    shell:
        """
        python3 scripts/datastats.py \
            --bam_files {input.bam_files} \
            --experiment_names {params.experiment_names} \
            --sizes_output {output.sizes_output} \
            --lengths_output {output.lengths_output} \
            --group {wildcards.group} \
        """ 
        
rule create_all_plots:
    input:
        expand("outputs/visual/diff_exp/{time}/{column}.pdf",
            time=time_data.keys(),
            column=['padj','pvalue'],
        ),
        expand('outputs/visual/{prediction_type}/{model_name}/{pair_name}_{pooling}_pooling_chr_{plot_type}.pdf',
            prediction_type=prediction_type, 
            model_name = model_name, 
            pooling=pooling,
            pair_name=pos_neg_pairs.keys(),
            plot_type=['auroc','f1'],
        ),
        expand('outputs/visual/{prediction_type}/{model_name}/{pair_name}_{pooling}_pooling_{plot_type}.pdf',
            pair_name=[pair_name for pair_name in pos_neg_pairs.keys()],
            prediction_type=prediction_type, 
            model_name = model_name, 
            pooling=pooling,
            plot_type=['Uperc', 'length'],
        ),
        expand('outputs/visual/{prediction_type}/{model_name}/{group}_{pooling}_pooling_{plot_type}.pdf',
            group=[group for group in comparison_groups.keys()],
            prediction_type=prediction_type, 
            model_name = model_name, 
            pooling=pooling,
            plot_type=['auroc', 'thresholds','pr_curve'],
        ),
        expand('outputs/visual/{prediction_type}/{model_name}/{pair_name}_{pooling}_pooling_{plot_type}.pdf',
            prediction_type=prediction_type,
            model_name=model_name,
            pooling=pooling,
            pair_name=[pair_name for pair_name in pos_neg_pairs.keys()],
            plot_type=['violin'],    
        ),
        # expand('outputs/visual/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling_{reference_level}_{plot_type}.pdf',
        #     prediction_type=prediction_type,
        #     model_name=model_name,
        #     pooling=pooling,
        #     experiment_name=decay_exps,
        #     plot_type=['decay_plot'],
        #     reference_level=['gene','transcript'],
        #     #TODO add support for multiple (mion pion) files?
        # ),
        expand('outputs/visual/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling_{reference_level}_{plot_type}.pdf',
            prediction_type=prediction_type,
            model_name=model_name,
            pooling=pooling,
            experiment_name=hela_decay_exps,#+['ALL_NoArs60'],
            plot_type=['decay_plot'],
            reference_level=['transcript'],
            #TODO add support for multiple (mion pion) files?
        ),
        expand('outputs/visual/{prediction_type}/{model_name}/decay/{pooling}_pooling_{reference_level}_{plot_type}.pdf',
            prediction_type=prediction_type,
            model_name=model_name,
            pooling=pooling,
            # experiment_name=decay_exps,
            plot_type=['read_limit_decay_plot'],
            reference_level=['gene','transcript'],
            #TODO add support for multiple (mion pion) files?
        ),
        expand('outputs/visual/{prediction_type}/{model_name}/{time}/{pooling}_pooling_fc_{pred_col}.pdf',
            prediction_type=prediction_type,
            model_name=model_name,
            pooling=pooling,
            time=time_data.keys(),
            pred_col=['average_score','percentage_modified'],
        ),
        expand('outputs/visual/{prediction_type}/{model_name}/self_corr_{pooling}_pooling_{reference_level}_decay_plot.pdf',
            prediction_type=prediction_type,
            model_name=model_name,
            pooling=pooling,
            reference_level=['gene','transcript'],
        ),
        # expand('outputs/visual/{prediction_type}/{model_name}/{experiment_name}/speedtest_threads_{threads}.pdf',
        #     prediction_type=prediction_type,
        #     model_name=model_name,
        #     experiment_name=['20220520_hsa_dRNA_HeLa_DMSO_1'],
        #     threads=[16,64],
        # ),
        # expand('outputs/visual/datastats/{group}_sizes_stats.csv',group=datastats_groups.keys()),
        # expand('outputs/visual/datastats/{group}_lengths_dist.pdf',group=datastats_groups.keys()),
        # expand('outputs/visual/nanoid_preds/{group_name}_auroc.pdf', group_name=['nanoid']), #TODO remove from model-specific rule
    output:
        'outputs/visual/{prediction_type}/{model_name}/{pooling}_ALL_DONE.txt'
    shell:
        """
        touch {output}
        """
        
        
        
        