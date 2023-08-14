def get_halflives(experiment_name):
    transcriptome = get_transcriptome_version(experiment_name)
    transcriptome_to_file = {
        'Mus_musculus.GRCm39.cdna.all': 'mouse_halflives.txt',
        'Homo_sapiens.GRCh38.cdna.all': 'tani_halflives.txt',
    }
    print('using', transcriptome_to_file[transcriptome], 'map file')
    return transcriptome_to_file[transcriptome]

rule create_distribution_plot:
    input:
        files = lambda wildcards: expand(
            'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling.pickle', 
            experiment_name=comparisons[wildcards.experiment_group_name]['negatives']+comparisons[wildcards.experiment_group_name]['positives'],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
            prediction_type=wildcards.prediction_type,
        ),
    output:
        'outputs/visual/{prediction_type}/{model_name}/{experiment_group_name}_{pooling}_pooling_{plot_type}.pdf'
    wildcard_constraints:
        plot_type='(boxplot|violin)'
    conda:
        '../envs/visual.yaml'
    params:
        exp_names = lambda wildcards: comparisons[wildcards.experiment_group_name]['negatives']+comparisons[wildcards.experiment_group_name]['positives']
    shell:
        """
        python3 scripts/{wildcards.plot_type}.py \
            --files {input.files} \
            --output {output} \
            --model-name {wildcards.model_name} \
            --exp-names {params.exp_names} \
        """

rule create_classification_plot:
    input:
        neg_files = lambda wildcards: expand(
            'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling.pickle', 
            experiment_name=[exp for group, subdict in comparisons.items() for exp in subdict['negatives']],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
            prediction_type=wildcards.prediction_type,
        ),
        pos_files = lambda wildcards: expand(
            'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling.pickle', 
            experiment_name=[exp for group, subdict in comparisons.items() for exp in subdict['positives']],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
            prediction_type=wildcards.prediction_type,
        ),
    output:
        'outputs/visual/{prediction_type}/{model_name}/{pooling}_pooling_{plot_type}.pdf'
    wildcard_constraints:
        plot_type='(auroc|thresholds|pr_curve)'
    conda:
        '../envs/visual.yaml'
    params:
        neg_experiments = [exp for group, subdict in comparisons.items() for exp in subdict['negatives']],
        pos_experiments = [exp for group, subdict in comparisons.items() for exp in subdict['positives']],
        neg_group_names = [group for group, subdict in comparisons.items() for _ in subdict['negatives']],  
        pos_group_names = [group for group, subdict in comparisons.items() for _ in subdict['positives']],       
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
            --model-name {wildcards.model_name} \
        """

#TODO why is there plot for other chromosomes if TEST should only have some???? CHECK!  
rule create_chromosome_plot:
    input:
        neg_files = lambda wildcards: expand(
            'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling.pickle', 
            experiment_name=[exp for group, subdict in comparisons.items() for exp in subdict['negatives']],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
            prediction_type=wildcards.prediction_type,
        ),
        pos_files = lambda wildcards: expand(
            'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling.pickle', 
            experiment_name=[exp for group, subdict in comparisons.items() for exp in subdict['positives']],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
            prediction_type=wildcards.prediction_type,
        ),
        neg_bams = lambda wildcards: expand('outputs/alignment/{experiment_name}/reads-align.genome.sorted.bam', experiment_name=[exp for group, subdict in comparisons.items() for exp in subdict['negatives']]),
        pos_bams = lambda wildcards: expand('outputs/alignment/{experiment_name}/reads-align.genome.sorted.bam', experiment_name=[exp for group, subdict in comparisons.items() for exp in subdict['positives']]),
    output:
        'outputs/visual/{prediction_type}/{model_name}/{pooling}_pooling_chr_acc.pdf'
    conda:
        '../envs/visual.yaml'
    params:
        threshold=lambda wildcards: MODELS[wildcards.model_name]['threshold'],
        neg_group_names = [group for group, subdict in comparisons.items() for _ in subdict['negatives']],  
        pos_group_names = [group for group, subdict in comparisons.items() for _ in subdict['positives']], 
    shell:
        """
        python3 scripts/chr_acc.py \
            --positives_bams {input.pos_bams} \
            --negatives_bams {input.neg_bams} \
            --positives_predictions {input.pos_files} \
            --negatives_predictions {input.neg_files} \
            --negatives-groups-in-order {params.neg_group_names} \
            --positives-groups-in-order {params.pos_group_names} \
            --threshold {params.threshold} \
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
        '_0060m_':2.0,
        '_0030m_':1.5,
        '_ctrl_':1.0,
        '20230706_mmu_dRNA_3T3_5EU_400_1':2.0, 
    }
    for k,v in pattern_to_time.items():
        if(k in experiment_name):
            return v
    raise Exception('Uknown decay time constant')
    
rule create_decay_plot:
    input:
        gene_predictions = 'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling_gene_level_predictions.tsv',
        gene_halflifes = lambda wildcards: get_halflives(wildcards.experiment_name),
    output:
        'outputs/visual/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling_decay_plot.pdf'
    conda:
        "../envs/visual.yaml"
    params:
        tl = lambda wildcards: get_time_from_expname(wildcards.experiment_name)
    #TODO add parameter for column (average_score vs percentage_modified[dependent on my threshold])
    shell:
        """
        python3 scripts/decay_plot.py \
            --gene-predictions {input.gene_predictions} \
            --gene-halflifes {input.gene_halflifes} \
            --tl {params.tl} \
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
        
        

rule create_all_plots:
    input:
        expand("outputs/visual/diff_exp/{time}/{column}.pdf",
            time=time_data.keys(),
            column=['padj','pvalue'],
        ),
        expand('outputs/visual/{prediction_type}/{model_name}/{pooling}_pooling_chr_acc.pdf',
            prediction_type=prediction_type, 
            model_name = model_name, 
            pooling=pooling,
            # experiment_group_name=comparisons.keys(),
        ),
        expand('outputs/visual/{prediction_type}/{model_name}/{pooling}_pooling_{plot_type}.pdf',
            prediction_type=prediction_type, 
            model_name = model_name, 
            pooling=pooling,
            plot_type=['auroc', 'thresholds','pr_curve'],
        ),
        expand('outputs/visual/{prediction_type}/{model_name}/{experiment_group_name}_{pooling}_pooling_{plot_type}.pdf',
            prediction_type=prediction_type,
            model_name=model_name,
            pooling=pooling,
            experiment_group_name=comparisons.keys(),
            plot_type=['boxplot', 'violin'],    
        ),
        expand('outputs/visual/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling_{plot_type}.pdf',
            prediction_type=prediction_type,
            model_name=model_name,
            pooling=pooling,
            experiment_name=individual_exps,
            plot_type=['decay_plot'],    
        ),
        expand('outputs/visual/{prediction_type}/{model_name}/{time}/{pooling}_pooling_fc_{pred_col}.pdf',
            prediction_type=prediction_type,
            model_name=model_name,
            pooling=pooling,
            time=time_data.keys(),
            pred_col=['average_score','percentage_modified'],
        ),
    output:
        'outputs/visual/{prediction_type}/{model_name}/{pooling}_ALL_DONE.txt'
    shell:
        """
        touch {output}
        """
        
        
        
        