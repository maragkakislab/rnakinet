rule create_distribution_plot:
    input:
        files = lambda wildcards: expand(
            'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling.pickle', 
            experiment_name=timesteps[wildcards.plot_name],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
            prediction_type=wildcards.prediction_type,
        ),
    output:
        'outputs/visual/{prediction_type}/{model_name}/{plot_name}_{pooling}_pooling_{plot_type}.pdf'
    wildcard_constraints:
        plot_type='(boxplot|violin)'
    conda:
        '../envs/visual.yaml'
    params:
        exp_names = lambda wildcards: timesteps[wildcards.plot_name]
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
            experiment_name=[v['negatives'] for v in clf_tuples[wildcards.plot_name].values()],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
            prediction_type=wildcards.prediction_type,
        ),
        pos_files = lambda wildcards: expand(
            'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling.pickle', 
            # experiment_name=[pos_exp for neg_exp, pos_exp in config['PLOT_NEG_POS_PAIRS']],
            experiment_name=[v['positives'] for v in clf_tuples[wildcards.plot_name].values()],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
            prediction_type=wildcards.prediction_type,
        ),
    output:
        'outputs/visual/{prediction_type}/{model_name}/{plot_name}_{pooling}_pooling_{plot_type}.pdf'
    wildcard_constraints:
        plot_type='(auroc|thresholds|pr_curve)'
    conda:
        '../envs/visual.yaml'
    params:
        neg_experiments = lambda wildcards: [v['negatives'] for v in clf_tuples[wildcards.plot_name].values()],
        pos_experiments = lambda wildcards: [v['positives'] for v in clf_tuples[wildcards.plot_name].values()],
    shell:
        """
        python3 scripts/{wildcards.plot_type}.py \
            --positives-in-order {input.pos_files} \
            --negatives-in-order {input.neg_files} \
            --positives-names-in-order {params.pos_experiments} \
            --negatives-names-in-order {params.neg_experiments} \
            --output {output} \
            --model-name {wildcards.model_name} \
        """

        
rule create_chromosome_plots:
    input:
        neg_files = lambda wildcards: expand(
            'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling.pickle', 
            experiment_name=comparisons[wildcards.plot_name]['negatives'],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
            prediction_type=wildcards.prediction_type,
        ),
        pos_files = lambda wildcards: expand(
            'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling.pickle', 
            experiment_name=comparisons[wildcards.plot_name]['positives'],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
            prediction_type=wildcards.prediction_type,
        ),
        neg_bams = lambda wildcards: expand('outputs/alignment/{experiment_name}/reads-align.genome.sorted.bam', experiment_name=comparisons[wildcards.plot_name]['negatives']),
        pos_bams = lambda wildcards: expand('outputs/alignment/{experiment_name}/reads-align.genome.sorted.bam', experiment_name=comparisons[wildcards.plot_name]['positives']),
    output:
        'outputs/visual/{prediction_type}/{model_name}/{plot_name}_{pooling}_pooling_chr_acc.pdf'
    conda:
        '../envs/visual.yaml'
    shell:
        """
        python3 scripts/chr_acc.py \
            --positives_bams {input.pos_bams} \
            --negatives_bams {input.neg_bams} \
            --positives_predictions {input.pos_files} \
            --negatives_predictions {input.neg_files} \
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

rule create_decay_plot:
    input:
        gene_predictions = 'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling_gene_level_predictions.tsv',
        gene_halflifes = 'tani_halflives.txt'
    output:
        'outputs/visual/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling_decay_plot.pdf'
    conda:
        "../envs/visual.yaml"
    shell:
        """
        python3 scripts/decay_plot.py \
            --gene-predictions {input.gene_predictions} \
            --gene-halflifes {input.gene_halflifes} \
            --output {output} \
        """
        

rule create_all_plots:
    input:
        # "outputs/visual/diff_exp/{time}/{column}.pdf"
        expand('outputs/visual/{prediction_type}/{model_name}/{plot_name}_{pooling}_pooling_chr_acc.pdf',
            prediction_type=prediction_type, 
            model_name = model_name, 
            pooling=pooling,
            plot_name=comparisons.keys(),
        ),
        expand('outputs/visual/{prediction_type}/{model_name}/{plot_name}_{pooling}_pooling_{plot_type}.pdf',
            prediction_type=prediction_type, 
            model_name = model_name, 
            pooling=pooling,
            plot_name=clf_tuples.keys(), 
            plot_type=['auroc', 'thresholds','pr_curve'],
        ),
        expand('outputs/visual/{prediction_type}/{model_name}/{plot_name}_{pooling}_pooling_{plot_type}.pdf',
            prediction_type=prediction_type,
            model_name=model_name,
            pooling=pooling,
            plot_name=timesteps.keys(),
            plot_type=['boxplot', 'violin'],    
        ),
        expand('outputs/visual/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling_{plot_type}.pdf',
            prediction_type=prediction_type,
            model_name=model_name,
            pooling=pooling,
            experiment_name=individual_exps,
            plot_type=['decay_plot'],    
        ),
    output:
        'outputs/visual/{prediction_type}/{model_name}/{pooling}_ALL_DONE.txt'
    shell:
        """
        touch {output}
        """
        
        
        
        