rule create_violin_plots:
    input:
        files = lambda wildcards: expand(
            'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling.pickle', 
            experiment_name=config['EXPERIMENTS_TO_PROCESS'],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
            prediction_type=wildcards.prediction_type,
        ),
    output:
        'outputs/visual/{prediction_type}/{model_name}/{plot_name}_{pooling}_pooling_violin.pdf'
    conda:
        '../envs/visual.yaml'
    params:
        exp_names = config['EXPERIMENTS_TO_PROCESS']
    shell:
        """
        python3 scripts/violin.py \
            --files {input.files} \
            --output {output} \
            --model-name {wildcards.model_name} \
            --exp-names {params.exp_names} \
        """
        
rule create_boxplots:
    input:
        files = lambda wildcards: expand(
            'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling.pickle', 
            experiment_name=config['EXPERIMENTS_TO_PROCESS'],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
            prediction_type=wildcards.prediction_type,
        ),
    output:
        'outputs/visual/{prediction_type}/{model_name}/{plot_name}_{pooling}_pooling_boxplot.pdf'
    conda:
        '../envs/visual.yaml'
    params:
        exp_names = config['EXPERIMENTS_TO_PROCESS']
    shell:
        """
        python3 scripts/boxplot.py \
            --files {input.files} \
            --output {output} \
            --model-name {wildcards.model_name} \
            --exp-names {params.exp_names} \
        """

rule create_classification_plot:
    input:
        neg_files = lambda wildcards: expand(
            'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling.pickle', 
            experiment_name=[neg_exp for neg_exp, pos_exp in config['PLOT_NEG_POS_PAIRS']],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
            prediction_type=wildcards.prediction_type,
        ),
        pos_files = lambda wildcards: expand(
            'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling.pickle', 
            experiment_name=[pos_exp for neg_exp, pos_exp in config['PLOT_NEG_POS_PAIRS']],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
            prediction_type=wildcards.prediction_type,
        ),
        
    output:
        'outputs/visual/{prediction_type}/{model_name}/{plot_name}_{pooling}_pooling_{plot_type}_multi.pdf'
    conda:
        '../envs/visual.yaml'
    params:
        neg_experiments = [neg_exp for neg_exp, pos_exp in config['PLOT_NEG_POS_PAIRS']],
        pos_experiments = [pos_exp for neg_exp, pos_exp in config['PLOT_NEG_POS_PAIRS']],
                                
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
        neg_file = lambda wildcards: expand(
            'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling.pickle', 
            experiment_name=config['CHR_PLOT_NEG_POS_PAIR'][0],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
            prediction_type=wildcards.prediction_type,
        ),
        pos_file = lambda wildcards: expand(
            'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling.pickle', 
            experiment_name=config['CHR_PLOT_NEG_POS_PAIR'][1],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
            prediction_type=wildcards.prediction_type,
        ),
        neg_bam = expand('outputs/alignment/{experiment_name}/reads-align.genome.sorted.bam', experiment_name=config['CHR_PLOT_NEG_POS_PAIR'][0]),
        pos_bam = expand('outputs/alignment/{experiment_name}/reads-align.genome.sorted.bam', experiment_name=config['CHR_PLOT_NEG_POS_PAIR'][1]),
    output:
        'outputs/visual/{prediction_type}/{model_name}/{plot_name}_{pooling}_pooling_chr_acc.pdf'
    conda:
        '../envs/visual.yaml'
    shell:
        """
        python3 scripts/chr_acc.py \
            --positives_bam {input.pos_bam} \
            --negatives_bam {input.neg_bam} \
            --positives_predictions {input.pos_file} \
            --negatives_predictions {input.neg_file} \
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