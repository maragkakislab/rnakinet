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

#TODO change model so i dont have to run inference to test
rule create_auroc_plots:
    input:
        neg_files = lambda wildcards: expand(
            'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling.pickle', 
            experiment_name=[neg_exp for neg_exp, pos_exp in config['AUROC_NEG_POS_PAIRS']],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
            prediction_type=wildcards.prediction_type,
        ),
        pos_files = lambda wildcards: expand(
            'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling.pickle', 
            experiment_name=[pos_exp for neg_exp, pos_exp in config['AUROC_NEG_POS_PAIRS']],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
            prediction_type=wildcards.prediction_type,
        ),
        
    output:
        'outputs/visual/{prediction_type}/{model_name}/{plot_name}_{pooling}_pooling_auroc.pdf'
    conda:
        '../envs/visual.yaml'
    params:
        neg_experiments = [neg_exp for neg_exp, pos_exp in config['AUROC_NEG_POS_PAIRS']],
        pos_experiments = [pos_exp for neg_exp, pos_exp in config['AUROC_NEG_POS_PAIRS']],
                                
    shell:
        """
        python3 scripts/auroc.py \
            --positives-in-order {input.pos_files} \
            --negatives-in-order {input.neg_files} \
            --positives-names-in-order {params.pos_experiments} \
            --negatives-names-in-order {params.neg_experiments} \
            --output {output} \
            --model-name {wildcards.model_name} \
        """
    