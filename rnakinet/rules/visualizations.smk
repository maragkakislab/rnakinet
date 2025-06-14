rule create_classification_plot:
    input:
        pos_files = lambda wildcards: expand(
            'outputs/predictions/{model_name}/{inference_run}/preds.csv', 
            inference_run=inference_run_groups[wildcards.group]['positives'],
            model_name=wildcards.model_name,
        ),
        neg_files = lambda wildcards: expand(
            'outputs/predictions/{model_name}/{inference_run}/preds.csv',
            inference_run=inference_run_groups[wildcards.group]['negatives'],
            model_name=wildcards.model_name,
        ),
    output:
        'outputs/visual/predictions/{model_name}/{group}_{plot_type}.pdf'
    wildcard_constraints:
        plot_type='(auroc|thresholds|pr_curve|pr_ratios)'
    conda:
        '../envs/visual.yaml'
    params:
        # This can be used to plot multiple lines in one plot, by specifying groups in-order of both positives and negatives
        pos_group_names = lambda wildcards: wildcards.group,
        neg_group_names = lambda wildcards: wildcards.group,
        chosen_threshold = lambda wildcards: model_inference_params[wildcards.model_name]['threshold'],
    shell:
        """
        python3 scripts/{wildcards.plot_type}.py \
            --positives-in-order {input.pos_files} \
            --negatives-in-order {input.neg_files} \
            --negatives-groups-in-order {params.neg_group_names} \
            --positives-groups-in-order {params.pos_group_names} \
            --output {output} \
            --chosen_threshold {params.chosen_threshold} \
            --model-name {wildcards.model_name} \
        """
        