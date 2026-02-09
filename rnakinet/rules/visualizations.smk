rule create_classification_plot:
    input:
        pos_files = lambda wildcards: expand(
            OUTPUTS_DIR + '/predictions/{model_name}/{inference_run}/preds.csv', 
            inference_run=INFERENCE_RUN_GROUPS[wildcards.group]['positives'],
            model_name=wildcards.model_name,
        ),
        neg_files = lambda wildcards: expand(
            OUTPUTS_DIR + '/predictions/{model_name}/{inference_run}/preds.csv',
            inference_run=INFERENCE_RUN_GROUPS[wildcards.group]['negatives'],
            model_name=wildcards.model_name,
        ),
    output:
        OUTPUTS_DIR + '/visual/predictions/{model_name}/{group}_{plot_type}.pdf'
    wildcard_constraints:
        plot_type='(auroc|thresholds|pr_curve|pr_ratios)'
    conda:
        '../envs/visual.yaml'
    params:
        # This can be used to plot multiple lines in one plot, by specifying groups in-order of both positives and negatives
        pos_group_names = lambda wildcards: wildcards.group,
        neg_group_names = lambda wildcards: wildcards.group,
        chosen_threshold = lambda wildcards: MODEL_INFERENCE_PARAMS[wildcards.model_name]['threshold'],
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

rule plot_pct_positive:
    input:
        lambda wc: expand(
            "outputs/{pred_type}/{model}/{exp}_percent_positive.txt",
            pred_type=PCT_POS_GRAPHING_PARAMS[wc.plot_name]["predictions_type"],
            model=PCT_POS_GRAPHING_PARAMS[wc.plot_name]["models"],
            exp=PCT_POS_GRAPHING_PARAMS[wc.plot_name]["experiments"],
        ),
    output:
        plot = "outputs/visual/pct_pos/{plot_name}.png",
    conda:
        "../envs/visual.yaml",
    params:
        title = lambda wc: wc.plot_name,
        pred_type = lambda wc: PCT_POS_GRAPHING_PARAMS[wc.plot_name]["predictions_type"],
        models = lambda wc: PCT_POS_GRAPHING_PARAMS[wc.plot_name]["models"],
        experiments = lambda wc: PCT_POS_GRAPHING_PARAMS[wc.plot_name]["experiments"],
        colors = lambda wc: PCT_POS_GRAPHING_PARAMS.get("colors", []),
    shell:
        """
        python3 scripts/plot_pct_pos.py \
            --title "{params.title}" \
            --predictions-type "{params.pred_type}" \
            --models {params.models} \
            --experiments {params.experiments} \
            --colors {params.colors} \
            --out "{output.plot}"
        """