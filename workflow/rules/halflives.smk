rule aggregate_predictions:
    input:
        transcriptome_bam=lambda wildcards: f"{OUTPUTS_DIR}/alignment/{wildcards.experiment_name}/{BASECALLING_CONFIG['dorado_version']}/{BASECALLING_CONFIG['basecalling_model']}/reads-align.transcriptome.sorted.bam",
        transcript_to_gene_table = lambda wildcards: f'{REFERENCES_DIR}/{EXPERIMENTS[wildcards.experiment_name]["ensembl_species"]}/transcript-gene-ids.tab',
        predictions= OUTPUTS_DIR + '/predictions/{model_name}/{experiment_name}/preds.csv',
    output:
        gene_out = OUTPUTS_DIR + '/predictions/{model_name}/{experiment_name}/gene_level_predictions.tsv',
        transcript_out = OUTPUTS_DIR + '/predictions/{model_name}/{experiment_name}/transcript_level_predictions.tsv',
    conda:
        "../envs/gene_aggregation.yaml"
    params:
        threshold=lambda wildcards: MODEL_INFERENCE_PARAMS[wildcards.model_name]['threshold'],
    shell:
        """
        python3 scripts/gene_to_preds.py \
            --transcriptome-bam {input.transcriptome_bam} \
            --transcript-to-gene-table {input.transcript_to_gene_table}\
            --predictions {input.predictions} \
            --threshold {params.threshold} \
            --output-gene {output.gene_out} \
            --output-transcript {output.transcript_out} \
        """

rule aggregate_auroc:
    input:
        positive_predictions = lambda wildcards: expand(
            OUTPUTS_DIR + '/predictions/{model_name}/{inference_run}/preds.csv', 
            inference_run=INFERENCE_RUN_GROUPS[wildcards.group]['positives'],
            model_name=wildcards.model_name,
        ),
        negative_predictions = lambda wildcards: expand(
            OUTPUTS_DIR + '/predictions/{model_name}/{inference_run}/preds.csv', 
            inference_run=INFERENCE_RUN_GROUPS[wildcards.group]['negatives'],
            model_name=wildcards.model_name,
        ),
        # get the exp name from the group and then use that to get the correct transcriptome bam and transcript-to-gene table for the auroc calculation
        positive_transcriptome_bam = lambda wildcards: f"{OUTPUTS_DIR}/alignment/{INFERENCE_RUN_GROUPS[wildcards.group]['positives'][0]}/{BASECALLING_CONFIG['dorado_version']}/{BASECALLING_CONFIG['basecalling_model']}/reads-align.transcriptome.sorted.bam",
        negative_transcriptome_bam = lambda wildcards: f"{OUTPUTS_DIR}/alignment/{INFERENCE_RUN_GROUPS[wildcards.group]['negatives'][0]}/{BASECALLING_CONFIG['dorado_version']}/{BASECALLING_CONFIG['basecalling_model']}/reads-align.transcriptome.sorted.bam",
        # only need to pull once since pos and neg should be same species
        transcript_to_gene_table = lambda wildcards: f'{REFERENCES_DIR}/{EXPERIMENTS[INFERENCE_RUN_GROUPS[wildcards.group]["positives"][0]]["ensembl_species"]}/transcript-gene-ids.tab',
    output:
        gene_aurocs_out = OUTPUTS_DIR + '/predictions/{model_name}/aurocs/{group}/gene_level_aurocs.tsv',
        transcript_aurocs_out = OUTPUTS_DIR + '/predictions/{model_name}/aurocs/{group}/transcript_level_aurocs.tsv',
    conda:
        "../envs/gene_aggregation.yaml"
    shell:
        """
        python3 scripts/aggregate_auroc.py \
            --positive-predictions {input.positive_predictions} \
            --negative-predictions {input.negative_predictions} \
            --positive-transcriptome-bam {input.positive_transcriptome_bam} \
            --negative-transcriptome-bam {input.negative_transcriptome_bam} \
            --transcript-to-gene-table {input.transcript_to_gene_table} \
            --output-gene-aurocs {output.gene_aurocs_out} \
            --output-transcript-aurocs {output.transcript_aurocs_out} \
        """




rule calculate_decay:
    input:
        gene_predictions = OUTPUTS_DIR + '/predictions/{model_name}/{experiment_name}/{reference_level}_level_predictions.tsv',
    output:
        OUTPUTS_DIR + '/predictions/{model_name}/{experiment_name}/{reference_level}_level_halflives_predictions.tsv',
    conda:
        "../envs/visual.yaml"
    params:
        tl = lambda wildcards: EXPERIMENTS[wildcards.experiment_name]['labeling_time'],
    shell:
        """
        python3 scripts/calculate_decay.py \
            --gene-predictions {input.gene_predictions} \
            --tl {params.tl} \
            --output {output} \
        """

#wrapping in single elements in a list to allow for compatibility with the multi-plot script
#TODO only plottting halflives<5 (hardcoded inside)
rule create_decay_read_limit_plot:
    input:
        gene_predictions = [OUTPUTS_DIR + '/predictions/{model_name}/{experiment_name}/{reference_level}_level_predictions.tsv'],
        gene_halflives = lambda wildcards: [f'{HALFLIVES_FOLDER}/{HALFLIVES_NAME_TO_FILE[wildcards.halflives_name]}'],
    output:
        OUTPUTS_DIR + '/visual/predictions/{model_name}/{experiment_name}/decay/{halflives_name}_halflives_{reference_level}_read_limit_decay_plot.pdf'
    conda:
        "../envs/visual.yaml"
    params:
        tl = lambda wildcards: [EXPERIMENTS[wildcards.experiment_name]['labeling_time']],
        exp_name_list = lambda wildcards: [wildcards.experiment_name],
    shell:
        """
        python3 scripts/decay_read_limit_plot_multi.py \
            --gene-predictions-list {input.gene_predictions} \
            --gene-halflifes-list {input.gene_halflives} \
            --gene-halflifes-gene-column {wildcards.reference_level} \
            --tl-list {params.tl} \
            --exp-name-list {params.exp_name_list} \
            --output {output} \
        """