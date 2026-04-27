rule aggregate_predictions:
    input:
        transcriptome_bam=lambda wildcards: f"{OUTPUTS_DIR}/alignment/{wildcards.experiment_name}/{BASECALLING_CONFIG['dorado_version']}/{BASECALLING_CONFIG['basecalling_model']}/reads-align.transcriptome.sorted.bam",
        transcript_to_gene_table = lambda wildcards: f'{REFERENCES_DIR}/{EXP_TO_ENSEMBL_SPECIES[wildcards.experiment_name]}/transcript-gene-ids.tab',
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

rule calculate_decay:
    input:
        gene_predictions = OUTPUTS_DIR + '/predictions/{model_name}/{experiment_name}/{reference_level}_level_predictions.tsv',
    output:
        OUTPUTS_DIR + '/predictions/{model_name}/{experiment_name}/{reference_level}_level_halflives_predictions.tsv',
    conda:
        "../envs/visual.yaml"
    params:
        tl = lambda wildcards: EXP_TO_TL[wildcards.experiment_name],
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
        tl = lambda wildcards: [EXP_TO_TL[wildcards.experiment_name]],
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