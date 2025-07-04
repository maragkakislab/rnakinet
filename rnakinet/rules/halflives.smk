rule aggregate_predictions:
    input:
        transcriptome_bam=lambda wildcards: f"outputs/alignment/{wildcards.experiment_name}/{basecalling_config['dorado_version']}/{basecalling_config['basecalling_model']}/reads-align.transcriptome.sorted.bam",
        transcript_to_gene_table = lambda wildcards: f'references/{exp_to_ensembl_species[wildcards.experiment_name]}/transcript-gene-ids.tab',
        predictions='outputs/predictions/{model_name}/{experiment_name}/preds.csv',
    output:
        gene_out = 'outputs/predictions/{model_name}/{experiment_name}/gene_level_predictions.tsv',
        transcript_out = 'outputs/predictions/{model_name}/{experiment_name}/transcript_level_predictions.tsv',
    conda:
        "../envs/gene_aggregation.yaml"
    params:
        threshold=lambda wildcards: model_inference_params[wildcards.model_name]['threshold'],
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
        gene_predictions = 'outputs/predictions/{model_name}/{experiment_name}/{reference_level}_level_predictions.tsv',
    output:
        'outputs/predictions/{model_name}/{experiment_name}/{reference_level}_level_halflives_predictions.tsv',
    conda:
        "../envs/visual.yaml"
    params:
        tl = lambda wildcards: exp_to_tl[wildcards.experiment_name],
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
        gene_predictions = ['outputs/predictions/{model_name}/{experiment_name}/{reference_level}_level_predictions.tsv'],
        gene_halflives = lambda wildcards: [halflives_name_to_file[wildcards.halflives_name]],
    output:
        'outputs/visual/predictions/{model_name}/{experiment_name}/decay/{halflives_name}_halflives_{reference_level}_read_limit_decay_plot.pdf'
    conda:
        "../envs/visual.yaml"
    params:
        tl = lambda wildcards: [exp_to_tl[wildcards.experiment_name]],
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