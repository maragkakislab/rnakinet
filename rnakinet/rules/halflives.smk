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