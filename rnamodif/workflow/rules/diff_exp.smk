# transcript-gene.tab downloaded and renamed from 
# http://useast.ensembl.org/biomart/martview/989e4fff050168c3154e5398a6f27dde
rule transcript_counts:
    input:
        "outputs/alignment/{experiment_name}/reads-align.transcriptome.sorted.bam"
    output:
        "outputs/alignment/{experiment_name}/reads.sorted.counts.txt"
    conda:
        "../envs/transcript_counts.yaml"
    shell:
        #TODO transcript-gene.tab export to input
        """
        python3 scripts/sam-per-ref-count.py \
            --ifile {input} \
            --ref-col-name transcript \
            --cnt-col-name count \
            | table-paste-col.py \
                --table - \
                --col-name sample \
                --col-val {wildcards.experiment_name} \
            | table-join.py \
                --table1 - \
                --table2 transcript-gene.tab \
                --key1 transcript \
                --key2 'Transcript stable ID version' \
                > {output}
        """
        
        
rule aggregate_counts:
    input:
        all=lambda wildcards: expand("outputs/alignment/{experiment_name}/reads.sorted.counts.txt", experiment_name=time_data[wildcards.time]['controls']+time_data[wildcards.time]['conditions'])
    output:
        "outputs/diff_exp/{time}/reads.sorted.counts.aggregate.txt"
    conda:
        "../envs/transcript_counts.yaml"
    shell:
        """
        table-cat.py {input.all} \
          | table-group-summarize.py \
            -t - \
            -g sample 'Gene stable ID' \
            -y count \
            -f sum \
            -s '\t' \
            > {output}
        """
    
rule generate_files_for_deseq2:
    input:
        "outputs/diff_exp/{time}/reads.sorted.counts.aggregate.txt"
    output:
        counts_table_path="outputs/diff_exp/{time}/counts_table.txt",
        metadata_path="outputs/diff_exp/{time}/metadata.txt",
    conda:
        "../envs/pandas_basic.yaml"
    params:
        controls=lambda wildcards: time_data[wildcards.time]['controls'],
        conditions=lambda wildcards: time_data[wildcards.time]['conditions'],
    shell:
        """
        python3 scripts/DESeq2_prep.py \
            --input-path {input} \
            --controls {params.controls} \
            --conditions {params.conditions} \
            --output-path {output.counts_table_path} \
            --output-meta {output.metadata_path} \
        """

#TODO r argparse named arguments
rule diff_expression:
    input:
        counts_table="outputs/diff_exp/{time}/counts_table.txt",
        metadata="outputs/diff_exp/{time}/metadata.txt",
    output:
        "outputs/diff_exp/{time}/DESeq_output.tab"
    conda:
        "../envs/rscript.yaml"
    shell:
        r"""
        Rscript scripts/DESeq2_run.R \
            {input.counts_table} \
            {input.metadata} \
            {output} \
        """
        
        
MODELS = config['MODELS']
rule generate_gene_prediction_stats:
    input:
        transcriptome_bam='outputs/alignment/{experiment_name}/reads-align.transcriptome.sorted.bam',
        transcript_to_gene_table='transcript-gene.tab',
        predictions='outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling.pickle',
    output:
        'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling_gene_level_predictions.tsv'
    conda:
        "../envs/gene_aggregation.yaml"
    params:
        threshold=lambda wildcards: MODELS[wildcards.model_name]['threshold'],
    shell:
        """
        python3 scripts/gene_to_preds.py \
            --transcriptome-bam {input.transcriptome_bam} \
            --transcript-to-gene-table {input.transcript_to_gene_table}\
            --predictions {input.predictions} \
            --threshold {params.threshold} \
            --output {output}\
        """
        
# #TODO
#GOAL FOR DIFF_EXP validation
# Gene - pred_replicate_1_ctrl - p_r_2 - p_r_3 - pred_replicate_1_cond - p_r_2 - p_r_3
# Gene - (avg_pred_ctrl - avg_pred_cond)
# Link to DESEQ_output - join tables - correlation
# USE METADATA TO DECIDE HOW TO AVERAGE
# TODO AVERAGE SCORE VS AVERAGE_PERCENTAGE_MODIFIED???
# rule create_deseq_prediction_stats:
#     input:
#         expand('outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling_gene_level_predictions.tsv',
#               experiment_name=TODO)
#     output:
#         'outputs/{prediction_type}/{model_name}/{time}/{pooling}_pooling_deseq_pred_stats.tsv'
    
        