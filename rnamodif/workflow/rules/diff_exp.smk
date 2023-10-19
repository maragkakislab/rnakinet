rule transcript_counts:
    input:
        bam_file = "outputs/alignment/{experiment_name}/reads-align.transcriptome.sorted.bam",
        map_file = lambda wildcards: experiments_data[wildcards.experiment_name].get_gene_transcript_table(),
    output:
        "outputs/alignment/{experiment_name}/reads.sorted.counts.txt"
    conda:
        "../envs/transcript_counts.yaml"
    shell:
        #TODO transcript-gene.tab export to input
        """
        python3 scripts/sam-per-ref-count.py \
            --ifile {input.bam_file} \
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
        all=lambda wildcards: expand("outputs/alignment/{experiment_name}/reads.sorted.counts.txt", experiment_name=condition_control_pairs[wildcards.time_group]['controls']+condition_control_pairs[wildcards.time_group]['conditions'])
    output:
        "outputs/diff_exp/{time_group}/reads.sorted.counts.aggregate.txt"
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
        "outputs/diff_exp/{time_group}/reads.sorted.counts.aggregate.txt"
    output:
        counts_table_path="outputs/diff_exp/{time_group}/counts_table.txt",
        metadata_path="outputs/diff_exp/{time_group}/metadata.txt",
    conda:
        "../envs/pandas_basic.yaml"
    params:
        controls=lambda wildcards: condition_control_pairs[wildcards.time_group]['controls'],
        conditions=lambda wildcards: condition_control_pairs[wildcards.time_group]['conditions'],
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
        counts_table="outputs/diff_exp/{time_group}/counts_table.txt",
        metadata="outputs/diff_exp/{time_group}/metadata.txt",
    output:
        "outputs/diff_exp/{time_group}/DESeq_output.tab"
    conda:
        "../envs/rscript.yaml"
    shell:
        r"""
        Rscript scripts/DESeq2_run.R \
            {input.counts_table} \
            {input.metadata} \
            {output} \
        """
        
        
rule generate_gene_prediction_stats:
    input:
        transcriptome_bam='outputs/alignment/{experiment_name}/reads-align.transcriptome.sorted.bam',
        # transcript_to_gene_table='transcript-gene.tab',
        transcript_to_gene_table = lambda wildcards: experiments_data[wildcards.experiment_name].get_gene_transcript_table(),
        predictions='outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling.pickle', #TODO redo to csv
    output:
        gene_out = 'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling_gene_level_predictions.tsv',
        transcript_out = 'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling_transcript_level_predictions.tsv',
        
    conda:
        "../envs/gene_aggregation.yaml"
    params:
        threshold=lambda wildcards: models_data[wildcards.model_name].get_threshold(),
    wildcard_constraints:
        prediction_type='(predictions|predictions_limited)'
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
        
rule generate_gene_prediction_stats_joined:
    input:
        transcriptome_bam='outputs/alignment/{experiment_name}/reads-align.transcriptome.sorted.bam',
        # transcript_to_gene_table='transcript-gene.tab',
        transcript_to_gene_table = lambda wildcards: experiments_data[wildcards.experiment_name].get_gene_transcript_table(),
        predictions='outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling_joined.pickle',
    output:
        gene_out = 'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling_gene_level_predictions.tsv',
        transcript_out = 'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling_transcript_level_predictions.tsv',
        
    conda:
        "../envs/gene_aggregation.yaml"
    params:
        threshold=lambda wildcards: models_data[wildcards.model_name].get_threshold(),
    wildcard_constraints:
        prediction_type='(joined_predictions)'
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

merge_bam_experiments = {
    'ALL_NoArs60':exp_groups['hela_decay_exps']
}     
rule create_joined_experiment:
    input:
        transcriptome_bam_list=lambda wildcards: expand('outputs/alignment/{experiment_name}/reads-align.transcriptome.sorted.bam', experiment_name=merge_bam_experiments[wildcards.group_name]),
    output:
        'outputs/alignment/{group_name}/reads-align.transcriptome.sorted.bam'
    conda:
        '../envs/samtools.yaml'
    shell:
        """
        samtools merge {output} {input.transcriptome_bam_list}
        """

rule create_joined_predictions:
    input:
        predictions=lambda wildcards: expand('outputs/predictions/{model_name}/{experiment_name}/{pooling}_pooling.pickle', 
            model_name=wildcards.model_name, 
            experiment_name=merge_bam_experiments[wildcards.group_name],
            pooling=wildcards.pooling)
    output:
        'outputs/joined_predictions/{model_name}/{group_name}/{pooling}_pooling_joined.pickle',
    conda:
        '../envs/pandas_basic.yaml'
    shell:
        """
        python3 scripts/merge_pickles.py \
            --pickles {input.predictions} \
            --output {output} \
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
#         'outputs/{prediction_type}/{model_name}/{time_group}/{pooling}_pooling_deseq_pred_stats.tsv'
    
        
