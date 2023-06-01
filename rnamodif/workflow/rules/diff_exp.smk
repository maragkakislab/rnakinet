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
        all=lambda wildcards: expand("outputs/alignment/{experiment_name}/reads.sorted.counts.txt", experiment_name=config['TIMEPOINTS'][wildcards.time])
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
    shell:
        """
        python3 scripts/DESeq2_prep.py \
            --input-path {input} \
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