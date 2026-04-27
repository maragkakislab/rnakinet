rule download_fasta:
    output:
        REFERENCES_DIR + '/fasta/{reference_name}'
    params:
        uri = lambda wildcards: FASTA_TO_DOWNLOAD[wildcards.reference_name]
    shell:
        """
        wget {params.uri} \
            -P references/fasta/
        gunzip -f {REFERENCES_DIR}/fasta/{wildcards.reference_name}.gz
        """

rule align_to_genome:
    input:
        basecalls = OUTPUTS_DIR + "/basecalling/{experiment_name}/{dorado_version}/{basecalling_model}/all_reads.fastq",
        dorado_location = lambda wildcards: f'{wildcards.dorado_version}/bin/dorado',
        reference_path = lambda wildcards: f'{REFERENCES_DIR}/fasta/{EXP_TO_REFERENCE[wildcards.experiment_name]}',
    output:
        bam = OUTPUTS_DIR + "/alignment/{experiment_name}/{dorado_version}/{basecalling_model}/reads-align.genome.sorted.bam",
        bai = OUTPUTS_DIR + "/alignment/{experiment_name}/{dorado_version}/{basecalling_model}/reads-align.genome.sorted.bam.bai"
    conda:
        "../envs/alignment.yaml"
    params:
        dorado_location = lambda wildcards: f'{wildcards.dorado_version}/bin/dorado',
    threads: 16
    shell:
        """
        {params.dorado_location} aligner \
            {input.reference_path} \
            {input.basecalls} \
            --threads {threads} \
            | samtools view -b - \
		 	| samtools sort --threads {threads} \
            > {output.bam}
        samtools index {output.bam}
        """
            # --mm2-opts '-K 100M' \ 

rule align_to_transcriptome:
    input:
        basecalls = OUTPUTS_DIR + "/basecalling/{experiment_name}/{dorado_version}/{basecalling_model}/all_reads.fastq",
        dorado_location = lambda wildcards: f'{wildcards.dorado_version}/bin/dorado',
        reference_path = lambda wildcards: f'{REFERENCES_DIR}/fasta/{EXP_TO_TRANSCRIPTOME[wildcards.experiment_name]}',
    output:
        bam = OUTPUTS_DIR + "/alignment/{experiment_name}/{dorado_version}/{basecalling_model}/reads-align.transcriptome.sorted.bam",
        bai = OUTPUTS_DIR + "/alignment/{experiment_name}/{dorado_version}/{basecalling_model}/reads-align.transcriptome.sorted.bam.bai"
    conda:
        "../envs/alignment.yaml"
    params:
        dorado_location = lambda wildcards: f'{wildcards.dorado_version}/bin/dorado',
    threads: 16
    shell:
        """
        {params.dorado_location} aligner \
            {input.reference_path} \
            {input.basecalls} \
            --threads {threads} \
            | samtools view -b -F 256 \
		 	| samtools sort --threads {threads} \
            > {output.bam}
        samtools index {output.bam}
        """

rule ensembl_build_tab_file_with_transcript_and_gene_ids:
    output:
        REFERENCES_DIR + "/{species_id}/transcript-gene-ids.tab",
    params:
        link = ENSEMBL_URL + '/biomart/martservice?query=',
        xml = '<?xml version="1.0" encoding="UTF-8"?>',
        qopen = '<!DOCTYPE Query><Query  virtualSchemaName = "default" formatter = "TSV" header = "1" uniqueRows = "0" count = "" datasetConfigVersion = "0.6" completionStamp = "1" >',
        dopen = lambda w: '<Dataset name = "' + w.species_id + '_gene_ensembl" interface = "default" >',
        attr = "".join(['<Attribute name = "'+ a +'" />' for a in 
            ['ensembl_transcript_id', 'ensembl_transcript_id_version', 'ensembl_gene_id',
             'external_transcript_name', 'external_gene_name',
             'transcript_tsl']]),
        dclose = '</Dataset>',
        qclose = '</Query>'
    resources:
        mem_mb = 4*1024,
        runtime = 4*60
    shell:
        """
        wget --quiet -O {output}.tmp '{params.link}{params.xml}{params.qopen}{params.dopen}{params.attr}{params.dclose}{params.qclose}'

        tail -n 1 {output}.tmp | grep '\[success\]' || (echo 'Missing [success] stamp' && exit 1)

        head -n -1 {output}.tmp > {output}
        rm {output}.tmp
        """
