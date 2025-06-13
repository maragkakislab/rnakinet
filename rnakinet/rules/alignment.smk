rule align_to_genome:
    input:
        basecalls = "outputs/basecalling/{experiment_name}/{dorado_version}/{basecalling_model}/all_reads.fastq",
        dorado_location = lambda wildcards: f'{wildcards.dorado_version}/bin/dorado',
        reference_path = lambda wildcards: exp_to_reference[wildcards.experiment_name],
    output:
        bam = "outputs/alignment/{experiment_name}/{dorado_version}/{basecalling_model}/reads-align.genome.sorted.bam",
        bai = "outputs/alignment/{experiment_name}/{dorado_version}/{basecalling_model}/reads-align.genome.sorted.bam.bai"
    conda:
        "../envs/alignment.yaml"
    params:
        dorado_location = dorado_location,
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