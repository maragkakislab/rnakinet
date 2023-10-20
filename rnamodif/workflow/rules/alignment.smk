from pathlib import Path
      
rule align_to_genome:
    input:
        basecalls = lambda wildcards: experiments_data[wildcards.experiment_name].get_basecalls(),
        reference_path = lambda wildcards: experiments_data[wildcards.experiment_name].get_genome(),
    output:
        bam = "outputs/alignment/{experiment_name}/reads-align.genome.sorted.bam",
        bai = "outputs/alignment/{experiment_name}/reads-align.genome.sorted.bam.bai"
    conda:
        "../envs/alignment.yaml"
    threads: 32
    shell:
        """
		minimap2 \
			-x splice \
			-a \
			-t {threads} \
			-u b \
			-p 1 \
			--secondary=no \
			{input.reference_path} \
			{input.basecalls} \
			| samtools view -b - \
			| samtools sort --threads {threads} \
			> {output.bam}  
		samtools index {output.bam}
		"""   
        
rule align_to_transcriptome:
    input:
        basecalls = lambda wildcards: experiments_data[wildcards.experiment_name].get_basecalls(),
        transcriptome_path = lambda wildcards: experiments_data[wildcards.experiment_name].get_transcriptome(),
    output:
        bam = "outputs/alignment/{experiment_name}/reads-align.transcriptome.sorted.bam",
        bai = "outputs/alignment/{experiment_name}/reads-align.transcriptome.sorted.bam.bai",
    conda:
        "../envs/alignment.yaml"
    threads: 32
	shell:
		"""
		minimap2 \
			-x map-ont \
			-a \
			-t {threads} \
			-u f \
			-p 1 \
			--secondary=no \
			{input.transcriptome_path} \
			{input.basecalls} \
			| samtools view -b -F 256 \
			| samtools sort --threads {threads} \
			> {output.bam}
		samtools index {output.bam} 
		"""
    
rule run_flagstat_genome:
    input:
        "outputs/alignment/{experiment_name}/reads-align.genome.sorted.bam"
    output:
        "outputs/alignment/{experiment_name}/flagstat.txt"
    conda:
        "../envs/alignment.yaml"
    shell:
        "samtools flagstat {input} > {output}"
        
rule run_flagstat_transcriptome:
    input:
        "outputs/alignment/{experiment_name}/reads-align.transcriptome.sorted.bam"
    output:
        "outputs/alignment/{experiment_name}/flagstat_transcriptome.txt"
    conda:
        "../envs/alignment.yaml"
    shell:
        "samtools flagstat {input} > {output}" 
        