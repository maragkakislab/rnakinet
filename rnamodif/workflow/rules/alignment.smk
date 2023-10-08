from pathlib import Path
HUMAN_REF_VERSION = config['HUMAN_REF_VERSION']
HUMAN_TRANSCRIPTOME_VERSION = config['HUMAN_TRANSCRIPTOME_VERSION']
EXPERIMENT_NAME_TO_PATH = config['EXPERIMENT_NAME_TO_PATH']


#TODO require my own basecalls
def get_basecalls_path(dataset_name):
    basecalls_lookup_location = EXPERIMENT_NAME_TO_PATH[dataset_name]+'/guppy/reads.fastq.gz'
    if(Path(basecalls_lookup_location).exists() and dataset_name in config['USE_EXISTING_BASECALLS_EXPS']):
        # print(dataset_name, 'found')
        return basecalls_lookup_location
    return f"outputs/basecalling/{dataset_name}/guppy/reads.fastq.gz"


def get_genome_version(experiment_name):
    if experiment_name in config['EXPLICIT_REFS'].keys():
        # print('using custom genome')
        return config['EXPLICIT_REFS'][experiment_name]
    else:
        # print('using human genome')
        return HUMAN_REF_VERSION
    
def get_transcriptome_version(experiment_name):
    if experiment_name in config['EXPLICIT_TRANSCRIPTOMES'].keys():
        # print('using custom transcriptome')
        return config['EXPLICIT_TRANSCRIPTOMES'][experiment_name]
    else:
        # print('using human transcriptome')
        return HUMAN_TRANSCRIPTOME_VERSION

rule get_reference:
    output: f"{HUMAN_REF_VERSION}.fa"
    shell:
        f"""
        wget https://ftp.ensembl.org/pub/release-109/fasta/homo_sapiens/dna/{HUMAN_REF_VERSION}.fa.gz
        gzip -d {HUMAN_REF_VERSION}.fa.gz
        """
        
#TODO require my own basecalls
rule align_to_genome:
    input:
        basecalls = lambda wildcards: get_basecalls_path(wildcards.experiment_name),
        reference_path = lambda wildcards: get_genome_version(wildcards.experiment_name)+'.fa'
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
        basecalls = lambda wildcards: get_basecalls_path(wildcards.experiment_name),
        transcriptome_path = lambda wildcards: get_transcriptome_version(wildcards.experiment_name)+'.fa'
        
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
        