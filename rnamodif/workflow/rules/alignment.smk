HUMAN_REF_VERSION = config['HUMAN_REF_VERSION']
EXPLICIT_BASECALL_PATHS = config['EXPLICIT_BASECALL_PATHS']

def get_basecalls_path(dataset_name):
    if(dataset_name in EXPLICIT_BASECALL_PATHS.keys()):
        return EXPLICIT_BASECALL_PATHS[dataset_name]
    return f"outputs/basecalling/{dataset_name}/guppy/reads.fastq.gz"

rule get_reference:
    output: f"{HUMAN_REF_VERSION}.fa"
    shell:
        f"""
        wget https://ftp.ensembl.org/pub/release-109/fasta/homo_sapiens/dna/{HUMAN_REF_VERSION}.fa.gz
        gzip -d {HUMAN_REF_VERSION}.fa.gz
        """
        
rule align_to_genome:
    input:
        basecalls = lambda wildcards: get_basecalls_path(wildcards.experiment_name),
        reference_path = HUMAN_REF_VERSION+'.fa'
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
        
rule run_flagstat:
    input:
        "outputs/alignment/{experiment_name}/reads-align.genome.sorted.bam"
    output:
        "outputs/alignment/{experiment_name}/flagstat.txt"
    conda:
        "../envs/alignment.yaml"
    shell:
        "samtools flagstat {input} > {output}"