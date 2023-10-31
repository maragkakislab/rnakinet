BASECALLER_VERSION = 'ont-guppy_6.4.8_linux64'

# Downloads the basecaller software
rule get_basecaller:
    output: f"ont-guppy/bin/guppy_basecaller"
    shell:
        f"""
        wget https://cdn.oxfordnanoportal.com/software/analysis/ont-guppy_6.4.8_linux64.tar.gz
        tar -xf ont-guppy_6.4.8_linux64.tar.gz
        """
        
# Basecalls fast5 files into fastq files
rule basecalling:
    input: 
        experiment_path = lambda wildcards: directory(experiments_data[wildcards.experiment_name].get_path()),
        basecaller_location = "ont-guppy/bin/guppy_basecaller",
    output:
        'outputs/basecalling/{experiment_name}/DONE.txt'
    params:
        kit = lambda wildcards: experiments_data[wildcards.experiment_name].get_kit(),
        flowcell = lambda wildcards: experiments_data[wildcards.experiment_name].get_flowcell(),
    threads: 32
    resources: gpus=1
    shell:
        """
        {input.basecaller_location} \
            -x "auto" \
            --flowcell {params.flowcell} \
            --kit {params.kit} \
            --records_per_fastq 0 \
            --trim_strategy none \
            --save_path outputs/basecalling/{wildcards.experiment_name}/guppy/ \
            --recursive \
            --gpu_runners_per_device 1 \
            --num_callers {threads} \
            --chunks_per_runner 512 \
            --compress_fastq \
            --calib_detect \
            --input_path {input.experiment_path} \
            
        echo {input.experiment_path} > {output}
        """

# Merges multiple fastq files into a single fastq file
rule merge_fastq_files:
    input:
        'outputs/basecalling/{experiment_name}/DONE.txt'
    output:
        "outputs/basecalling/{experiment_name}/guppy/reads.fastq.gz"
    conda:
        "../envs/merge_fastq.yaml"
    shell:
        """
        zcat outputs/basecalling/{wildcards.experiment_name}/guppy/pass/fastq_runid*.fastq.gz | pigz > {output}
        """
        
        
rule merge_fastq_files_fail_pass:
    input:
        'outputs/basecalling/{experiment_name}/DONE.txt'
    output:
        "outputs/basecalling/{experiment_name}/guppy/all_reads.fastq.gz"
    conda:
        "../envs/merge_fastq.yaml"
    shell:
        """
        zcat outputs/basecalling/{wildcards.experiment_name}/guppy/*/fastq_runid*.fastq.gz | pigz > {output}
        """