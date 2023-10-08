BASECALLER_VERSION = config['BASECALLER_VERSION']

rule get_basecaller:
    output: f"ont-guppy/bin/guppy_basecaller"
    shell:
        f"""
        wget https://cdn.oxfordnanoportal.com/software/analysis/{BASECALLER_VERSION}.tar.gz
        tar -xf {BASECALLER_VERSION}.tar.gz
        """
        
rule basecalling:
    input: 
        #TODO input is a folder - specify to a file
        experiment_path = lambda wildcards: config['EXPERIMENT_NAME_TO_PATH'][wildcards.experiment_name],
        basecaller_location = "ont-guppy/bin/guppy_basecaller",
    output:
        'outputs/basecalling/{experiment_name}/DONE.txt'
    params:
        #TODO raise an error if the kit is not defined
        kit = lambda wildcards: config['KITS'][wildcards.experiment_name],
    threads: 32
    resources: gpus=1
    shell:
        """
        {input.basecaller_location} \
            -x "auto" \
            --flowcell FLO-MIN106 \
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

        
#TODO cleanup unused fastq files
rule merge_fastq_files:
    input:
        'outputs/basecalling/{experiment_name}/DONE.txt'
    output:
        "outputs/basecalling/{experiment_name}/guppy/reads.fastq.gz"
    conda:
        #TODO why cant it find envs/...???
        "../envs/merge_fastq.yaml"
    shell:
        """
        zcat outputs/basecalling/{wildcards.experiment_name}/guppy/pass/fastq_runid*.fastq.gz | pigz > {output}
        """