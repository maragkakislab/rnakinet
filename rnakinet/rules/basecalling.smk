rule get_dorado:
    output: 
        dorado_dir = '{dorado_version}/bin/dorado',
    shell:
        """
        wget https://cdn.oxfordnanoportal.com/software/analysis/{wildcards.dorado_version}.tar.gz -O {wildcards.dorado_version}.tar.gz
        tar -xf {wildcards.dorado_version}.tar.gz
        """

rule get_basecalling_model:
    input:
        dorado_dir = '{dorado_version}/bin/dorado',
    output:
        model_dir = directory('basecalling_models_{dorado_version}/{model_name}'),
    shell:
        """
        mkdir -p basecalling_models_{wildcards.dorado_version}
        {input.dorado_dir} download --model {wildcards.model_name} --models-directory basecalling_models_{wildcards.dorado_version}/
        """

rule basecalling_dorado:
    input: 
        pod5_folder = lambda wildcards: f'{DATA_DIR}/{EXP_TO_PATH[wildcards.experiment_name]}',
        dorado_location = lambda wildcards: f'{wildcards.dorado_version}/bin/dorado',
        basecaller_location = lambda wildcards: f'basecalling_models_{wildcards.dorado_version}/{wildcards.basecalling_model}',
    output:
        done_txt = OUTPUTS_DIR + '/basecalling/{experiment_name}/{dorado_version}/{basecalling_model}/DONE.txt',
        out_reads = OUTPUTS_DIR + '/basecalling/{experiment_name}/{dorado_version}/{basecalling_model}/all_reads.fastq',
    threads:
        8
    resources:
        gpu = 2,
        gpu_model = "[gpuv100x|gpua100]",
        mem_mb = 64*1024,
        runtime = 8*24*60
    shell:
        """
        {input.dorado_location} basecaller {input.basecaller_location} {input.pod5_folder} \
            --emit-fastq \
            --recursive \
            > {output.out_reads}

        echo {input.pod5_folder} > {output.done_txt}
        """