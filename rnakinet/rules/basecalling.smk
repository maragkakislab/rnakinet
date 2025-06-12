rule get_dorado:
    output: 
        dorado_dir = '{dorado_version}/bin/dorado'
        model_dir = model_dir
    shell:
        """
        wget https://cdn.oxfordnanoportal.com/software/analysis/{dorado_version}.tar.gz -O {dorado_version}.tar.gz
        tar -xf {dorado_version}.tar.gz
        {output.dorado_dir} download --model {output.model_dir}
        """

# TODO pass/fail ? outputs both set of reads or just pass?
rule basecalling_dorado:
    input: 
        pod5_folder = lambda wildcards: f'data/{wildcards.experiment_name}', #TODO generalize
        basecaller_location = lambda wildcards: f'{wildcards.dorado_version}/bin/dorado',
        basecalling_model = lambda wildcards: wildcards.basecalling_model, #TODO syntax
    output:
        done_txt ='outputs/basecalling/{experiment_name}/DONE.txt',
        out_reads = 'outputs/basecalling/{experiment_name}/{dorado_version}/{basecalling_model}/all_reads.fastq',
    threads: 32
    resources: gpus=1
    shell:
        """
        {input.basecaller_location} basecaller {input.basecalling_model} {input.pod5_folder} \
            --no-trim \
            --emit-fastq \
            > {output.out_reads}

        echo {input.pod5_folder} > {output.done_txt}
        """
            
rule zip_dorado_basecalls:
    input:
        'outputs/basecalling/{experiment_name}/{dorado_version}/{basecalling_model}/all_reads.fastq'
    output:
        'outputs/basecalling/{experiment_name}/{dorado_version}/{basecalling_model}/all_reads.fastq.gz'
    shell:
        'gzip -c {input} > {output}'