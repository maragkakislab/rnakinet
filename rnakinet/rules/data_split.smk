rule split_readids_on_chromosomes:
    input:
        bam_path=lambda wildcards, params: "outputs/alignment/{wildcards.experiment_name}/{params.dorado_version}/{params.basecalling_model}/reads-align.genome.sorted.bam",
        # indexed_bam_path="outputs/alignment/{experiment_name}/reads-align.genome.sorted.bam.bai",
    output:
        "outputs/splits/{experiment_name}/train_readids.txt",
        "outputs/splits/{experiment_name}/test_readids.txt",
        "outputs/splits/{experiment_name}/validation_readids.txt",
    conda:
        "../envs/bam_splitting.yaml"
    params:
        train_chromosomes=train_chrs,
        test_chromosomes=test_chrs,
        validation_chromosomes=valid_chrs,
        dorado_version='dorado-1.0.1-linux-x64',
        basecalling_model='rna004_130bps_hac@v5.2.0',
    shell:
        """
        python3 scripts/splitting.py \
            --bam_path {input.bam_path} \
            --output_path outputs/splits/{wildcards.experiment_name}/ \
            --train_chromosomes {params.train_chromosomes} \
            --test_chromosomes {params.test_chromosomes} \
            --validation_chromosomes {params.validation_chromosomes} \
        """
        
rule create_split_pod5s:
    '''
    Creates new multiread pod5 files for given readids, so they can be loaded faster during training/inference
    '''
    input:
        ids = "outputs/splits/{experiment_name}/{split}_readids.txt", #The split needs to be non-empty txt file
        experiment_path = lambda wildcards: experiments_data[wildcards.experiment_name].get_path(), #TODO set paths to raw pod5s
    output: #TODO add outputs/splits/expname/{split} folder as output for viz rules
        "outputs/splits/{experiment_name}/POD5_{split}_SPLIT_DONE.txt",
        directory("outputs/splits/{experiment_name}/{split}"),
    conda:
        "../envs/pod5_splitting.yaml"
    threads: 16
    shell:
        """
        pod5 filter \
            {input.experiment_path} \
            --recursive \
            --output outputs/splits/{wildcards.experiment_name}/{wildcards.split}/ \
            --threads {threads} \
            --ids {input.ids}
            
        touch {output}
        """
        
rule create_split_files_list:
    input:
        "outputs/splits/{experiment_name}/POD5_{split}_SPLIT_DONE.txt" #TODO for all split, dont require this - new rule, allow all or scratch compeltely?
    output:
        txt_file="outputs/splits/{experiment_name}/{split}_pod5s_list.txt" #TODO this will be only 1 pod5? unnecessary rule?
    run:
        # files_list = experiments_data[wildcards.experiment_name].get_split_fast5_files(wildcards.split)
        files_list = f"outputs/splits/{wildcards.experiment_name}/{wildcards.split}".rglob('*.pod5')
        with open(output.txt_file, "w") as out_file:
            for file_path in files_list:
                out_file.write(str(file_path) + "\n")

