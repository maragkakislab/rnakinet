rule split_readids_on_chromosomes:
    input:
        bam_path=lambda wildcards : f"outputs/alignment/{wildcards.experiment_name}/{config['dorado_version']}/{config['basecalling_model']}/reads-align.genome.sorted.bam",
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
    Creates new pod5 file for given readids, so they can be loaded faster during training
    '''
    input:
        ids = "outputs/splits/{experiment_name}/{split}_readids.txt", #The split needs to be non-empty txt file
        experiment_path = lambda wildcards: exp_to_path[wildcards.experiment_name], #TODO set paths to raw pod5s
    output: #TODO add outputs/splits/expname/{split} folder as output for viz rules
        "outputs/splits/{experiment_name}/POD5_{split}_SPLIT_DONE.txt",
        # directory("outputs/splits/{experiment_name}/{split}"),
    conda:
        "../envs/pod5_splitting.yaml"
    threads: 16
    shell:
        """
        pod5 filter \
            {input.experiment_path} \
            --recursive \
            --output outputs/splits/{wildcards.experiment_name}/{wildcards.split}.pod5 \
            --threads {threads} \
            --ids {input.ids} \
            --missing-ok
            
        touch {output}
        """