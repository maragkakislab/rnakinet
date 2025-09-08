rule split_readids_on_chromosomes:
    input:
        bam_path=lambda wildcards : f"{OUTPUTS_DIR}/alignment/{wildcards.experiment_name}/{BASECALLING_CONFIG['dorado_version']}/{BASECALLING_CONFIG['basecalling_model']}/reads-align.genome.sorted.bam",
    output:
        OUTPUTS_DIR + "/splits/{experiment_name}/train_readids.txt",
        OUTPUTS_DIR + "/splits/{experiment_name}/test_readids.txt",
        OUTPUTS_DIR + "/splits/{experiment_name}/validation_readids.txt",
    conda:
        "../envs/bam_splitting.yaml"
    params:
        train_chromosomes=lambda wildcards: REFERENCE_TO_SPLITS[EXP_TO_REFERENCE[wildcards.experiment_name]]['train_chrs'],
        test_chromosomes=lambda wildcards: REFERENCE_TO_SPLITS[EXP_TO_REFERENCE[wildcards.experiment_name]]['test_chrs'],
        validation_chromosomes=lambda wildcards: REFERENCE_TO_SPLITS[EXP_TO_REFERENCE[wildcards.experiment_name]]['valid_chrs'],
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
        ids = OUTPUTS_DIR + "/splits/{experiment_name}/{split}_readids.txt", #The split needs to be non-empty txt file
        experiment_path = lambda wildcards: f'{DATA_DIR}/{EXP_TO_PATH[wildcards.experiment_name]}',
    output: #TODO add outputs/splits/expname/{split} folder as output for viz rules
        OUTPUTS_DIR + "/splits/{experiment_name}/POD5_{split}_SPLIT_DONE.txt",
        OUTPUTS_DIR + "/splits/{experiment_name}/{split}.pod5",
    conda:
        "../envs/pod5_splitting.yaml"
    threads: 16
    resources:
        mem_mb = 32*1024
    shell:
        """
        pod5 filter \
            {input.experiment_path} \
            --recursive \
            --output {OUTPUTS_DIR}/splits/{wildcards.experiment_name}/{wildcards.split}.pod5 \
            --threads {threads} \
            --ids {input.ids} \
            --missing-ok
            
        touch {output}
        """