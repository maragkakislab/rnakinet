def get_splits(dataset_name):
    if dataset_name in config['EXPLICIT_SPLITS'].keys():
        return config['EXPLICIT_SPLITS'][dataset_name]
    return {
        'train':config['default_train_chrs'],
        'test':config['default_test_chrs'],
        'validation':config['default_validation_chrs'],
    }

rule split_readids_on_chromosomes:
    input:
        bam_path="outputs/alignment/{experiment_name}/reads-align.genome.sorted.bam",
    output:
        "outputs/splits/{experiment_name}/train_readids.txt",
        "outputs/splits/{experiment_name}/test_readids.txt",
        "outputs/splits/{experiment_name}/validation_readids.txt",
    conda:
        "../envs/bam_splitting.yaml"
    params:
        train_chromosomes=lambda wildcards: get_splits(wildcards.experiment_name)['train'],
        test_chromosomes=lambda wildcards: get_splits(wildcards.experiment_name)['test'],
        validation_chromosomes=lambda wildcards: get_splits(wildcards.experiment_name)['validation'],
    shell:
        """
        python3 scripts/splitting.py \
            --bam_path {input.bam_path} \
            --output_path outputs/splits/{wildcards.experiment_name}/ \
            --train_chromosomes {params.train_chromosomes} \
            --test_chromosomes {params.test_chromosomes} \
            --validation_chromosomes {params.validation_chromosomes} \
        """
        
        

rule create_split_fast5s:
    '''
    Creates new multiread fast5 files for given readids, so they can be loaded faster during training/inference
    '''
    input:
        ids = "outputs/splits/{experiment_name}/{split}_readids.txt", #The split needs to be non-empty txt file
        experiment_path = lambda wildcards: config['EXPERIMENT_NAME_TO_PATH'][wildcards.experiment_name],
    output:
        "outputs/splits/{experiment_name}/FAST5_{split}_SPLIT_DONE.txt"
    conda:
        "../envs/fast5_splitting.yaml"
    threads: workflow.cores
    shell:
        """
        fast5_subset \
            --input {input.experiment_path} \
            --recursive \
            --save_path outputs/splits/{wildcards.experiment_name}/{wildcards.split}/ \
            --threads {threads} \
            --read_id_list {input.ids}
            
        touch {output}
        """
