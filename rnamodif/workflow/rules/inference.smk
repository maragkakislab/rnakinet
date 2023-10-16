rule run_inference:
    """
    Slices each read into a window and predicts label for these windows
    """
    input: 
        experiment_path = lambda wildcards: experiments_data[wildcards.experiment_name].get_path(),
        model_path = lambda wildcards: models_data[wildcards.model_name].get_path(),
    output:
        'outputs/{prediction_type}/{model_name}/{experiment_name}/windows.pickle' #TODO redo to something else than pickle
    conda:
        "../envs/inference.yaml"
    params:
        batch_size = lambda wildcards: models_data[wildcards.model_name].get_batch_size(),
        max_len = lambda wildcards: models_data[wildcards.model_name].get_max_len(),
        min_len = lambda wildcards: models_data[wildcards.model_name].get_min_len(),
        skip = lambda wildcards: models_data[wildcards.model_name].get_skip(),
        
        limit = lambda wildcards: '', #TODO refactor away
        arch = lambda wildcards: models_data[wildcards.model_name].get_arch(),
    threads: 16 #TODO why 16
    resources: gpus=1
    wildcard_constraints:
        prediction_type='(predictions|predictions_limited)'
    shell:
        """
        python3 scripts/inference.py \
            --arch {params.arch} \
            --path {input.experiment_path} \
            --checkpoint {input.model_path} \
            --max_workers {threads} \
            --batch-size {params.batch_size} \
            {params.limit} \
            --max-len {params.max_len} \
            --min-len {params.min_len} \
            --skip {params.skip} \
            --output {output} \
        """

rule run_pooling:
    input:
        window_predictions = 'outputs/{prediction_type}/{model_name}/{experiment_name}/windows.pickle'
    output:
        out_pickle = 'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling.pickle', #redo to something else than pickle
        out_csv = 'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling.csv'
    conda:
        "../envs/inference.yaml"
    params:
        pooling = lambda wildcards: wildcards.pooling, #TODO remove pooling? Use max always?
        threshold = lambda wildcards: models_data[wildcards.model_name].get_threshold(), #TODO why do i need threshold
    threads: 1
    shell:
        """
        python3 scripts/pooling.py \
            --window_predictions {input.window_predictions} \
            --out_pickle {output.out_pickle} \
            --out_csv {output.out_csv} \
            --pooling {params.pooling} \
            --threshold {params.threshold} \
        """

# run with snakemake --resources parallel_lock=1 to avoid paralelization and multiple runs utilizing the gpu, slowing the time
rule run_inference_speedtest:
    input: 
        experiment_path = lambda wildcards: experiments_data[wildcards.experiment_name].get_path(),
        model_path = lambda wildcards: models_data[wildcards.model_name].get_path(),
    output:
        'outputs/{prediction_type}/{model_name}/{experiment_name}/speedtest/readlimit_{reads_limit}_threads_{threads}.json'
    conda:
        "../envs/inference.yaml"
    params:
        batch_size = lambda wildcards: models_data[wildcards.model_name].get_batch_size(),
        max_len = lambda wildcards: models_data[wildcards.model_name].get_max_len(),
        min_len = lambda wildcards: models_data[wildcards.model_name].get_min_len(),
        skip = lambda wildcards: models_data[wildcards.model_name].get_skip(),
        arch = lambda wildcards: models_data[wildcards.model_name].get_arch(),
        reads_limit = lambda wildcards: wildcards.reads_limit
    threads: lambda wildcards: int(wildcards.threads)
    resources: 
        gpus=1,
        parallel_lock=1, #used to restrict parallelization of this rule
    shell:
        """
        python3 scripts/inference_speedtest.py \
            --arch {params.arch} \
            --path {input.experiment_path} \
            --checkpoint {input.model_path} \
            --max_workers {threads} \
            --batch-size {params.batch_size} \
            --reads_limit {params.reads_limit} \
            --max-len {params.max_len} \
            --min-len {params.min_len} \
            --skip {params.skip} \
            --smake_threads {threads} \
            --exp_name {wildcards.experiment_name} \
            --output {output} \
        """
        
