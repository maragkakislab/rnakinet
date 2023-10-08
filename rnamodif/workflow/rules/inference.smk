def get_weighted_flag(is_weighted):
    return '--weighted' if is_weighted else ''

def get_limit(prediction_type):
    if(prediction_type == 'predictions_limited'):
        return f"--limit {config['LIMITED_INFERENCE_FILE_LIMIT']}"
    if(prediction_type == 'predictions'):
        #no limit
        return ''
    raise Exception('prediction type not supported')

rule run_inference:
    """
    Slices each read into a window and predicts label for these windows
    """
    input: 
        experiment_path = lambda wildcards: config['EXPERIMENT_NAME_TO_PATH'][wildcards.experiment_name],
        model_path = lambda wildcards: config['MODELS'][wildcards.model_name]['path'],
    output:
        'outputs/{prediction_type}/{model_name}/{experiment_name}/windows.pickle'
    conda:
        "../envs/inference.yaml"
    params:
        batch_size = config['INFERENCE_PARAMETERS']['BATCH_SIZE'],
        max_len = config['INFERENCE_PARAMETERS']['MAX_LEN'],
        min_len = config['INFERENCE_PARAMETERS']['MIN_LEN'],
        skip = config['INFERENCE_PARAMETERS']['SKIP'],
        limit = lambda wildcards: get_limit(wildcards.prediction_type),
        arch = lambda wildcards: config['MODELS'][wildcards.model_name]['arch'],
    threads: 16
    resources: gpus=1
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
        out_pickle = 'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling.pickle',
        out_csv = 'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling.csv'
    conda:
        "../envs/inference.yaml"
    params:
        pooling = lambda wildcards: wildcards.pooling,
        threshold=lambda wildcards: MODELS[wildcards.model_name]['threshold'],
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

# run with --resources parallel_lock=1 to avoid paralelization and multiple runs utilizing the gpu, slowing the time
rule run_inference_speedtest:
    """
    Slices each read into a window and predicts label for these windows
    """
    input: 
        experiment_path = lambda wildcards: config['EXPERIMENT_NAME_TO_PATH'][wildcards.experiment_name],
        model_path = lambda wildcards: config['MODELS'][wildcards.model_name]['path'],
    output:
        'outputs/{prediction_type}/{model_name}/{experiment_name}/speedtest/readlimit_{reads_limit}_threads_{threads}.json'
    conda:
        "../envs/inference.yaml"
    params:
        batch_size = config['INFERENCE_PARAMETERS']['BATCH_SIZE'],
        max_len = config['INFERENCE_PARAMETERS']['MAX_LEN'],
        min_len = config['INFERENCE_PARAMETERS']['MIN_LEN'],
        skip = config['INFERENCE_PARAMETERS']['SKIP'],
        arch = lambda wildcards: config['MODELS'][wildcards.model_name]['arch'],
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
        
