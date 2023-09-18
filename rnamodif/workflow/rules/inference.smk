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
        "../envs/inference.yaml" #TODO fix this - needs rnamodif as package
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
        'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling.pickle'
    # conda: TODO
    params:
        pooling = lambda wildcards: wildcards.pooling
    threads: 1
    shell:
        """
        python3 scripts/pooling.py \
            --window_predictions {input.window_predictions} \
            --output {output} \
            --pooling {params.pooling} \
        """
  