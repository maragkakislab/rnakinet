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
    # conda:
        # "../envs/inference.yaml" #TODO fix this - needs rnamodif as package
    params:
        window = lambda wildcards: config['MODELS'][wildcards.model_name]['window'],
        weighted = lambda wildcards: get_weighted_flag(config['MODELS'][wildcards.model_name]['weighted']),
        batch_size = config['INFERENCE_BATCH_SIZE'],
        limit = lambda wildcards: get_limit(wildcards.prediction_type),
    threads: 16
    resources: gpus=1
    shell:
        """
        python3 scripts/inference.py \
            --path {input.experiment_path} \
            --checkpoint {input.model_path} \
            --output {output} \
            --max_workers {threads} \
            --window {params.window} \
            --batch_size {params.batch_size} \
            {params.weighted} \
            {params.limit} \
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
  