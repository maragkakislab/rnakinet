rule run_inference:
    input: 
        pod5_files = lambda wildcards: inference_run_to_files[wildcards.experiment_name],
        model_path = lambda wildcards: model_to_path[wildcards.model_name],
    output:
        csv_path = 'outputs/predictions/{model_name}/{experiment_name}/preds.csv',
    conda:
        "../envs/inference.yaml"
    params:
        batch_size = lambda wildcards: model_inference_params[wildcards.model_name]['batch_size'],
        max_len = lambda wildcards: model_inference_params[wildcards.model_name]['max_len'],
        min_len = lambda wildcards: model_inference_params[wildcards.model_name]['min_len'],
        skip = lambda wildcards: model_inference_params[wildcards.model_name]['skip'],
        arch = lambda wildcards: model_inference_params[wildcards.model_name]['arch'],
        threshold = lambda wildcards: model_inference_params[wildcards.model_name]['threshold'],
    threads:  lambda wildcards: model_inference_params[wildcards.model_name]['threads'],
    resources: gpus=1
    shell:
        """
        python3 scripts/inference.py \
            --pod5-files {input.pod5_files} \
            --model-path {input.model_path} \
            --arch {params.arch} \
            --max-workers {threads} \
            --threshold {params.threshold} \
            --batch-size {params.batch_size} \
            --max-len {params.max_len} \
            --min-len {params.min_len} \
            --skip {params.skip} \
            --output {output.csv_path} \
        """