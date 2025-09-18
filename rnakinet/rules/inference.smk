rule run_inference:
    input: 
        pod5_files = lambda wildcards: INFERENCE_RUN_TO_FILES[wildcards.experiment_name],
        model_path = lambda wildcards: MODEL_TO_PATH[wildcards.model_name],
    output:
        csv_path = OUTPUTS_DIR + '/predictions/{model_name}/{experiment_name}/preds.csv',
    conda:
        "../envs/inference.yaml"
    params:
        batch_size = lambda wildcards: MODEL_INFERENCE_PARAMS[wildcards.model_name]['batch_size'],
        max_len = lambda wildcards: MODEL_INFERENCE_PARAMS[wildcards.model_name]['max_len'],
        min_len = lambda wildcards: MODEL_INFERENCE_PARAMS[wildcards.model_name]['min_len'],
        skip = lambda wildcards: MODEL_INFERENCE_PARAMS[wildcards.model_name]['skip'],
        arch = lambda wildcards: MODEL_INFERENCE_PARAMS[wildcards.model_name]['arch'],
        threshold = lambda wildcards: MODEL_INFERENCE_PARAMS[wildcards.model_name]['threshold'],
    threads:  lambda wildcards: MODEL_INFERENCE_PARAMS[wildcards.model_name]['threads'],
    resources:
        gpu = 1,
        gpu_model = "[gpuv100x|gpua100]",
        mem_mb = 64*1024,
        runtime = 8*24*60
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