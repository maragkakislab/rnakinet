import os
import glob

def get_model_path(wc):
    # if user specifies manual path, use that
    manual = MODEL_TO_PATH.get(wc.model_name)
    if manual:
        return manual

    # fetching model from output specified by training rule
    out = checkpoints.run_training.get(training_run_name=wc.model_name).output
    ckpt_dir = out.ckpt_out_dir  # = CHECKPOINTS_DIR/{name}/{name}

    # searching for best checkpoint
    pattern = os.path.join(ckpt_dir, "best-step=*valid_loss=*.ckpt")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]

    # if filename pattern for best checkpoint not found, use last.ckpt instead
    last = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.exists(last):
        return last

    # if no checkpoints found, raise error
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

rule run_inference:
    input: 
        pod5_files = lambda wildcards: INFERENCE_RUN_TO_FILES[wildcards.experiment_name],
        model_path = get_model_path,
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
        gpu = GPUS_FOR_RULES["inference_rnakinet"]["gpu"],
        gpu_model = GPUS_FOR_RULES["inference_rnakinet"]["gpu_model"],
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

rule run_full_exp_inference:
    input: 
        pod5_files = lambda wildcards: f'{DATA_DIR}/experiments/{wildcards.experiment_name}/',
        model_path = get_model_path,
    output:
        csv_path = OUTPUTS_DIR + '/full_exp_predictions/{model_name}/{experiment_name}/preds.csv',
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
        gpu = GPUS_FOR_RULES["inference_rnakinet"]["gpu"],
        gpu_model = GPUS_FOR_RULES["inference_rnakinet"]["gpu_model"],
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

rule calculate_percent_positive:
    input:
        preds = lambda wildcards: expand(
            OUTPUTS_DIR + '/{predictions}/{model_name}/{experiment_name}/preds.csv',
            predictions=wildcards.predictions,
            experiment_name=wildcards.experiment_name,
            model_name=wildcards.model_name,
        ),
    output:
        OUTPUTS_DIR + '/{predictions}_pct_pos/{model_name}/{experiment_name}_percent_positive.txt',
    conda:
        "../envs/inference.yaml"
    shell:
        """
        awk -F',' 'NR>1 {{{{total++; if ($3 == "True") count++}}}} END {{{{if (total > 0) printf "%.20f\\n", count/total; else print "0"}}}}' {input.preds} > {output}
        """