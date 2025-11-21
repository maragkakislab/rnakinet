import yaml

checkpoint run_training:
    input:
        train_positives_pod5s=lambda wildcards: expand(OUTPUTS_DIR + "/splits/{experiment_name}/train.pod5",
                                          experiment_name=TRAINING_CONFIGS[wildcards.training_run_name]['training_positives_exps']),
        train_negatives_pod5s=lambda wildcards: expand("outputs/splits/{experiment_name}/train.pod5",
                                          experiment_name=TRAINING_CONFIGS[wildcards.training_run_name]['training_negatives_exps']),
        validation_positives_pod5s=lambda wildcards: expand("outputs/splits/{experiment_name}/validation.pod5",
                                          experiment_name=TRAINING_CONFIGS[wildcards.training_run_name]['validation_positives_exps']),
        validation_negatives_pod5s=lambda wildcards: expand("outputs/splits/{experiment_name}/validation.pod5",
                                          experiment_name=TRAINING_CONFIGS[wildcards.training_run_name]['validation_negatives_exps']),
    output:
        done_txt = CHECKPOINTS_DIR + '/{training_run_name}/DONE.txt',
        arch_hyperparams_yaml = CHECKPOINTS_DIR + '/{training_run_name}/arch_hyperparams.yaml',
        ckpt_out_dir = directory(CHECKPOINTS_DIR + "/{training_run_name}/{training_run_name}")
    params:
        min_len = lambda wildcards: TRAINING_CONFIGS[wildcards.training_run_name]['min_len'],
        max_len = lambda wildcards: TRAINING_CONFIGS[wildcards.training_run_name]['max_len'],
        valid_read_limit = lambda wildcards: TRAINING_CONFIGS[wildcards.training_run_name]['valid_read_limit'],
        skip = lambda wildcards: TRAINING_CONFIGS[wildcards.training_run_name]['skip'],
        workers = lambda wildcards: TRAINING_CONFIGS[wildcards.training_run_name]['workers'],
        batch_size = lambda wildcards: TRAINING_CONFIGS[wildcards.training_run_name]['batch_size'],
        sampler = lambda wildcards: TRAINING_CONFIGS[wildcards.training_run_name]['sampler'],
        lr = lambda wildcards: TRAINING_CONFIGS[wildcards.training_run_name]['lr'],
        warmup_steps = lambda wildcards: TRAINING_CONFIGS[wildcards.training_run_name]['warmup_steps'],
        wd = lambda wildcards: TRAINING_CONFIGS[wildcards.training_run_name]['wd'],
        arch = lambda wildcards: TRAINING_CONFIGS[wildcards.training_run_name]['arch'],
        arch_hyperparams = lambda wildcards: TRAINING_CONFIGS[wildcards.training_run_name]['arch_hyperparams'],
        grad_acc = lambda wildcards: TRAINING_CONFIGS[wildcards.training_run_name]['grad_acc'],
        early_stopping_patience = lambda wildcards: TRAINING_CONFIGS[wildcards.training_run_name]['early_stopping_patience'],
        wandb_api_key = WANDB_API_KEY,
        wandb_project_name = lambda wildcards: TRAINING_CONFIGS[wildcards.training_run_name]['wandb_project_name'],
        logging_step = lambda wildcards: TRAINING_CONFIGS[wildcards.training_run_name]['logging_step'],
        enable_progress_bar = lambda wildcards: TRAINING_CONFIGS[wildcards.training_run_name]['enable_progress_bar'],
        log_to_file = lambda wildcards: TRAINING_CONFIGS[wildcards.training_run_name]['log_to_file'],
        save_path = lambda wildcards: f'{CHECKPOINTS_DIR}/{wildcards.training_run_name}',
    threads: 32 
    resources:
        gpu = GPUS_FOR_RULES["run_training"]["gpu"],
        gpu_model = GPUS_FOR_RULES["run_training"]["gpu_model"],
        mem_mb = 200*1024,
        runtime = 8*24*60
    log:
        CHECKPOINTS_DIR + '/{training_run_name}/stdout.log'
    conda:
        '../envs/training.yaml'
    shell:
        """
        echo "{params.arch_hyperparams}" > {output.arch_hyperparams_yaml}

        command="
        python3 scripts/train.py \
            --training-positives-pod5s {input.train_positives_pod5s} \
            --training-negatives-pod5s {input.train_negatives_pod5s} \
            --validation-positives-pod5s {input.validation_positives_pod5s} \
            --validation-negatives-pod5s {input.validation_negatives_pod5s} \
            --min-len {params.min_len} \
            --max-len {params.max_len} \
            --valid-read-limit {params.valid_read_limit} \
            --skip {params.skip} \
            --workers {params.workers} \
            --batch-size {params.batch_size} \
            --sampler {params.sampler} \
            --lr {params.lr} \
            --warmup-steps {params.warmup_steps} \
            --wd {params.wd} \
            --arch {params.arch} \
            --arch-hyperparams-yaml {output.arch_hyperparams_yaml} \
            --grad-acc {params.grad_acc} \
            --early-stopping-patience {params.early_stopping_patience} \
            --experiment-name {wildcards.training_run_name} \
            --wandb-api-key {params.wandb_api_key} \
            --wandb-project-name {params.wandb_project_name} \
            --logging-step {params.logging_step} \
            --enable-progress-bar {params.enable_progress_bar} \
            --save-path {params.save_path} \
            "

        if [ "{params.log_to_file}" = "True" ]; then
            $command &>{log}
        else
            $command
        fi

        touch {output.done_txt}
        """
        
  