import yaml

#TODO wandb logging
rule run_training:
    input:
        train_positives_pod5s=lambda wildcards: expand("outputs/splits/{experiment_name}/train.pod5",
                                          experiment_name=training_positives_exps),
        train_negatives_pod5s=lambda wildcards: expand("outputs/splits/{experiment_name}/train.pod5",
                                          experiment_name=training_negatives_exps),
        validation_positives_pod5s=lambda wildcards: expand("outputs/splits/{experiment_name}/validation.pod5",
                                          experiment_name=validation_positives_exps),
        validation_negatives_pod5s=lambda wildcards: expand("outputs/splits/{experiment_name}/validation.pod5",
                                          experiment_name=validation_negatives_exps),
    output:
        done_txt = 'checkpoints_pl/{training_experiment_name}/DONE.txt',
        arch_hyperparams_yaml = 'checkpoints_pl/{training_experiment_name}/arch_hyperparams.yaml',
    params:
        min_len = lambda wildcards: training_configs[wildcards.training_experiment_name]['min_len'],
        max_len = lambda wildcards: training_configs[wildcards.training_experiment_name]['max_len'],
        valid_read_limit = lambda wildcards: training_configs[wildcards.training_experiment_name]['valid_read_limit'],
        skip = lambda wildcards: training_configs[wildcards.training_experiment_name]['skip'],
        workers = lambda wildcards: training_configs[wildcards.training_experiment_name]['workers'],
        sampler = lambda wildcards: training_configs[wildcards.training_experiment_name]['sampler'],
        lr = lambda wildcards: training_configs[wildcards.training_experiment_name]['lr'],
        warmup_steps = lambda wildcards: training_configs[wildcards.training_experiment_name]['warmup_steps'],
        wd = lambda wildcards: training_configs[wildcards.training_experiment_name]['wd'],
        arch = lambda wildcards: training_configs[wildcards.training_experiment_name]['arch'],
        arch_hyperparams = lambda wildcards: training_configs[wildcards.training_experiment_name]['arch_hyperparams'],
        grad_acc = lambda wildcards: training_configs[wildcards.training_experiment_name]['grad_acc'],
        early_stopping_patience = lambda wildcards: training_configs[wildcards.training_experiment_name]['early_stopping_patience'],
        comet_api_key = lambda wildcards: training_configs[wildcards.training_experiment_name]['comet_api_key'],
        comet_project_name = lambda wildcards: training_configs[wildcards.training_experiment_name]['comet_project_name'],
        logging_step = lambda wildcards: training_configs[wildcards.training_experiment_name]['logging_step'],
        enable_progress_bar = lambda wildcards: training_configs[wildcards.training_experiment_name]['enable_progress_bar'],
        log_to_file = lambda wildcards: training_configs[wildcards.training_experiment_name]['log_to_file'],
        save_path = lambda wildcards: f'checkpoints_pl/{wildcards.training_experiment_name}',
    threads: 32 
    resources: 
        gpus=1,
    log:
        'checkpoints_pl/{training_experiment_name}/stdout.log'
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
            --sampler {params.sampler} \
            --lr {params.lr} \
            --warmup-steps {params.warmup_steps} \
            --wd {params.wd} \
            --arch {params.arch} \
            --arch-hyperparams-yaml {output.arch_hyperparams_yaml} \
            --grad-acc {params.grad_acc} \
            --early-stopping-patience {params.early_stopping_patience} \
            --experiment-name {wildcards.training_experiment_name} \
            --comet-api-key {params.comet_api_key} \
            --comet-project-name {params.comet_project_name} \
            --logging-step {params.logging_step} \
            --enable-progress-bar {params.enable_progress_bar} \
            --save-path checkpoints_pl \
            "

        if [ "{params.log_to_file}" = "True" ]; then
            $command &>{log}
        else
            $command
        fi

        touch {output.done_txt}
        """
        
  