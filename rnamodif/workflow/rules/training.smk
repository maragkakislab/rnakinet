import yaml

rule run_training:
    # input: 
        #TODO require files?
    output:
        done_txt = 'checkpoints_pl/{training_experiment_name}/DONE.txt',
        arch_hyperparams_yaml = temp('checkpoints_pl/{training_experiment_name}/arch_hyperparams.yaml'),
    params:
        training_positives_exps = lambda wildcards: training_configs[wildcards.training_experiment_name]['training_positives_exps'],
        training_negatives_exps = lambda wildcards: training_configs[wildcards.training_experiment_name]['training_negatives_exps'],
        min_len = lambda wildcards: training_configs[wildcards.training_experiment_name]['min_len'],
        max_len = lambda wildcards: training_configs[wildcards.training_experiment_name]['max_len'],
        skip = lambda wildcards: training_configs[wildcards.training_experiment_name]['skip'],
        workers = lambda wildcards: training_configs[wildcards.training_experiment_name]['workers'],
        sampler = lambda wildcards: training_configs[wildcards.training_experiment_name]['sampler'],
        lr = lambda wildcards: training_configs[wildcards.training_experiment_name]['lr'],
        warmup_steps = lambda wildcards: training_configs[wildcards.training_experiment_name]['warmup_steps'],
        pos_weight = lambda wildcards: training_configs[wildcards.training_experiment_name]['pos_weight'],
        arch = lambda wildcards: training_configs[wildcards.training_experiment_name]['arch'],
        arch_hyperparams = lambda wildcards: training_configs[wildcards.training_experiment_name]['arch_hyperparams'],
        grad_acc = lambda wildcards: training_configs[wildcards.training_experiment_name]['grad_acc'],
        early_stopping_patience = lambda wildcards: training_configs[wildcards.training_experiment_name]['early_stopping_patience'],
        comet_api_key = lambda wildcards: training_configs[wildcards.training_experiment_name]['comet_api_key'],
        comet_project_name = lambda wildcards: training_configs[wildcards.training_experiment_name]['comet_project_name'],
        logging_step = lambda wildcards: training_configs[wildcards.training_experiment_name]['logging_step'],
        enable_progress_bar = lambda wildcards: training_configs[wildcards.training_experiment_name]['enable_progress_bar'],
        save_path = lambda wildcards: f'checkpoints_pl/{wildcards.training_experiment_name}',
    threads: 64 #lambda wildcards: training_configs[wildcards.training_experiment_name]['workers']
    resources: 
        gpus=1,
        # mem_mb=1024*16,
    log:
        'checkpoints_pl/{training_experiment_name}/stdout.log'
    run:
        with open(output.arch_hyperparams_yaml, 'w') as file:
            yaml.dump(params.arch_hyperparams, file)

        shell(
            """
            python3 scripts/train.py \
                --training-positives-exps {params.training_positives_exps} \
                --training-negatives-exps {params.training_negatives_exps} \
                --min-len {params.min_len} \
                --max-len {params.max_len} \
                --skip {params.skip} \
                --workers {params.workers} \
                --sampler {params.sampler} \
                --lr {params.lr} \
                --warmup-steps {params.warmup_steps} \
                --pos-weight {params.pos_weight} \
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
                &> {log}
            """
        )
        shell('touch {output.done_txt}')
  