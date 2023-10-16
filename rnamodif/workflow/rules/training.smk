import yaml
# from rnamodif.data_utils.data_paths import name_to_expname #TODO refactor, dont import from main package here

#TODO finish refactoring
rule run_training:
    # Using testing for validation (for logging), not everything needs a split -> logic is in train.py and not here
    # input: 
    #     lambda wildcards: expand("outputs/splits/{experiment_name}/FAST5_{split}_SPLIT_DONE.txt",
    #           experiment_name=[name_to_expname[name] for name in training_configs[wildcards.training_experiment_name]['training_positives_exps']+training_configs[wildcards.training_experiment_name]['training_negatives_exps'] if name in name_to_expname.keys()], #Ignoring name_to_path_extras, which do not require splitting and are explicitly set
    #           split=['train','validation','test'],
    #     )
    input:
        train_positives_lists=lambda wildcards: expand("outputs/splits/{experiment_name}/train_fast5s_list.txt",
                                          experiment_name=training_configs[wildcards.training_experiment_name]['training_positives_exps']),
        train_negatives_lists=lambda wildcards: expand("outputs/splits/{experiment_name}/train_fast5s_list.txt",
                                          experiment_name=training_configs[wildcards.training_experiment_name]['training_negatives_exps']),
    output:
        done_txt = 'checkpoints_pl/{training_experiment_name}/DONE.txt',
        arch_hyperparams_yaml = 'checkpoints_pl/{training_experiment_name}/arch_hyperparams.yaml',
    params:
        # training_positives_exps = lambda wildcards: training_configs[wildcards.training_experiment_name]['training_positives_exps'],
        # training_negatives_exps = lambda wildcards: training_configs[wildcards.training_experiment_name]['training_negatives_exps'],
        min_len = lambda wildcards: training_configs[wildcards.training_experiment_name]['min_len'],
        max_len = lambda wildcards: training_configs[wildcards.training_experiment_name]['max_len'],
        skip = lambda wildcards: training_configs[wildcards.training_experiment_name]['skip'],
        workers = lambda wildcards: training_configs[wildcards.training_experiment_name]['workers'],
        sampler = lambda wildcards: training_configs[wildcards.training_experiment_name]['sampler'],
        lr = lambda wildcards: training_configs[wildcards.training_experiment_name]['lr'],
        warmup_steps = lambda wildcards: training_configs[wildcards.training_experiment_name]['warmup_steps'],
        pos_weight = lambda wildcards: training_configs[wildcards.training_experiment_name]['pos_weight'],
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
    threads: 32 #lambda wildcards: training_configs[wildcards.training_experiment_name]['workers']
    resources: 
        gpus=1,
        # mem_mb=1024*16,
    log:
        'checkpoints_pl/{training_experiment_name}/stdout.log'
    # conda: #TODO fix this, cant activate
        # '../envs/training.yaml'
    shell:
        """
        echo "{params.arch_hyperparams}" > {output.arch_hyperparams_yaml}

        command="
        python3 scripts/train.py \
            --training-positives-lists {input.train_positives_lists} \
            --training-negatives-lists {input.train_negatives_lists} \
            --min-len {params.min_len} \
            --max-len {params.max_len} \
            --skip {params.skip} \
            --workers {params.workers} \
            --sampler {params.sampler} \
            --lr {params.lr} \
            --warmup-steps {params.warmup_steps} \
            --pos-weight {params.pos_weight} \
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

        $command &>{log}

        touch {output.done_txt}
        """
  