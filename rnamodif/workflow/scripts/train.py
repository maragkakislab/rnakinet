import comet_ml
from rnamodif.data_utils.dataloading_uncut import TrainingDatamodule
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pathlib import Path
from rnamodif.data_utils.data_paths import name_to_files #TODO refactor to not take magic strings in snakemake training
from rnamodif.workflow.scripts.helpers import arch_map #TODO refactor model mapping from strings 
import argparse
import yaml


def parse_args(parser):
    parser.add_argument(
        '--training-positives-lists', 
        type=str, 
        required=True, 
        nargs='+', 
        help='Paths to the files containing positive fast5 files'
    )
    parser.add_argument(
        '--training-negatives-lists', 
        type=str, 
        required=True, 
        nargs='+', 
        help='Paths to the files containing negative fast5 files.'
    )
    
    
    
#     parser.add_argument(
#         '--training-positives-exps', 
#         type=str, 
#         required=True, 
#         nargs='+', 
#         help='Paths to the files containing positive sequences.'
#     )
#     parser.add_argument(
#         '--training-negatives-exps', 
#         type=str, 
#         required=True, 
#         nargs='+', 
#         help='Paths to the files containing negative sequences.'
#     )
    
    parser.add_argument('--min-len', type=int, required=True, help='Minimum length of the signal sequence for training and validation', default=5000)
    parser.add_argument('--max-len', type=int, required=True, help='Maximum length of the signal sequence for training and validation', default=400000)
    parser.add_argument('--skip', type=int, required=True, help='How many signal steps to skip at the beginning of each read', default=5000)
    parser.add_argument('--workers', type=int, required=True, help='How many workers to use for dataloading. Each worker makes replicate of the dataset', default=32)
    parser.add_argument('--sampler', type=str, required=True, help='How to sample from training datasets. Options: ratio (each fast5 file has the same probability) or uniform (each experiment has the same probability)', default='ratio')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate', default=1e-3)
    parser.add_argument('--warmup-steps', type=int, required=True, help='Learning rate warmup steps', default=1000)
    parser.add_argument('--pos-weight', type=float, required=True, help='Positive class weight', default=1.0)
    parser.add_argument('--wd', type=float, required=True, help='Weight decay', default=0.01)
    
    
    parser.add_argument('--arch', type=str, required=True, help='Type of architecture to use, options: custom_cnn')
    parser.add_argument('--arch-hyperparams-yaml', type=str, required=True, help='Path to a yaml file containing architecture-specific hyperparameters')
    
    parser.add_argument('--grad-acc', type=int, required=True, help='Gradient accumulation steps. The same as effective batch size when batch_size == 1', default=64)
    parser.add_argument('--early-stopping-patience', type=int, required=True, help='How many validation steps to wait for improvement beore stopping the training', default=100)
    
    parser.add_argument('--experiment-name', type=str, required=True, help='Name of the experiment to use for logging and naming')
    
    #TODO make logging optional
    parser.add_argument('--comet-api-key', type=str, required=True, help='Comet API key for logging')
    parser.add_argument('--comet-project-name', type=str, required=True, help='Comet project name for logging')
    parser.add_argument('--logging-step', type=int, required=True, help='After how many effective batches to log metrics', default=500)
    #TODO parse boolean or remove
    parser.add_argument('--enable-progress-bar', type=str, required=True, help='Whether to print proress bar, options yes or no')
    
    parser.add_argument('--save-path', type=str, required=True, help='Path for the final model file')
    return parser    


def read_txt_to_list(file_path):
    with open(file_path, "r") as file:
        strings_list = [line.strip() for line in file]
    return strings_list

#Configs
def get_datasets_config(args):
    # matrix
    train_pos_files = [read_txt_to_list(txt_path) for txt_path in args.training_positives_lists]
    train_neg_files = [read_txt_to_list(txt_path) for txt_path in args.training_negatives_lists]
    
    # train_pos_files = [
    #     name_to_files[exp]['train'] for exp in args.training_positives_exps
    # ]
    # train_neg_files = [
    #     name_to_files[exp]['train'] for exp in args.training_negatives_exps
    # ]

    valid_exp_to_files_pos = {
        '5eu_2020_pos':name_to_files['nia_2020_pos']['test'], 
        'Nanoid_pos_1':name_to_files['nano_pos_1']['test'], 
        'Nanoid_pos_2':name_to_files['nano_pos_2']['test'], 
        'Nanoid_pos_3':name_to_files['nano_pos_3']['test'], 
        
        '5eu_2022_chr1_pos_march':name_to_files['nia_2022_pos_march']['test'],
        '5eu_2022_chr20_pos_march':name_to_files['nia_2022_pos_march']['validation'],
        
        '5eu_2022_chr1_pos_may':name_to_files['nia_2022_pos_may']['test'],
        '5eu_2022_chr20_pos_may':name_to_files['nia_2022_pos_may']['validation'],
    }

    valid_exp_to_files_neg = {
        'UNM_2020':name_to_files['nia_2020_neg']['test'], 
        'Nanoid_neg_1':name_to_files['nano_neg_1']['test'], 
        'Nanoid_neg_2':name_to_files['nano_neg_2']['test'], 
        'Nanoid_neg_3':name_to_files['nano_neg_3']['test'], 
        
        '5eu_2022_chr1_neg_march':name_to_files['nia_2022_neg_march']['test'],
        '5eu_2022_chr20_neg_march':name_to_files['nia_2022_neg_march']['validation'],
        
        '5eu_2022_chr1_neg_may':name_to_files['nia_2022_neg_may']['test'],
        '5eu_2022_chr20_neg_may':name_to_files['nia_2022_neg_may']['validation'],
        
    }

    valid_auroc_tuples = [
        ('5eu_2020_pos', 'UNM_2020', '2020'),
        ('Nanoid_pos_1', 'Nanoid_neg_1', 'Nanoid_1'),
        ('Nanoid_pos_2', 'Nanoid_neg_2', 'Nanoid_2'),
        ('Nanoid_pos_3', 'Nanoid_neg_3', 'Nanoid_3'),
        ('5eu_2022_chr1_pos_march', '5eu_2022_chr1_neg_march', '2022 march test'),
        ('5eu_2022_chr1_pos_may', '5eu_2022_chr1_neg_may', '2022 may test'),
        ('5eu_2022_chr20_pos_march', '5eu_2022_chr20_neg_march', '2022 march valid'),
        ('5eu_2022_chr20_pos_may', '5eu_2022_chr20_neg_may', '2022 may valid'),
        
    ]
    datasets = {
        'train_pos_files':train_pos_files,
        'train_neg_files':train_neg_files,
        'valid_exp_to_files_pos':valid_exp_to_files_pos,
        'valid_exp_to_files_neg':valid_exp_to_files_neg,
        'valid_auroc_tuples':valid_auroc_tuples,
    }
    return datasets



def get_dataloading_config(args):
    data_params = {
        'batch_size':1,
        'valid_per_dset_read_limit':250,
        'shuffle_valid':False,
        'workers':args.workers,
        'max_len':args.max_len,
        'min_len':args.min_len,
        'skip':args.skip,
        'multiexp_generator_type':args.sampler,
        'preprocess':'rodan',
    }
    return data_params



def get_model_config(args): 
    with open(args.arch_hyperparams_yaml, 'r') as file:
        arch_hyperparams = yaml.safe_load(file)
    
    model_params = {
        'arch':arch_map[args.arch],
        'lr':args.lr,
        'warmup_steps':args.warmup_steps,
        'pos_weight':args.pos_weight,
        'wd':args.wd,
        **arch_hyperparams,
    }
    return model_params

def get_training_config(args):
    training_params = {
        'grad_accumulation':args.grad_acc,
        'accelerator':'gpu',
        "early_stopping_metric":"2022 may valid auroc",
        'early_stopping_patience':args.early_stopping_patience,

    }
    return training_params

def get_logging_config(args):
    prog_bar_map = {
        'yes':True,
        'no':False,
    }
    logging_params = {
        'experiment_name':args.experiment_name,
        'logging_step':args.logging_step,
        'api_key':args.comet_api_key,
        'project_name':args.comet_project_name,
        "model_save_path":args.save_path,
        'enable_progress_bar':prog_bar_map[args.enable_progress_bar],
    }
    return logging_params

def train_save(datasets, model_params, data_params, training_params, logging_params):
    arch = model_params.pop('arch')
    model = arch(
        **model_params, 
        logging_steps=logging_params['logging_step']*training_params['grad_accumulation'],
    )

    dm = TrainingDatamodule(
        **datasets,
        **data_params,
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{logging_params['model_save_path']}/{logging_params['experiment_name']}", 
        save_top_k=1, 
        monitor=training_params['early_stopping_metric'], 
        filename='best-{step}-{'+training_params['early_stopping_metric'].replace(' ','_')+':.2f}',
        mode='max',
        save_last=True, 
        save_weights_only=False
    )
    early_stopping_callback = EarlyStopping(
        monitor=training_params['early_stopping_metric'], 
        mode="max", 
        patience=training_params['early_stopping_patience'],
        verbose=True,
    )
    callbacks = [checkpoint_callback, early_stopping_callback]
    #TODO am i fallbacking to the best model?
    logger = CometLogger(
        api_key=logging_params['api_key'], 
        project_name=logging_params['project_name'], 
        experiment_name=logging_params['experiment_name'],
    ) 
    
    val_logging_step = logging_params['logging_step']*training_params['grad_accumulation']
    
    trainer= pl.Trainer(
        max_steps = 10000000, logger=logger, accelerator=training_params['accelerator'],
        auto_lr_find=False, val_check_interval=val_logging_step,  
        log_every_n_steps=logging_params['logging_step'], benchmark=False, precision=16,
        callbacks=callbacks, 
        accumulate_grad_batches=training_params['grad_accumulation'],
        resume_from_checkpoint=None,
        enable_progress_bar  = logging_params['enable_progress_bar'],
        # ckpt_path=f'/home/jovyan/RNAModif/rnamodif/checkpoints_pl/{experiment_name}/last.ckpt'
    )

    trainer.fit(model, dm)

def main(args):
    train_save(
        datasets=get_datasets_config(args), 
        model_params=get_model_config(args), 
        data_params=get_dataloading_config(args), 
        training_params=get_training_config(args), 
        logging_params=get_logging_config(args),
    )
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run training')
    parser = parse_args(parser)
    args = parser.parse_args()
    main(args)