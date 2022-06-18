import os, csv
import time
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import sys
from collections import defaultdict
from easydict import EasyDict as edict
import wilds
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper
import wandb

from utils import set_seed, Logger, BatchLogger, log_config, ParseKwargs, load, log_group_data, \
    parse_bool, get_model_prefix
from train_script import train, evaluate
from algorithms.initializer import initialize_algorithm
from transforms import initialize_transform
from configs.utils import populate_defaults
import configs.supported as supported

import torch.multiprocessing

def main():
    config_ours = dict(
        dataset='camelyon17', # [poverty, camelyon17, iwildcam, civilcomments,amazon, fmow, rxrx1]
        algorithm='ladg',
        warmupFeaSteps=200,
        warmupDiscSteps=100,

        weightAdv=0.1,
        tmpScale=2,
        lprestartrate=0.8,
        numIterDisc=3,
        discModel='gatedgcn',
        latentDimDisc=256,
        domainMethod='lp',
        topk=10,
        # mcr
        weightMCR = 1,
        tolFactor = 0.2,
        mcr='all',
        coefMCRUpdate = 0.99,
        random_seed = 1, # to load pretrained model with different seed
        fold = 'B', # only for poverty map
        batch_size = 120,
        n_groups_per_batch = 3,
    )

    wandb.init(project=f"ladg")
    wandb.config.update(config_ours)
    argsWandb = wandb.config
    dataset = argsWandb.dataset
    config = edict(algo_log_metric=None, algorithm='ladg', batch_size=None, coral_penalty_weight=None, dataset=dataset,
              dataset_kwargs={}, distinct_groups=None, download=False, eval_epoch=None, eval_loader='standard',
              eval_only=False, eval_splits=[], evaluate_all_splits=True, frac=1.0, gpu=0, group_dro_step_size=None,
              groupby_fields=None, irm_lambda=None, irm_penalty_anneal_iters=None, loader_kwargs={},
              log_dir='../../../logs', log_every=50, loss_function=None, loss_kwargs={}, lr=None, max_grad_norm=None,
              max_token_length=None, model=None, model_kwargs={}, n_epochs=None, n_groups_per_batch=None,
              no_group_logging=None, optimizer=None, optimizer_kwargs={}, process_outputs_function=None,
              progress_bar=True, resize_scale=None, resume=False, root_dir='../data/wild',
              save_best=True, save_last=True, save_pred=True, save_step=None, scheduler=None, scheduler_kwargs={},
              scheduler_metric_name=None, scheduler_metric_split='val', seed=0, split_scheme=None,
              target_resolution=None, train_loader='group', transform=None, uniform_over_groups=None, use_wandb=True,
              val_metric=None, val_metric_decreasing=None, version=None, weight_decay=None)

    config = populate_defaults(config)

    config.seed = argsWandb.random_seed

    if dataset == 'poverty':
        config.dataset_kwargs['fold'] = argsWandb.fold

    # for pretraining, we use standard dataloader for these datasets

    # if dataset in ['civilcomments','fmow','iwildcam','amazon']:
    #     config.train_loader='standard'
    #     config.uniform_over_groups=False
    # else:
    #     config.train_loader='group'

    config.batch_size = argsWandb.batch_size
    config.n_groups_per_batch = argsWandb.n_groups_per_batch


    config.log_every = 50
    # Set device
    config.device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")
    for k, v in argsWandb.items():
        if k in ['batch_size', 'n_groups_per_batch']:
            continue
        config[k] = v

    if os.path.exists(config.log_dir) and config.resume:
        resume = True
        mode = 'a'
    elif os.path.exists(config.log_dir) and config.eval_only:
        resume = False
        mode = 'a'
    else:
        resume = False
        mode = 'w'

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    logger = Logger(os.path.join(config.log_dir, 'log.txt'), mode)

    # Record config
    log_config(config, logger)

    # Set random seed
    set_seed(config.seed)

    # Data
    full_dataset = wilds.get_dataset(
        dataset=config.dataset,
        version=config.version,
        root_dir=config.root_dir,
        download=config.download,
        split_scheme=config.split_scheme,
        **config.dataset_kwargs)

    train_transform = initialize_transform(
        transform_name=config.transform,
        config=config,
        dataset=full_dataset,
        is_training=True)
    eval_transform = initialize_transform(
        transform_name=config.transform,
        config=config,
        dataset=full_dataset,
        is_training=False)

    train_grouper = CombinatorialGrouper(
        dataset=full_dataset,
        groupby_fields=config.groupby_fields)

    datasets = defaultdict(dict)
    for split in full_dataset.split_dict.keys():
        if split == 'train':
            transform = train_transform
            verbose = True
        elif split == 'val':
            transform = eval_transform
            verbose = True
        else:
            transform = eval_transform
            verbose = True
        # Get subset
        datasets[split]['dataset'] = full_dataset.get_subset(
            split,
            frac=config.frac,
            transform=transform)

        if split == 'train':
            datasets[split]['loader'] = get_train_loader(
                loader=config.train_loader,
                dataset=datasets[split]['dataset'],
                batch_size=config.batch_size,
                uniform_over_groups=config.uniform_over_groups,
                grouper=train_grouper,
                distinct_groups=config.distinct_groups,
                n_groups_per_batch=config.n_groups_per_batch,
                **config.loader_kwargs)
        else:
            datasets[split]['loader'] = get_eval_loader(
                loader=config.eval_loader,
                dataset=datasets[split]['dataset'],
                grouper=train_grouper,
                batch_size=config.batch_size,
                **config.loader_kwargs)

        # Set fields
        datasets[split]['split'] = split
        datasets[split]['name'] = full_dataset.split_names[split]
        datasets[split]['verbose'] = verbose

        # Loggers
        datasets[split]['eval_logger'] = BatchLogger(
            os.path.join(config.log_dir, f'{split}_eval.csv'), mode=mode, use_wandb=(config.use_wandb and verbose))
        datasets[split]['algo_logger'] = BatchLogger(
            os.path.join(config.log_dir, f'{split}_algo.csv'), mode=mode, use_wandb=(config.use_wandb and verbose))

    if config.no_group_logging and full_dataset.is_classification and full_dataset.y_size == 1 and full_dataset.n_classes <= 10:
        log_grouper = CombinatorialGrouper(
            dataset=full_dataset,
            groupby_fields=['y'])
    elif config.no_group_logging:
        log_grouper = None
    else:
        log_grouper = train_grouper
    log_group_data(datasets, log_grouper, logger)

    ## Initialize algorithm
    algorithm = initialize_algorithm(
        config=config,
        datasets=datasets,
        train_grouper=train_grouper)

    datasetName = config.dataset
    loadpretrain = ['fmow', 'civilcomments', 'iwildcam', 'amazon']

    if datasetName in loadpretrain:
        if datasetName == 'poverty':
            idModel = config.fold
        else:
            idModel = config.random_seed
        if datasetName == 'fmow':
            save_path = '../../../model/standard_model_' + datasetName + '_' + str(idModel) + '_epoch_3.pt'
        elif datasetName == 'civilcomments':
            save_path = '../../../model/standard_model_' + datasetName + '_' + str(idModel) + '_epoch_0.pt'
        elif datasetName == 'iwildcam':
            save_path = '../../../model/standard_model_' + datasetName + '_' + str(idModel) + '_epoch_0.pt'
        elif datasetName == 'amazon':
            save_path = '../../../model/standard_model_' + datasetName + '_' + str(idModel) + '_epoch_0.pt'
        else:
            raise ValueError('not implemented')
        model_saved = torch.load(save_path, map_location=config.device)
        savedparams = model_saved['algorithm']
        pretrained_dict = {k: v for k, v in savedparams.items() if not k.startswith('disc')}
        algorithm.load_state_dict(pretrained_dict, strict=False)

        algorithm.opt_cls.load_state_dict(model_saved['optcls'])
        algorithm.opt_disc.load_state_dict(model_saved['optdisc'])
        algorithm.opt_fea.load_state_dict(model_saved['optfea'])
        logger.write(save_path +' model loaded')

        pretrainEpoch = {
            'civilcomments': 1,
            'amazon': 1,
            'iwildcam':1,
            'fmow':4,
        }
        epoch_offset = pretrainEpoch[datasetName]
    else:
        epoch_offset = 0

    best_val_metric = None
    config.args = config
    logger.write('********************** start training ************************')
    train(
            algorithm=algorithm,
            datasets=datasets,
            general_logger=logger,
            config=config,
            epoch_offset=epoch_offset,
            best_val_metric=best_val_metric
    )


    logger.close()
    for split in datasets:
        datasets[split]['eval_logger'].close()
        datasets[split]['algo_logger'].close()


if __name__ == '__main__':
    main()
