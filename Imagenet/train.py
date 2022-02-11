""" ImageNet Training Script used for the paper of Coded-ResNeXt
It is a modified version from: https://github.com/rwightman/pytorch-image-models/tree/bits_and_tpu 
The changes we incorporated are described like:
#===================
#blah blah blah
#===================
"""
import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from datetime import datetime
from dataclasses import replace
from typing import Tuple
from copy import deepcopy

import torch
import torch.nn as nn
import torchvision.utils

from timm.bits import initialize_device, setup_model_and_optimizer, DeviceEnv, Monitor, Tracker,\
    TrainState, TrainServices, TrainCfg, CheckpointManager, AccuracyTopK, AvgTensor, distribute_bn
from timm.data import create_dataset, create_transform_v2, create_loader_v2, resolve_data_config,\
    PreprocessCfg, AugCfg, MixupCfg, AugMixDataset
from timm.models import create_model, safe_model_name, convert_splitbn_model
from timm.loss import *
from timm.optim import optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import setup_default_logging, random_seed, get_outdir, unwrap_model

from CodedResNeXt import create_CodedResNeXt
import pickle, os, random

_logger = logging.getLogger('train')


# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

#The arguments we added/modified for Coded ResNeXt
parser.add_argument('--model', default='Coded-ResNeXt-50', type=str, metavar='MODEL',
                    help='Name of model to train (default: "Coded-ResNeXt-50"')
parser.add_argument('--Control', action='store_true', default=False,
                    help='Use baseline architecture (ResNeXt)')
parser.add_argument('--Energy-normalization', action='store_false', default=True,
                    help='Use of Energy Normalization step (default: True)')
parser.add_argument('--Mask-grads', action='store_true', default=False,
                    help='Mask the gradients according to the coding scheme. Could be used instead of coding loss (default: False)')
parser.add_argument('--Same-code-Same-mask', action='store_false', default=True,
                    help= 'If TRUE then every two consecutive ResNeXt blocks with the same coding scheme will \n' + \
                          'also have the same dropout mask applied to them. Therefore out of N consecutive ResNeXt blocks \n' +\
                          'with the same coding scheme it will be the first one drop_out_probability that counts (default: True)')
parser.add_argument('--LossDisentangle-type', type=str, default='power4_threshold0.0',
                    help='Coding Loss = diff(Energy_subNN, target_Energy, threshold)^power. The function diff  \n'+\
                         'is: max{ |Energy_subNN-target_Energy|-threshold, 0} (default: power4_threshold0.0)')
parser.add_argument('--Coef-LossDisentangle', type=float, default=1.0,
                    help='The coefficient of the coding loss. In the paper it is denoted with $\mu$ (default: 1.0)')
parser.add_argument('--dp-prob', type=float, default=0.1,
                    help='The dropSubNN probability. In the paper it is denoted with $p_{drop}$ (default: 0.1)')
parser.add_argument('--coding-ratio-per-stage', nargs= 4, type=str, default = ['32/32', '32/32', '16/32', '8/32'],
                    help='The ratios of the coding schemes that are applied to all blocks of each stage. Assumption of using \n'+ \
                         'Coded-ResNeXt which has 4 stages, so 4 inputs should be given. Default: 32/32 32/32 16/32 8/32.' )
parser.add_argument('--only-eval',  action='store_true', default=False,
                    help='If true then it does only one evaluation step. A checkpoint should be provided \n' +\
                    'for loading the network (default: False)')
parser.add_argument('--Remove-subNNs-from-block',  action='store_true', default=False,
                    help='Used for generating the plot where subNNs are removed either from the set of active or  \n'+\
                    'or from the set of inactive. Calls the function "test_print_acc_removing_subNNs" where the first \n'+\
                    'two lines define from which block and how many subNNs will be removed. The default block is the 14th. \n'+\
                    'A checkpoint should be provided for loading the network. (default: False)')
parser.add_argument('--BinaryClassifier', type=int, default=-1, 
                    help="Give the class for which its binary classifier will be evaluating. The output of the binary \n"+\
                    'for every image in the evaluation set will be stored in a folder in the output path.')     


parser.add_argument('--output', default='/', type=str, metavar='PATH',
                    help='path to output folder (default my drive)')
parser.add_argument('--experiment', default='new_experiment', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')


# Dataset parameters
parser.add_argument('data_dir', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')

# Model parameters
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                    help='validation batch size override (default: None)')


# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')


# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "cosine"')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.05)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                    help='amount to decay each learning rate cycle (default: 0.5)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit, cycles enabled if > 1')
parser.add_argument('--lr-k-decay', type=float, default=1.0,
                    help='learning rate k-decay for cosine/poly (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd-loss', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--bce-loss', action='store_true', default=False,
                    help='Enable BCE loss w/ Mixup/CutMix use.')
parser.add_argument('--bce-target-thresh', type=float, default=None,
                    help='Threshold for binarizing softened BCE targets (default: None, disabled)')
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='reduce',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')                    
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--force-cpu', action='store_true', default=False,
                    help='Force CPU to be used even if HW accelerator exists.')
parser.add_argument('--log-wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')





def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    setup_default_logging()
    args, args_text = _parse_args()

    dev_env = initialize_device(force_cpu=args.force_cpu, amp=args.amp, channels_last=args.channels_last)

    #================================
    #Printing relevant information.
    #================================
    if dev_env.primary:
        print('===============================')
        print('Your arguments for the experiment -',args.experiment,'- are:')
        print('     Control: ',args.Control)
        print('     Energy_normalization: ',args.Energy_normalization)
        print('     Same_code_Same_mask: ',args.Same_code_Same_mask) 
        print('     LossDisentangle_type: ',args.LossDisentangle_type)       
        print('     Coef_LossDisentangle: ',args.Coef_LossDisentangle)
        print('     Dropout SubNN prob. : ',args.dp_prob)
        print('     Mask_grads: ', args.Mask_grads)
        print('     Coding Ratios per stage:', args.coding_ratio_per_stage)
        print('Saving at directory: ', args.output)
        print('And the experiment name: ', args.experiment)
        print('===============================')

    if args.Remove_subNNs_from_block or (args.BinaryClassifier>=0):
        args.only_eval = True
    #================================



    if dev_env.distributed:
        _logger.info('Training in distributed mode with multiple processes, 1 device per process. Process %d, total %d.'
                     % (dev_env.global_rank, dev_env.world_size))
    else:
        _logger.info('Training with a single process on 1 device.')

    random_seed(args.seed, 0)  # Set all random seeds the same for model/state init (mandatory for XLA)

    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    assert args.aug_splits == 0 or args.aug_splits > 1, 'A split of 1 makes no sense'

    train_state = setup_train_task(args, dev_env, mixup_active)
    train_cfg = train_state.train_cfg

    # Set random seeds across ranks differently for train
    random_seed(args.seed, dev_env.global_rank)

    data_config, loader_eval, loader_train = setup_data(
        args,
        unwrap_model(train_state.model).default_cfg,
        dev_env,
        mixup_active)

    # setup checkpoint manager
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    checkpoint_manager = None
    output_dir = None
    if dev_env.primary:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = get_outdir(args.output if args.output else './output/train', exp_name)
        checkpoint_manager = CheckpointManager(
            hparams=vars(args),
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            metric_name=eval_metric,
            metric_decreasing=True if eval_metric == 'loss' else False,
            max_history=args.checkpoint_hist)                    
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)
            

    services = TrainServices(
        monitor=Monitor(
            output_dir=output_dir,
            logger=_logger,
            hparams=vars(args),
            output_enabled=dev_env.primary),
        checkpoint=checkpoint_manager,
        )

    #================================
    #Some additional functionality. 
    #Mostly used to get the necessary 
    #data for the plots of the paper.
    #================================
    try:
        if args.only_eval:
            if args.Remove_subNNs_from_block:
                test_print_acc_removing_subNNs(
                        train_state.model,
                        train_state.eval_loss,
                        loader_eval,
                        services.monitor,
                        dev_env)
            elif args.BinaryClassifier >= 0:
                Name_process = random.randint(0, 100000)#To give to each process a different number id.
                    #There is a probability of 99.97% to get all different id numbers. If two processes 
                    #have identical id number increase the 100000.
                print(Name_process)
                BinaryClassifierOneClass(
                        train_state.model,
                        loader_eval,
                        args.output,
                        dev_env,
                        Name_process,
                        args.BinaryClassifier)   
            else:
                eval_metrics = evaluate(
                        train_state.model,
                        train_state.eval_loss,
                        loader_eval,
                        services.monitor,
                        dev_env)
                if services.monitor is not None:
                    services.monitor.write_summary( results=dict( eval=eval_metrics ) )
            return            
    except KeyboardInterrupt:
        pass
    #===============================

    try:
        for epoch in range(train_state.epoch, train_cfg.num_epochs):
            if dev_env.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)
            if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
                if loader_train.mixup_enabled:
                    loader_train.mixup_enabled = False
            
            train_metrics = train_one_epoch(
                state=train_state,
                services=services,
                loader=loader_train,
                dev_env=dev_env,
                coef_loss_dis=args.Coef_LossDisentangle,
            )

            if dev_env.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if dev_env.primary:
                    _logger.info("Distributing BatchNorm running means and vars")
                distribute_bn(train_state.model, args.dist_bn == 'reduce', dev_env)

            eval_metrics = evaluate(
                train_state.model,
                train_state.eval_loss,
                loader_eval,
                services.monitor,
                dev_env)


            if train_state.model_ema is not None:
                if dev_env.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(train_state.model_ema, args.dist_bn == 'reduce', dev_env)

                ema_eval_metrics = evaluate(
                    train_state.model_ema.module,
                    train_state.eval_loss,
                    loader_eval,
                    services.monitor,
                    dev_env,
                    phase_suffix='EMA')
                eval_metrics = ema_eval_metrics

            if train_state.lr_scheduler is not None:
                # step LR for next epoch
                train_state.lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])
            
            if services.monitor is not None:
                services.monitor.write_summary(
                    index=epoch,
                    results=dict(train=train_metrics, eval=eval_metrics))

            if checkpoint_manager is not None:
                # save proper checkpoint with eval metric
                best_checkpoint = checkpoint_manager.save_checkpoint(train_state, eval_metrics)
                best_metric, best_epoch = best_checkpoint.sort_key, best_checkpoint.epoch

            train_state = replace(train_state, epoch=epoch + 1)
            

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def setup_train_task(args, dev_env: DeviceEnv, mixup_active: bool):
    #===================================
    #Creating the (Coded-)ResNeXt model
    #===================================
    model = create_CodedResNeXt(
        Energy_normalization=args.Energy_normalization,
        Mask_grads = args.Mask_grads,
        Same_code_Same_mask = args.Same_code_Same_mask,
        LossDisentangle_type = args.LossDisentangle_type,
        Control = args.Control,
        coding_ratio_per_stage = args.coding_ratio_per_stage,
        dropSubNNs_prob  = args.dp_prob,
        checkpoint_path = args.initial_checkpoint,
    )
    
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  

    if dev_env.primary:
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert args.aug_splits > 1
        model = convert_splitbn_model(model, max(args.aug_splits, 2))

    train_state = setup_model_and_optimizer(
        dev_env=dev_env,
        model=model,
        optimizer=args.opt,
        optimizer_cfg=optimizer_kwargs(cfg=args),
        clip_fn=args.clip_mode if args.clip_grad is not None else None,
        clip_value=args.clip_grad,
        model_ema=args.model_ema,
        model_ema_decay=args.model_ema_decay,
        resume_path=args.resume,
        resume_opt=not args.no_resume_opt,
        use_syncbn=args.sync_bn,
    )

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, train_state.updater.optimizer)
    if lr_scheduler is not None and train_state.epoch > 0:
        lr_scheduler.step(train_state.epoch)

    # setup loss function
    if args.jsd_loss:
        assert args.aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=args.aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        # smoothing is handled with mixup target transform
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    eval_loss_fn = nn.CrossEntropyLoss()
    dev_env.to_device(train_loss_fn, eval_loss_fn)

    if dev_env.primary:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    train_cfg = TrainCfg(
        num_epochs=num_epochs,
        log_interval=args.log_interval,
        recovery_interval=args.recovery_interval,
    )

    train_state = replace(
        train_state,
        lr_scheduler=lr_scheduler,
        train_loss=train_loss_fn,
        eval_loss=eval_loss_fn,
        train_cfg=train_cfg,
    )

    return train_state


def setup_data(args, default_cfg, dev_env: DeviceEnv, mixup_active: bool):
    data_config = resolve_data_config(vars(args), default_cfg=default_cfg, verbose=dev_env.primary)

    # create the train and eval datasets
    if args.only_eval:
        dataset_train = None
    else:
        dataset_train = create_dataset(
            name=args.dataset, root=args.data_dir, split=args.train_split, is_training=True,
            class_map=args.class_map,
            download=args.dataset_download,
            batch_size=args.batch_size,
            repeats=args.epoch_repeats)

    dataset_eval = create_dataset(
        name=args.dataset, root=args.data_dir, split=args.val_split, is_training=False,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size)

    # setup mixup / cutmix
    mixup_cfg = None
    if mixup_active:
        mixup_cfg = MixupCfg(
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            label_smoothing=args.smoothing, num_classes=args.num_classes)

    # wrap dataset in AugMix helper
    if args.aug_splits > 1 and not args.only_eval:
        dataset_train = AugMixDataset(dataset_train, num_splits=args.aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']

    if args.no_aug:
        train_aug_cfg = None
    else:
        train_aug_cfg = AugCfg(
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            ratio_range=args.ratio,
            scale_range=args.scale,
            hflip_prob=args.hflip,
            vflip_prob=args.vflip,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            num_aug_splits=args.aug_splits,
        )

    train_pp_cfg = PreprocessCfg(
        input_size=data_config['input_size'],
        interpolation=train_interpolation,
        crop_pct=data_config['crop_pct'],
        mean=data_config['mean'],
        std=data_config['std'],
        aug=train_aug_cfg,
    )

    # if using PyTorch XLA and RandomErasing is enabled, we must normalize and do RE in transforms on CPU
    normalize_in_transform = dev_env.type_xla and args.reprob > 0
    if args.only_eval:
        loader_train = None
    else:
        loader_train = create_loader_v2(
            dataset_train,
            batch_size=args.batch_size,
            is_training=True,
            pp_cfg=train_pp_cfg,
            mix_cfg=mixup_cfg,
            normalize_in_transform=normalize_in_transform,
            num_workers=args.workers,
            pin_memory=args.pin_mem,
            use_multi_epochs_loader=args.use_multi_epochs_loader
        )

    eval_pp_cfg = PreprocessCfg(
        input_size=data_config['input_size'],
        interpolation=data_config['interpolation'],
        crop_pct=data_config['crop_pct'],
        mean=data_config['mean'],
        std=data_config['std'],
    )

    eval_workers = args.workers
    if 'tfds' in args.dataset:
        eval_workers = min(2, args.workers)

    loader_eval = create_loader_v2(
        dataset_eval,
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        pp_cfg=eval_pp_cfg,
        normalize_in_transform=normalize_in_transform,
        num_workers=eval_workers,
        pin_memory=args.pin_mem,
    )
    return data_config, loader_eval, loader_train


def train_one_epoch(
        state: TrainState,
        services: TrainServices,
        loader,
        dev_env: DeviceEnv,
        coef_loss_dis,
    ):
    #=================================================================
    # Performing one training epoch. Added more metric trackers.  
    # The model must be a (Coded-)ResNeXt which has multiple outputs.
    #=================================================================
    tracker = Tracker()
    loss_class_meter = AvgTensor()
    loss_disTotal_meter = AvgTensor()
    List_loss_disentangle_meter = [AvgTensor() for _ in range(state.model.num_codedBlocks)]

    state.model.train()
    state.updater.reset()  # zero-grad

    step_end_idx = len(loader) - 1
    tracker.mark_iter()
    for step_idx, (sample, target) in enumerate(loader):
        tracker.mark_iter_data_end()

        with dev_env.autocast():
            output, loss_disentangle_total, Losses_disentangle, _ = state.model(sample, target)
            assert len(List_loss_disentangle_meter)==len(Losses_disentangle)
            loss_class = state.train_loss(output, target)
            loss = loss_class + coef_loss_dis*loss_disentangle_total

        state.updater.apply(loss)

        tracker.mark_iter_step_end()

        state.updater.after_step(
            after_train_step,
            state,
            services,
            dev_env,
            step_idx,
            step_end_idx,
            tracker,
            loss_class_meter,
            loss_disTotal_meter,
            List_loss_disentangle_meter,
            (output, loss_disentangle_total, Losses_disentangle, target, loss_class),
        )

        tracker.mark_iter()

    if hasattr(state.updater.optimizer, 'sync_lookahead'):
        state.updater.optimizer.sync_lookahead()

    results = OrderedDict([('loss', loss_class_meter.compute().item())])
    return results


def after_train_step(
        state: TrainState,
        services: TrainServices,
        dev_env: DeviceEnv,
        step_idx: int,
        step_end_idx: int,
        tracker: Tracker,
        loss_class_meter: AvgTensor,
        loss_disTotal_meter: AvgTensor,
        List_loss_disentangle_meter,
        tensors: Tuple[torch.Tensor, ...]
    ):
    """
    After the core loss / backward / gradient apply step, we perform all non-gradient related
    activities here including updating meters, metrics, performing logging, and writing checkpoints.

    Many / most of these operations require tensors to be moved to CPU, they shoud not be done
    every step and for XLA use they should be done via the optimizer step_closure. This function includes
    everything that should be executed within the step closure.

    """
    end_step = step_idx == step_end_idx

    with torch.no_grad():
        #=========================================================================
        # Tracking additional metrics. Commented the part of saving a checkpoint.
        #=========================================================================
        output, loss_disentangle_total, Losses_disentangle, target, loss_class =  tensors
        loss_class_meter.update(loss_class, output.shape[0])
        loss_disTotal_meter.update(loss_disentangle_total, output.shape[0])
        for loss_dis_meter, loss_dis in zip(List_loss_disentangle_meter, Losses_disentangle):
            loss_dis_meter.update(loss_dis, output.shape[0])


        if state.model_ema is not None:
            state.model_ema.update(state.model)

        state = replace(state, step_count_global=state.step_count_global + 1)
        cfg = state.train_cfg
        if services.monitor is not None and end_step or (step_idx + 1) % cfg.log_interval == 0:
            global_batch_size = dev_env.world_size * output.shape[0]
            loss_avg = loss_class_meter.compute()
            if services.monitor is not None:
                lr_avg = state.updater.get_average_lr()
                services.monitor.log_step(
                    'Train',
                    step_idx=step_idx,
                    step_end_idx=step_end_idx,
                    epoch=state.epoch,
                    loss=loss_avg.item(),
                    rate=tracker.get_avg_iter_rate(global_batch_size),
                    lr=lr_avg,
                )            
        '''
        if services.checkpoint is not None and cfg.recovery_interval and (
                end_step or (step_idx + 1) % cfg.recovery_interval == 0):
            services.checkpoint.save_recovery(state.epoch, batch_idx=step_idx)
        '''

        if state.lr_scheduler is not None:
            state.lr_scheduler.step_update(num_updates=state.step_count_global)


def evaluate(
        model: nn.Module,
        loss_fn: nn.Module,
        loader,
        logger: Monitor,
        dev_env: DeviceEnv,
        phase_suffix: str = '',
        log_interval: int = 10,
    ):
    
    #=========================================================
    # The model must be a (Coded-)ResNeXt which has multiple 
    # outputs. Additional tracking metrics are printed/logged.
    #=========================================================

    tracker = Tracker()
    losses_class_meter = AvgTensor()
    accuracy_m = AccuracyTopK()  
    loss_disTotal_meter = AvgTensor()
    List_loss_disentangle_meter = [AvgTensor() for _ in range(model.num_codedBlocks)]
    List_decodeEnergies_acc = [AccuracyTopK(topk=(1,)) for _ in range(model.num_codedBlocks)]

    model.eval()

    end_idx = len(loader) - 1
    tracker.mark_iter()
    with torch.no_grad():
        for step_idx, (sample, target) in enumerate(loader):
            tracker.mark_iter_data_end()
            last_step = step_idx == end_idx

            with dev_env.autocast():
                output, loss_disentangle_total, Losses_disentangle, DecodeEnergies = model(sample, target)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                loss_class = loss_fn(output, target)

            if dev_env.type_xla:
                dev_env.mark_step()
            elif dev_env.type_cuda:
                dev_env.synchronize()

            tracker.mark_iter_step_end()
            losses_class_meter.update(loss_class, output.size(0))
            accuracy_m.update(output, target)
            loss_disTotal_meter.update(loss_disentangle_total, output.size(0))            
            for loss_dis_meter, loss_dis_block in zip(List_loss_disentangle_meter, Losses_disentangle):
                loss_dis_meter.update(loss_dis_block, output.shape[0])
            for decodeEnergies_acc, decodeEnergies_block in zip(List_decodeEnergies_acc, DecodeEnergies):
                decodeEnergies_acc.update(decodeEnergies_block, target)

            if last_step or step_idx % log_interval == 0:
                top1, top5 = accuracy_m.compute().values()
                loss_class_avg = losses_class_meter.compute()
                logger.log_step(
                    'Eval',
                    step_idx=step_idx,
                    step_end_idx=end_idx,
                    loss=loss_class_avg.item(),
                    top1=top1.item(),
                    top5=top5.item(),
                    phase_suffix=phase_suffix,
                )   
            if last_step:             
                log_dictionary = {"val/loss_class":loss_class_avg.item()}
                log_dictionary["val/acc_top1"] = top1.item()
                log_dictionary["val/acc_top5"] = top5.item()
                log_dictionary['val/loss_dis_total'] = loss_disTotal_meter.compute().item()
                Additional_results = [('loss_dis_total', loss_disTotal_meter.compute().item())]
                for block_i, loss_dis_meter in enumerate(List_loss_disentangle_meter):
                    log_dictionary['val/loss_dis_'+str(block_i)] = loss_dis_meter.compute().item()  
                    Additional_results.append( ('loss_dis_'+str(block_i), loss_dis_meter.compute().item()) )
                for block_i, decodeEnergies_acc in enumerate(List_decodeEnergies_acc):
                    top1, = decodeEnergies_acc.compute().values()
                    log_dictionary['decodeEN_acc_'+str(block_i)] = top1.item()
                    Additional_results.append( ('decodeEN_acc_'+str(block_i),  top1.item()) )
                if dev_env.primary:
                    for k, v in log_dictionary.items():
                        print(k, ' : ',v)
            tracker.mark_iter()

    top1, top5 = accuracy_m.compute().values()
    results = OrderedDict([('loss', losses_class_meter.compute().item()), ('top1', top1.item()), ('top5', top5.item())])
    results.update(Additional_results)
    
    return results


def test_print_acc_removing_subNNs(
        model: nn.Module,
        loss_fn: nn.Module,
        loader,
        logger: Monitor,
        dev_env: DeviceEnv,
        phase_suffix: str = '',
        log_interval: int = 10,
    ):
    indx_block = 14
    List_N_subNNs_to_remove = [1,2,3,4,5,6,7,8]#[i+1 for i in range(1)]

    model.eval()
    with torch.no_grad():
        for remove_type in ['remove_inactive',  'remove_active']:       
            for k in List_N_subNNs_to_remove:    
                tracker = Tracker()
                losses_class_meter = AvgTensor()
                accuracy_m = AccuracyTopK()  # FIXME move loss and accuracy modules into task specific TaskMetric obj
                
                end_idx = len(loader) - 1
                tracker.mark_iter()

                for step_idx, (sample, target) in enumerate(loader):
                    tracker.mark_iter_data_end()
                    last_step = step_idx == end_idx

                    with dev_env.autocast():
                        output, _, _, _ = model(sample, target, mask_subNNs_scheme=(indx_block, k, remove_type))
                        if isinstance(output, (tuple, list)):
                            output = output[0]
                        loss_class = loss_fn(output, target)

                    if dev_env.type_xla:
                        dev_env.mark_step()
                    elif dev_env.type_cuda:
                        dev_env.synchronize()

                    tracker.mark_iter_step_end()
                    losses_class_meter.update(loss_class, output.size(0))
                    accuracy_m.update(output, target)

                    if last_step or step_idx % log_interval == 0:
                        top1, top5 = accuracy_m.compute().values()
                        loss_class_avg = losses_class_meter.compute()
                        logger.log_step(
                            'Eval',
                            step_idx=step_idx,
                            step_end_idx=end_idx,
                            loss=loss_class_avg.item(),
                            top1=top1.item(),
                            top5=top5.item(),
                            phase_suffix=phase_suffix,
                        )   
                        
                    tracker.mark_iter()
                top1, top5 = accuracy_m.compute().values()
                print('If we {0} {1} subNNs the accuracy is {2}'.format(remove_type, k, top1.item()))


def BinaryClassifierOneClass(
        model: nn.Module,
        loader,
        root_saving_path: str,
        dev_env: DeviceEnv,
        Name_process: int,
        class_for_BC: int = 0,
    ):

    model.eval()
    with torch.no_grad():     
        tracker = Tracker()
        tracker.mark_iter()

        ListOfOutput = []
        ListOfTargets = []
        for step_idx, (sample, target) in enumerate(loader):
            tracker.mark_iter_data_end()

            with dev_env.autocast():
                bin_class = class_for_BC*torch.ones_like(target) 
                output, _, _, _ = model(sample, bin_class, mask_subNNs_scheme=('all', 'remove_inactive'))  
                if isinstance(output, (tuple, list)):
                    output = output[0]
                output_BinaryClass = output[:, class_for_BC]
                ListOfOutput.append(output_BinaryClass.to('cpu'))
                ListOfTargets.append(target.to('cpu'))

            if dev_env.type_xla:
                dev_env.mark_step()
            elif dev_env.type_cuda:
                dev_env.synchronize()

            tracker.mark_iter_step_end()
            tracker.mark_iter()
        ListOfOutput = torch.cat(ListOfOutput)
        ListOfTargets = torch.cat(ListOfTargets)
        
    saving_path = os.path.join(root_saving_path, "BinaryClassifier/class_"+str(class_for_BC))
    os.makedirs(saving_path, exist_ok = True)
    with open(saving_path+'/core_'+ str(Name_process)+'.txt', "wb") as f:
        pickle.dump([ListOfOutput,ListOfTargets], f)

    '''
    #To plot for example the distribution something like the following should be run:
    Outputs, Labels, ActualPos, ActualNeg = [],[], [], []
    for f in glob.iglob('/content/drive/MyDrive/Imagenet-ResNeXt/checkpoints/BinaryClassifier/class_0/*.txt'):
        with open(f, "rb") as fp:  
            Out, label = pickle.load(fp)
            Outputs.append(Out)
            Labels.append(label)
    Outputs = torch.cat(Outputs)
    Labels = torch.cat(Labels)
    fig, axs = plt.subplots(1, 1, sharey=False, tight_layout=False, figsize = (10,3))
    for o, l in zip(Outputs, Labels):
        if l == cl:  ActualPos.append(o)
        else: ActualNeg.append(o)
    axs.hist(ActualPos, density=True, bins=15, alpha=0.8, label='In-distribution Positives',color='steelblue')
    axs.hist(ActualNeg , density=True, bins=40, alpha=0.6, label='In-distribution Negatives', color='brown')
    plt.show()
    '''

def _mp_entry(*args):
    main()


if __name__ == '__main__':
    main()
