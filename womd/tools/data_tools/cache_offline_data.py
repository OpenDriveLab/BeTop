'''
Behavioral Topology (BeTop): https://arxiv.org/abs/2409.18031
'''
'''
Mostly from Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
'''

import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_sched
from tensorboardX import SummaryWriter

from betopnet.datasets import build_dataloader
from betopnet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from betopnet.utils import common_utils
from betopnet.models import model as model_utils

from train_utils.train_utils import train_model
import numpy as np
from tqdm import tqdm


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')

    parser.add_argument('--cache_path', type=str, default=None, help='specify the path to cache offline data')

    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--without_sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=2, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=None, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=5, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--not_eval_with_train', action='store_true', default=False, help='')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')

    parser.add_argument('--add_worker_init_fn', action='store_true', default=False, help='')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def build_collection_data(data_dict, save_path):
    ret_dict = data_dict['input_dict']
    batch = ret_dict['scenario_id'].shape[0]
    for b in range(batch):
        save_dict = dict()
        for k in ret_dict.keys():
            if k in ['obj_types', 'obj_ids', 'center_objects_id']:
                continue
            if k in ['scenario_id', 'center_objects_type']:
                save_dict[k] = np.array(ret_dict[k][b])
            else:
                save_dict[k] = ret_dict[k][b].cpu().numpy()
        s_id = ret_dict['scenario_id'][b]
        t2pred = save_dict['track_index_to_predict']

        save_name = save_path + f'/{s_id}_{t2pred}.npz'
        np.savez(save_name, **save_dict)


def main():
    args, cfg = parse_config()

    output_dir = Path('./')

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        batch_size=args.batch_size,
        dist=True, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        add_worker_init_fn=args.add_worker_init_fn,
    )

    logger.info('******************End Loading %s/%s(%s), start Caching Data****************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    save_path = args.cache_path
    
    for data in tqdm(train_loader):
        build_collection_data(data, save_path)
    
    logger.info('**********************End Cache %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
