'''
Behavioral Topology (BeTop): https://arxiv.org/abs/2409.18031
'''

import torch
import csv
import argparse
import torch.functional as F
from torch.utils.data import DataLoader
import time

import sys
from shapely.geometry import LineString
from shapely.affinity import affine_transform, rotate
from tqdm import tqdm

import tarfile

import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from betopnet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from betopnet.models import build_model
from betopnet.utils import common_utils, motion_utils
from betopnet.datasets import build_dataloader

from easydict import EasyDict

from waymo_open_dataset.protos.motion_submission_pb2 import *


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--sub_output_dir', type=str, default=None, help='path to save output submission file')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--output_dir', type=str, default=None, help='output directory for submission')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--eval', action='store_true', default=False, help='')
    parser.add_argument('--interactive', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def save_to_file(scnerio_predictions, args, save_name, submission_info):
    print('saving....')
    submission_type=2 if args.interactive else 1

    submission = MotionChallengeSubmission(
        account_name=submission_info['account_name'], 
        unique_method_name=submission_info['unique_method_name'],
        authors=submission_info['authors'], 
        affiliation=submission_info['affiliation'], 
        submission_type=submission_type, 
        scenario_predictions=scnerio_predictions,
        uses_lidar_data=submission_info['uses_lidar_data'],
        uses_camera_data=submission_info['uses_camera_data'],
        uses_public_model_pretraining=submission_info['uses_public_model_pretraining'],
        public_model_names=submission_info['public_model_names'],
        num_model_parameters=submission_info['num_model_parameters']
        )

    save_path = args.output_dir + f"{save_name}.proto"
    tar_path = args.output_dir + f"{save_name}.gz"
    f = open(save_path, "wb")
    f.write(submission.SerializeToString())
    f.close()

    print(f'Testing_saved:{tar_path},zipping...')

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(save_path)
        tar.close()

    print('Finished!')

def traj_serialize(trajectories, scores, object_ids):
    scored_obj_trajs = []
    for i in range(trajectories.shape[0]):
        center_x, center_y = trajectories[i, 4::5, 0], trajectories[i, 4::5, 1]
        traj = Trajectory(center_x=center_x, center_y=center_y)
        object_traj = ScoredTrajectory(confidence=scores[i], trajectory=traj)
        scored_obj_trajs.append(object_traj)
    return SingleObjectPrediction(trajectories=scored_obj_trajs, object_id=object_ids)


def serialize_single_scenario(scenario_list):
    single_prediction_list = []
    scenario_id = scenario_list[0]['scenario_id']
    for single_dict in scenario_list:
        sc_id = single_dict['scenario_id']
        assert sc_id == scenario_id
        single_prediction = traj_serialize(single_dict['pred_trajs'],
            single_dict['pred_scores'], single_dict['object_id'])
        single_prediction_list.append(single_prediction)
    prediction_set = PredictionSet(predictions=single_prediction_list)
    return ChallengeScenarioPredictions(scenario_id=scenario_id, single_predictions=prediction_set)

def joint_serialize_single_scenario(scenario_list):
    assert len(scenario_list)==2
    scenario_id = scenario_list[0]['scenario_id']
    joint_score = scenario_list[0]['pred_scores']
    full_scored_trajs = []
    for j in range(6):
        object_trajs = []
        for i in range(2):
            center_x = scenario_list[i]['pred_trajs'][j, 4::5, 0]
            center_y = scenario_list[i]['pred_trajs'][j, 4::5, 1]
            traj = Trajectory(center_x=center_x, center_y=center_y)
            score_traj = ObjectTrajectory(object_id=scenario_list[i]['object_id'], trajectory=traj)
            object_trajs.append(score_traj)   
        full_scored_trajs.append(ScoredJointTrajectory(trajectories=object_trajs, confidence=joint_score[j]))
    joint_prediction = JointPrediction(joint_trajectories=full_scored_trajs)
    return ChallengeScenarioPredictions(scenario_id=scenario_id, joint_prediction=joint_prediction)

def serialize_single_batch(final_pred_dicts, scenario_predictions, joint_pred=False):
    for b in range(len(final_pred_dicts)):
        scenario_list = final_pred_dicts[b]
        if joint_pred:
            scenario_preds = joint_serialize_single_scenario(scenario_list)
        else:
            scenario_preds = serialize_single_scenario(scenario_list)
        scenario_predictions.append(scenario_preds)
    return scenario_predictions


def test(model, test_set, test_data, args, joint_pred):
    scenario_predictions = []
    size = len(test_data)*args.batch_size
    with torch.no_grad():
        for i, batch_dict in enumerate(test_data):

            batch_pred_dicts = model(batch_dict)
            final_pred_dicts = test_set.generate_prediction_dicts(batch_pred_dicts, 
                                    joint_pred=joint_pred, output_path=None, submission=True)
                                    
            scenario_predictions = serialize_single_batch(final_pred_dicts, 
                scenario_predictions, joint_pred)
            
            sys.stdout.write(f'\rProcessed:{i*args.batch_size}-{size}')
            sys.stdout.flush()

    return scenario_predictions    
        

def main():

    args, cfg = parse_config()

    submission_info = dict(
        account_name='your Waymo account email',
        unique_method_name='xxx',
        authors=['A', 'B', 'xxx'],
        affiliation='your affiliation',
        uses_lidar_data=False,
        uses_camera_data=False,
        uses_public_model_pretraining=False,
        public_model_names='N/A',
        num_model_parameters='N/A',
    )

    test_interactive = args.interactive
    if args.eval:
        print('evaluating submission BeTop...')
        log_prefix = 'eval_'
        test_sub=False
    else:
        print('evaluating submission BeTop...')
        log_prefix = 'test_'
        test_sub=True

    method_name = args.cfg_file.split('/')[-1].split('.')[0]


    if test_interactive:
        log_prefix += 'interactive_'
    log_file = args.sub_output_dir + log_prefix +\
         method_name + '%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        batch_size=args.batch_size,
        workers=args.workers, logger=logger, training=False,
        inter_pred=test_interactive, dist=False, test=test_sub
    )

    model = build_model(config=cfg.MODEL)
    
    it, epoch = model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    scenario_predictions = test(model, test_set, test_loader, args, test_interactive)
    save_to_file(scenario_predictions, args, log_prefix + method_name, submission_info)



if __name__ == '__main__':
    main()
