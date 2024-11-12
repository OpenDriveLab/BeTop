'''
Behavioral Topology (BeTop): https://arxiv.org/abs/2409.18031
'''

'''
Mostly from Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
'''

import pickle
import time

import numpy as np
import torch
import tqdm

from betopnet.utils import common_utils
import os


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False,
     save_to_file=False, result_dir=None, logger_iter_interval=50,
    joint_pred=False, final_output_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    if final_output_dir is None:
        final_output_dir = result_dir / 'final_result' / 'data'
        if save_to_file:
            final_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        if save_to_file:
            os.makedirs(final_output_dir, exist_ok=True)

    dataset = dataloader.dataset

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            num_gpus = torch.cuda.device_count()
            local_rank = cfg.LOCAL_RANK % num_gpus
            model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[local_rank],
                    broadcast_buffers=False
            )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()

    pred_dicts = []
    for i, batch_dict in enumerate(dataloader):
        with torch.no_grad():
            batch_pred_dicts = model(batch_dict)
            final_pred_dicts = dataset.generate_prediction_dicts(batch_pred_dicts, 
                joint_pred=joint_pred, output_path=final_output_dir if save_to_file else None)
            pred_dicts += final_pred_dicts

        disp_dict = {}

        if cfg.LOCAL_RANK == 0 and (i % logger_iter_interval == 0 or i == 0 or i + 1== len(dataloader)):
            past_time = progress_bar.format_dict['elapsed']
            second_each_iter = past_time / max(i, 1.0)
            remaining_time = second_each_iter * (len(dataloader) - i)
            disp_str = ', '.join([f'{key}={val:.3f}' for key, val in disp_dict.items() if key != 'lr'])
            batch_size = batch_dict.get('batch_size', None)
            logger.info(f'eval: epoch={epoch_id}, batch_iter={i}/{len(dataloader)}, batch_size={batch_size}, iter_cost={second_each_iter:.2f}s, '
                        f'time_cost: {progress_bar.format_interval(past_time)}/{progress_bar.format_interval(remaining_time)}, '
                        f'{disp_str}')

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        logger.info(f'Total number of samples before merging from multiple GPUs: {len(pred_dicts)}')
        pred_dicts = common_utils.merge_results_dist(pred_dicts, len(dataset), tmpdir=result_dir / 'tmpdir')
        if pred_dicts is not None:
            logger.info(f'Total number of samples after merging from multiple GPUs (removing duplicate): {len(pred_dicts)}')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0 or (pred_dicts is None):
        return {}

    ret_dict = {}

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(pred_dicts, f)

    result_str, result_dict = dataset.evaluation(
        pred_dicts,
        output_path=final_output_dir, 
        joint_pred=joint_pred
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')

    return ret_dict

def get_top_k_inter(batch_dict, k=6):
    gt_trajs = batch_dict['center_gt_trajs'].cuda()
    gt_valid_mask = batch_dict['center_gt_trajs_mask'].cuda()

    gt_trajs = gt_trajs * gt_valid_mask[..., None].float()
    tgt_trajs = batch_dict['obj_trajs_future_state'].cuda()
    tgt_trajs_mask = batch_dict['obj_trajs_future_mask'].cuda()
    tgt_trajs = tgt_trajs * tgt_trajs_mask[..., None].float()

    actor_occ_mask = torch.any(tgt_trajs_mask, dim=-1)[:, :]

    track_index_to_predict = batch_dict['track_index_to_predict'].cuda()

    new_indice = track_index_to_predict.reshape(-1, 2)[:, [1, 0]].reshape(-1)

    idle_occ = torch.zeros_like(actor_occ_mask)
    idle_occ.scatter_(1, new_indice.unsqueeze(-1), 1.)

    dist = torch.linalg.norm(tgt_trajs[:, :, 0, :2] - gt_trajs[:, None, 0, :2], dim=-1)
    dist[~actor_occ_mask] = 1000
    _, inds = torch.topk(dist, k=k, largest=False, dim=-1)

    ego_occ = torch.zeros_like(actor_occ_mask)
    ego_occ.scatter_(1, inds, 1.)
    return ego_occ, idle_occ, actor_occ_mask



if __name__ == '__main__':
    pass
