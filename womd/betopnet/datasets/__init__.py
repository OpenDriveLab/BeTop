'''
Behavioral Topology (BeTop): https://arxiv.org/abs/2409.18031
'''
'''
Mostly from Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
'''


import numpy as np
import torch
from torch.utils.data import DataLoader
from betopnet.utils import common_utils

from .waymo.waymo_dataset import BeTopWaymoDataset
from .dataset import BeTopWaymoCacheDataset


__all__ = {
    'BeTopWaymoDataset': BeTopWaymoDataset,
    "BeTopWaymoCacheDataset":BeTopWaymoCacheDataset
}


def build_dataloader(dataset_cfg, batch_size, dist, workers=4,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, 
                     total_epochs=0, add_worker_init_fn=False,
                     inter_pred=False, test=False):
    
    def worker_init_fn_(worker_id):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2 ** 32 - 1
        np.random.seed(np_seed)
    
    finetune = False
    if training and (finetune==False):
        dataset = __all__['BeTopWaymoCacheDataset'](
            dataset_cfg=dataset_cfg,
            training=training,
            logger=logger, 
        )
    else:
        dataset = __all__[dataset_cfg.DATASET](
            dataset_cfg=dataset_cfg,
            training=training,
            logger=logger, 
            inter_pred=inter_pred,
            test=test
        )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None

    drop_last = dataset_cfg.get('DATALOADER_DROP_LAST', False) and training

    if training or inter_pred:
        input_batch_size = batch_size
    else:
       input_batch_size =  4
    dataloader = DataLoader(
        dataset, batch_size=input_batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
        drop_last=drop_last, sampler=sampler, timeout=0, 
        worker_init_fn=worker_init_fn_ if add_worker_init_fn and training else None
    )

    return dataset, dataloader, sampler
