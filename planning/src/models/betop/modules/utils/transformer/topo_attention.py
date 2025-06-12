'''
Behavioral Topology (BeTop): https://arxiv.org/abs/2409.18031
'''
'''
Mostly from EQNet: https://arxiv.org/abs/2203.01252
Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
'''

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

def agent_topo_indexing(
    agent_topo_pred, 
    obj_mask, 
    max_agents=192
    ):
    num_actors = obj_mask.shape[1]
    agent_topo_pred = agent_topo_pred.detach().sigmoid()

    agent_topo_pred = agent_topo_pred[..., 0]

    agent_topo_pred = agent_topo_pred * obj_mask.float()[:, None, :]
    top_dist, top_indice = torch.topk(agent_topo_pred, 
                        k=min(max_agents, num_actors), dim=-1)
    top_indice[top_dist==0] = -1

    if top_indice.shape[-1] < max_agents:
        top_indice = F.pad(
            top_indice, 
            pad=(0, max_agents - top_indice.shape[-1]), 
            mode='constant', value=-1
            )

    return top_indice.int()

def map_topo_indexing(
    map_pos, map_mask, 
    pred_waypoints, 
    base_region_offset, num_query, 
    num_waypoint_polylines=128, 
    num_base_polylines=256, 
    top_occ=128,
    base_map_idxs=None,
    map_topo=None):

    """
    function inherent from dynamic_map_collection in MTRDecoder
    applying further map topo indexing
    """
    
    map_pos = map_pos.clone()
    map_pos[~map_mask] = 10000000.0
    num_polylines = map_pos.shape[1]

    if map_topo is not None:
        _, map_topo_idx = map_topo.topk(
            k=min(num_polylines, top_occ), dim=-1)
        if map_topo_idx.shape[-1] < top_occ:
            map_topo_idx = F.pad(
                map_topo_idx, 
                pad=(0, top_occ - map_topo_idx.shape[-1]), 
                mode='constant', value=-1)

    if base_map_idxs is None:
        base_points = torch.tensor(base_region_offset).type_as(map_pos)
        # (num_center_objects, num_polylines)
        base_dist = (map_pos[:, :, 0:2] - base_points[None, None, :]).norm(dim=-1)  
        # (num_center_objects, topk)
        base_topk_dist, base_map_idxs = base_dist.topk(
            k=min(num_polylines, num_base_polylines), dim=-1, largest=False) 
        base_map_idxs[base_topk_dist > 10000000] = -1
        # (num_center_objects, num_query, num_base_polylines)
        base_map_idxs = base_map_idxs[:, None, :].repeat(1, num_query, 1)  
        if base_map_idxs.shape[-1] < num_base_polylines:
            base_map_idxs = F.pad(base_map_idxs,
                pad=(0, num_base_polylines - base_map_idxs.shape[-1]), 
                mode='constant', value=-1)

    # (num_center_objects, num_query, num_polylines, num_timestamps)
    dynamic_dist = (pred_waypoints[:, :, None, :, 0:2] - map_pos[:, None, :, None, 0:2]).norm(dim=-1) 
    # (num_center_objects, num_query, num_polylines)
    dynamic_dist = dynamic_dist.min(dim=-1)[0]  

    dynamic_topk_dist, dynamic_map_idxs = dynamic_dist.topk(
        k=min(num_polylines, num_waypoint_polylines), dim=-1, largest=False)
    dynamic_map_idxs[dynamic_topk_dist > 10000000] = -1
    if dynamic_map_idxs.shape[-1] < num_waypoint_polylines:
        dynamic_map_idxs = F.pad(dynamic_map_idxs, 
        pad=(0, num_waypoint_polylines - dynamic_map_idxs.shape[-1]), mode='constant', value=-1)
    
    # (num_center_objects, num_query, num_collected_polylines)
    collected_idxs = torch.cat((base_map_idxs, dynamic_map_idxs), dim=-1)  
    if map_topo is not None:
        collected_idxs = torch.cat((map_topo_idx, collected_idxs), dim=-1) 

    # remove duplicate indices
    sorted_idxs = collected_idxs.sort(dim=-1)[0]
    # (num_center_objects, num_query, num_collected_polylines - 1)
    duplicate_mask_slice = (sorted_idxs[..., 1:] - sorted_idxs[..., :-1] != 0)  
    duplicate_mask = torch.ones_like(collected_idxs).bool()
    duplicate_mask[..., 1:] = duplicate_mask_slice
    sorted_idxs[~duplicate_mask] = -1

    return sorted_idxs.int(), base_map_idxs

def apply_topo_attention(
    attention_layer,
    query_feat,
    query_pos_feat,
    query_searching_feat,
    kv_feat,
    kv_pos_feat,
    kv_mask,
    topo_indexing,
    is_first=False,
    sa_padding_mask=None,
):
    """
    Applying the TopoAttention function given reasoned Topology indexing
    Args:
        attention_layer (func): LocalTransformer Layer (as in EQNet and MTR)
        is_first (bool): whether to concat query pos feature (as in MTR) 
        query_feat, query_pos_feat, query_searching_feat  [M, B, D]
        kv_feat, kv_pos_feat  [B, N, D]
        kv_mask [B, N]
        topo_indexing [B, N, N_top]
    Return:
        query_feat [M, B, D]
    """

    batch_size, num_kv, _ = kv_feat.shape
    num_q, _, d_model = query_feat.shape
    #flat and stack for local attention
    if attention_layer.use_local_attn:
        kv_feat_stack = kv_feat.flatten(start_dim=0, end_dim=1)
        kv_pos_feat_stack = kv_pos_feat.permute(1, 0, 2).contiguous().flatten(start_dim=0, end_dim=1)
        kv_mask_stack = kv_mask.view(-1)
    else:
        kv_feat_stack = kv_feat.permute(1, 0, 2)
        kv_pos_feat_stack = kv_pos_feat
        kv_mask_stack = kv_mask

    if attention_layer.use_local_attn == False:
        # full attn cases:
        key_batch_cnt, topo_indexing, index_pair_batch = None, None, None
    else:
        # local attn cases:
        topo_indexing = topo_indexing.view(batch_size * num_q, -1)

    key_batch_cnt = num_kv * torch.ones(batch_size).int().to(kv_feat.device) 
    index_pair_batch = torch.arange(batch_size).type_as(key_batch_cnt)[:, None].repeat(1, num_q).view(-1) 

    query_feature = attention_layer(
        tgt=query_feat,
        query_pos=query_pos_feat,
        query_sine_embed=query_searching_feat,

        sa_padding_mask=sa_padding_mask,

        memory=kv_feat_stack,
        pos=kv_pos_feat_stack,
        memory_valid_mask=kv_mask_stack,

        key_batch_cnt=key_batch_cnt,
        index_pair=topo_indexing,
        index_pair_batch=index_pair_batch,

        is_first=is_first,
    )
    query_feature = query_feature.view(batch_size, num_q, d_model).permute(1, 0, 2) 
    return query_feature


