'''
Behavioral Topology (BeTop): https://arxiv.org/abs/2409.18031
'''
'''
Mostly from Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
'''

import torch

def batch_nms(pred_trajs, pred_scores, dist_thresh, num_ret_modes=6, return_mask=False):
    """

    Args:
        pred_trajs (batch_size, num_modes, num_timestamps, 7)
        pred_scores (batch_size, num_modes)
        dist_thresh (batch_size)
        num_ret_modes (int, optional): Defaults to 6.

    Returns:
        ret_trajs (batch_size, num_ret_modes, num_timestamps, 7)
        ret_scores (batch_size, num_ret_modes)
        ret_idxs (batch_size, num_ret_modes)
    """
    batch_size, num_modes, num_timestamps, num_feat_dim = pred_trajs.shape

    sorted_idxs = pred_scores.argsort(dim=-1, descending=True)
    bs_idxs_full = torch.arange(batch_size).type_as(sorted_idxs)[:, None].repeat(1, num_modes)
    sorted_pred_scores = pred_scores[bs_idxs_full, sorted_idxs]
    sorted_pred_trajs = pred_trajs[bs_idxs_full, sorted_idxs]  # (batch_size, num_modes, num_timestamps, 7)
    sorted_pred_goals = sorted_pred_trajs[:, :, -1, :]  # (batch_size, num_modes, 7)

    dist = (sorted_pred_goals[:, :, None, 0:2] - sorted_pred_goals[:, None, :, 0:2]).norm(dim=-1)
    if isinstance(dist_thresh, float):
        point_cover_mask = (dist < dist_thresh)
    else:
        point_cover_mask = (dist < dist_thresh[:, None, None])

    point_val = sorted_pred_scores.clone()  # (batch_size, N)
    point_val_selected = torch.zeros_like(point_val)  # (batch_size, N)
    ret_mask_sorted = torch.ones_like(point_val).bool() # (batch_size, N)

    ret_idxs = sorted_idxs.new_zeros(batch_size, num_ret_modes).long()
    ret_trajs = sorted_pred_trajs.new_zeros(batch_size, num_ret_modes, num_timestamps, num_feat_dim)
    ret_scores = sorted_pred_trajs.new_zeros(batch_size, num_ret_modes)
    bs_idxs = torch.arange(batch_size).type_as(ret_idxs)

    for k in range(num_ret_modes):
        cur_idx = point_val.argmax(dim=-1) # (batch_size)
        ret_idxs[:, k] = cur_idx

        new_cover_mask = point_cover_mask[bs_idxs, cur_idx]  # (batch_size, N)
        filter_mask = new_cover_mask.clone()
        filter_mask[bs_idxs, cur_idx] = False
        filter_mask *= (point_val.max(dim=-1, keepdim=True).values > 0)
        ret_mask_sorted[filter_mask] = False

        point_val = point_val * (~new_cover_mask).float()  # (batch_size, N)
        point_val_selected[bs_idxs, cur_idx] = -1
        point_val += point_val_selected

        ret_trajs[:, k] = sorted_pred_trajs[bs_idxs, cur_idx]
        ret_scores[:, k] = sorted_pred_scores[bs_idxs, cur_idx]

    bs_idxs = torch.arange(batch_size).type_as(sorted_idxs)[:, None]

    if return_mask:
        ret_mask = torch.zeros_like(ret_mask_sorted)
        ret_mask_sorted[torch.cumsum(ret_mask_sorted, dim=-1) > num_ret_modes] = False
        ret_mask[bs_idxs, sorted_idxs] = ret_mask_sorted
        return ret_mask
    ret_idxs = sorted_idxs[bs_idxs, ret_idxs]
    return ret_trajs, ret_scores, ret_idxs

def get_evolving_anchors(
    layer_idx, num_inter_layers, pred_list, 
    center_gt_goals, intention_points, 
    center_gt_trajs, center_gt_trajs_mask,
    ):
    """
    Anchor evolving by selected interaction layers
    By EDA: Evolving and Distinct Anchors for Multimodal Motion Prediction." Proceedings of the AAAI
    Args:
        layer_idx (int): current layer idx
        num_inter_layers (int): interactive layer for EDA anchors
        center_gt_goals, center_gt_trajs (Tensor): GT trajectories.
        pred_list (List[Tensor]): full prediction
    Returns:
        dist (Tensor): end-point distance
        anchor_trajs (Tensor): selected trajs for NMS
    """
    positive_layer_idx = (layer_idx//num_inter_layers) * num_inter_layers - 1
    if positive_layer_idx < 0:
        anchor_trajs = intention_points.unsqueeze(-2)
        # (num_center_objects, num_query)
        dist = (center_gt_goals[:, None, :] - intention_points).norm(dim=-1)  
    else:
        anchor_trajs = pred_list[positive_layer_idx][1]
         # (num_center_objects, num_query)
        dist = ((center_gt_trajs[:, None, :, 0:2] - anchor_trajs[..., 0:2]).norm(dim=-1) * \
            center_gt_trajs_mask[:, None]).sum(dim=-1) 
    return anchor_trajs, dist

def select_distinct_anchors(
    dist, pred_scores, pred_trajs, anchor_trajs,
    lower_dist=2.5, upper_dist=3.5, 
    lower_length=10, upper_length=50, scalar=1.5):
    """
    Selects distinct anchors based on trajectory length and configurable distance thresholds.
    By EDA: Evolving and Distinct Anchors for Multimodal Motion Prediction." Proceedings of the AAAI
    Args:
        dist (Tensor): end-point distance
        pred_scores (Tensor): Prediction scores for each trajectory.
        pred_trajs (Tensor): Predicted trajectories.
        anchor_trajs (Tensor): Anchor trajectories (layer 2, 4, 6) for NMS processing.
    Returns:
        Tensor: center_gt_positive_idx for distinctiveness criteria.
    """
    # Initialize the selection mask
    select_mask = torch.ones_like(pred_scores).bool()

    # Calculate the length of the top trajectory
    num_center_objects = pred_scores.shape[0]
    top_traj = pred_trajs[torch.arange(num_center_objects), pred_scores.argsort(dim=-1)[:, -1]][..., :2]
    top_traj_length = torch.norm(torch.diff(top_traj, dim=1), dim=-1).sum(dim=-1)

    # Set distance thresholds
    dist_thresh = torch.minimum(
        torch.tensor(upper_dist, device=pred_trajs.device),
        torch.maximum(
            torch.tensor(lower_dist, device=pred_trajs.device),
            lower_dist + scalar * (top_traj_length - lower_length) / (upper_length - lower_length)
        )
    )

    # Apply non-maximum suppression based on distance threshold
    select_mask = batch_nms(
        anchor_trajs, pred_scores.sigmoid(),
        dist_thresh=dist_thresh,
        num_ret_modes=anchor_trajs.shape[1],
        return_mask=True
    )

    dist =  dist.masked_fill(~select_mask, 1e10)
    center_gt_positive_idx = dist.argmin(dim=-1)
    return center_gt_positive_idx, select_mask

def inference_distance_nms(
    pred_scores, pred_trajs,
    lower_dist=2.5, upper_dist=3.5, 
    lower_length=10, upper_length=50, scalar=1.5
    ):
    """
    Perform NMS post-processing during inference
    Followed by MTRA
    """
    num_center_objects, num_query, num_future_timestamps, num_feat = pred_trajs.shape
    top_traj = pred_trajs[torch.arange(num_center_objects), pred_scores.argsort(dim=-1)[:, -1]][..., :2]
    top_traj_length = torch.norm(torch.diff(top_traj, dim=1), dim=-1).sum(dim=-1)

    dist_thresh = torch.minimum(
        torch.tensor(upper_dist, device=pred_trajs.device),
        torch.maximum(
            torch.tensor(lower_dist, device=pred_trajs.device),
            lower_dist+scalar*(top_traj_length-lower_length)/(upper_length-lower_length)
        )
    )

    pred_trajs_final, pred_scores_final, selected_idxs = motion_utils.batch_nms(
        pred_trajs=pred_trajs, pred_scores=pred_scores,
        dist_thresh=dist_thresh,
        num_ret_modes=self.num_motion_modes
    )
    return pred_trajs_final, pred_scores_final, selected_idxs



def get_ade_of_waymo(pred_trajs, gt_trajs, gt_valid_mask, calculate_steps=(5, 9, 15)) -> float:
    """Compute Average Displacement Error.

    Args:
        pred_trajs: (batch_size, num_modes, pred_len, 2)
        gt_trajs: (batch_size, pred_len, 2)
        gt_valid_mask: (batch_size, pred_len)
    Returns:
        ade: Average Displacement Error

    """
    # assert pred_trajs.shape[2] in [1, 16, 80]
    if pred_trajs.shape[2] == 80:
        pred_trajs = pred_trajs[:, :, 4::5]
        gt_trajs = gt_trajs[:, 4::5]
        gt_valid_mask = gt_valid_mask[:, 4::5]

    ade = 0
    for cur_step in calculate_steps:
        dist_error = (pred_trajs[:, :, :cur_step+1, :] - gt_trajs[:, None, :cur_step+1, :]).norm(dim=-1)  # (batch_size, num_modes, pred_len)
        dist_error = (dist_error * gt_valid_mask[:, None, :cur_step+1].float()).sum(dim=-1) / torch.clamp_min(gt_valid_mask[:, :cur_step+1].sum(dim=-1)[:, None], min=1.0)  # (batch_size, num_modes)
        cur_ade = dist_error.min(dim=-1)[0].mean().item()

        ade += cur_ade

    ade = ade / len(calculate_steps)
    return ade


def get_ade_of_each_category(pred_trajs, gt_trajs, gt_trajs_mask, object_types, valid_type_list, post_tag='', pre_tag=''):
    """
    Args:
        pred_trajs (num_center_objects, num_modes, num_timestamps, 2): 
        gt_trajs (num_center_objects, num_timestamps, 2): 
        gt_trajs_mask (num_center_objects, num_timestamps): 
        object_types (num_center_objects): 

    Returns:
        
    """
    ret_dict = {}
    
    for cur_type in valid_type_list:
        type_mask = (object_types == cur_type)
        ret_dict[f'{pre_tag}ade_{cur_type}{post_tag}'] = -0.0
        if type_mask.sum() == 0:
            continue

        # calculate evaluataion metric
        ade = get_ade_of_waymo(
            pred_trajs=pred_trajs[type_mask, :, :, 0:2].detach(),
            gt_trajs=gt_trajs[type_mask], gt_valid_mask=gt_trajs_mask[type_mask]
        )
        ret_dict[f'{pre_tag}ade_{cur_type}{post_tag}'] = ade
    return ret_dict