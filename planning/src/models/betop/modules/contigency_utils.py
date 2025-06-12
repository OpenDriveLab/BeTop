'''
Behavioral Topology (BeTop): https://arxiv.org/abs/2409.18031
'''
'''
Pipeline developed upon planTF: 
https://arxiv.org/pdf/2309.10443
'''

import torch
import numpy as np 
import torch.nn.functional as F 
import torch.nn as nn 

def contigency_selection(
    traj, 
    joint_cost, 
    marginal_cost,
    uncertainty_score,
    joint_mode=6,
    marginal_mode=6,
    branch_t=30):
    '''
    selecting trajectories with corresponding marginal and joint score
    traj: [b, m_j*m_m, t, d] joint_score[b, m_j],
    marginal_score[b, m_j*m_m], uncertainty_score[b, m_m]
    '''
    b, _, t, d = traj.shape
    full_joint_score = joint_cost[..., None].expand(-1, -1, marginal_mode)
    marginal_cost = marginal_cost.view(b, joint_mode, marginal_mode)
    contigency_cost = joint_cost +  torch.sum(uncertainty_score[:, None, :] * marginal_cost, dim=-1)
    best_contigency = torch.argmin(contigency_cost, dim=-1)

    plan_traj = traj.reshape(b, joint_mode, marginal_mode, t, d)
    best_plan_traj = plan_traj[torch.range(b), best_contigency]
    return best_plan_traj


def marginal_safe_cost(
    plan, pred_trajs, 
    only_agent_mask,
    joint_mode=6,
    marginal_mode=6,
    branch_t=30,
    lat_thres=1.5,
    lon_thres=3.0):
    '''
    conducting the marginal cost for branched trajs
    plan:[b, m_j*m_m, t, d]
    pred_traj: [b, top_n, m_m, t, d]
    only_agent_mask: [b, top_n]
    return marginal_cost: [b, m_j, m_m], cost_mask [b, m_j, m_m]
    '''
    b, n, _, t, d = pred_trajs.shape
    # plan = plan.reshape(b, joint_mode, marginal_mode, t, -1)[:, :, :, branch_t:, :]
    plan = plan[:, :, branch_t:, :]
    pred = pred_trajs[:, :, :, branch_t:, :]

    #[b, top_n, m_j, m_m, t]
    # print(pred.shape, plan.shape)
    diff =  pred[:, :, None, :, :, :2] - plan[:, None, :, None, :, :2]
    #[b, topn, m_plan_full, m_j, t, 2]
    dist = torch.linalg.norm(diff, dim=-1)
    full_agent_mask = only_agent_mask[:, :, None, None, None].float()
    dist = dist * full_agent_mask + (1 - full_agent_mask)*10000
    min_dist = torch.min(dist, dim=1)[0]
    min_dist_mask  = (min_dist < lon_thres).float()
    
    #[b, m_plan_full, m_joint, t]
    cost = min_dist_mask * 1 / (1 + min_dist) 
    #[b, 32, 6]
    cost = torch.sum(cost, dim=-1)
    return cost, torch.sum(min_dist_mask, dim=-1)


def joint_centerline_cost(
    plan, 
    centerline,
    joint_mode=6,
    marginal_mode=6,
    branch_t=30,
    thres=3.0):
    plan = plan[:, :, :branch_t, :]
    b, f_m, t, d = plan.shape
    plan = plan.reshape(b, f_m*t, d)

    #[b, l, f_m*t, 2]
    dist = plan[:, None, :, :2] - centerline[:, :, None, :2]
    min_ind = torch.argmin(torch.linalg.norm(dist, dim=-1), dim=1)
    #[b, f_m*t]
    min_line = centerline[torch.arange(b)[:, None], min_ind]
    min_diff = plan[..., :2] - min_line[..., :2]
    min_angle = min_line[..., 2]
    lateral_cost = torch.cos(min_angle) * min_diff[..., 1] - torch.sin(min_angle) * min_diff[..., 0]
    lateral_cost = lateral_cost.reshape(b, f_m, t)
    lateral_cost = torch.abs(lateral_cost)
    centerline_cost = lateral_cost - thres
    centerline_mask = (lateral_cost > thres).float()
    centerline_cost = centerline_cost * centerline_mask
    sum_centerline_mask = torch.sum(centerline_mask, dim=-1)
    return torch.sum(centerline_cost, dim=-1) / (sum_centerline_mask + (sum_centerline_mask==0).float()), sum_centerline_mask


def joint_safety_cost(
    plan, pred_trajs, 
    pred_mask,
    joint_mode=6,
    marginal_mode=6,
    branch_t=30,
    lat_thres=1.5,
    lon_thres=3.0):
    
    '''
    conducting the joint cost for plan trajs considering full preds
    plan:[b, m_j, branch_t, d]
    pred_traj: [b, n, m_m, t, d]
    pred_mask: [b, n]
    return marginal_cost: [b, m_j, m_m], cost_mask [b, m_j, m_m]
    '''
    b, n, _, t, d = pred_trajs.shape
    plan = plan.reshape(b, joint_mode, branch_t, -1)
    # print(pred_mask.shape, pred_trajs.shape)
    pred_mask = pred_mask[:, :, None].repeat(1, 1, marginal_mode)
    pred_mask = pred_mask.reshape(b, n*marginal_mode)
    pred_trajs = pred_trajs.reshape(b, n*marginal_mode, t, d)
    pred = pred_trajs[:, :, :branch_t, :]

    #[b, top_n*m_m, m_j, t]
    angle = plan[..., -1] #[b, mj, t]
    diff =  pred[:, :, None, :, :2] - plan[:, None, :, :, :2]
    dist = torch.linalg.norm(diff, dim=-1)

    full_agent_mask = pred_mask[:, :, None, None].float()
    dist = dist * full_agent_mask + (1 - full_agent_mask)*10000
    min_dist = torch.min(dist, dim=1)[0]
    min_dist_mask  = (min_dist < lon_thres).float()

    #[b, m_plan_full, t]
    cost = min_dist_mask * 1 / (1 + min_dist) 
    #[b, 32, 6]
    cost = torch.sum(cost, dim=-1)
    return cost, torch.sum(min_dist_mask, dim=-1)

def joint_combination(pred, score, mask, top_n=6):
    '''
    conducting full combinations across all modal 
    and keep the top-k joint modes
    pred: [b, n, m, t, d]; score: [b, n, m]; mask: [b, n]
    '''
    b, n, m, t, d = pred.shape
    m_inds = torch.arange(m).to(score.device)
    mask = mask.float()[:, :, None]
    # the masked score for certain agent will be 1 for its all modes
    score = score * mask 
    score = score + (1 - mask)
    # [b, m]
    m_inds = m_inds[None, :].repeat(b, 1)
    comb_inds_shape = (b,) + (n,) + (m,) * n
    comb_score_shape =  (b,) + (m,) * n 
    vanilla_inds = torch.zeros(comb_inds_shape).to(score.device)
    vanilla_scores = torch.ones(comb_score_shape).to(score.device)

    for i in range(n):
        insert_score_shape = (b,) + (1,)*i + (m,) + (1,)*(n-i-1)
        vanilla_scores = vanilla_scores * score[:, i].reshape(insert_score_shape)
        vanilla_inds[:, i] = vanilla_inds[:, i] + m_inds.reshape(insert_score_shape)

    vanilla_scores, vanilla_inds = torch.flatten(vanilla_scores, start_dim=1), torch.flatten(vanilla_inds, start_dim=2)
    top_score, top_score_inds = vanilla_scores.topk(top_n, dim=-1)
    top_score_inds = top_score_inds[:, None, :].repeat(1, n, 1)
    #[b, n, top_n]
    top_n_inds = vanilla_inds[torch.arange(b)[:, None, None], torch.arange(n)[None, :, None], top_score_inds]
    top_n_inds = top_n_inds.long()
    ret_traj = pred[torch.arange(b)[:, None, None], torch.arange(n)[None, :, None], top_n_inds]
    return ret_traj, top_score
 
def contigency_loss(
    joint_plan,
    plan, pred, score, only_agent_mask, behave_occ, 
    centerline,
    top_occ_agents=4,
    joint_mode=6,
    marginal_mode=6,
    branch_t=30,
    lat_thres=1.5,
    lon_thres=3.0,
    center_thres=3.0):
    '''
    calculating the contigency loss
    joint_plan: [b, m_j, branch_t, 3]
    plan: [b, m_j*m_m, t, 3]
    pred: [b, n, m_m, t, 3]
    only_agent_mask:[b, n]
    score: [b, n, m]
    behave_occ: [b, n]
    centerline:[b, L, 3]
    '''
    b, n, m = score.shape
    only_agent_mask = only_agent_mask.float()
    behave_occ = behave_occ * only_agent_mask
    _, top_occ_inds = behave_occ.topk(top_occ_agents, dim=-1)

    score = score[torch.arange(b)[:, None], top_occ_inds]
    jpred = pred[torch.arange(b)[:, None], top_occ_inds]
    filtered_agent_mask = only_agent_mask[torch.arange(b)[:, None], top_occ_inds]

    joint_pred_traj, joint_score = joint_combination(jpred, score, filtered_agent_mask, marginal_mode)
    joint_score = joint_score.softmax(-1)

    joint_loss, joint_mask = joint_safety_cost(joint_plan, pred, only_agent_mask,
            joint_mode, marginal_mode, branch_t, lat_thres, lon_thres)
    sum_joint_mask = torch.sum(joint_mask)
    joint_loss = torch.sum(joint_loss) / (sum_joint_mask + (sum_joint_mask==0).float())

    marginal_cost, marginal_mask =  marginal_safe_cost(plan, joint_pred_traj, filtered_agent_mask,
            joint_mode, marginal_mode, branch_t, lat_thres, lon_thres)
    sum_marginal_mask = torch.sum(marginal_mask)
    marginal_loss = torch.sum(joint_score.detach()[:, None, :] * marginal_cost) / (sum_marginal_mask + (sum_marginal_mask==0).float())

    contigency_loss = joint_loss +  0.5 * marginal_loss
    return contigency_loss


def test_contigency_loss():
    from time import time
    for _ in range(100):
        joint_plan = torch.randn(32, 6, 30, 3)
        plan = torch.randn(32, 32, 80, 3)
        pred = torch.randn(32, 32, 6, 80, 3)
        score = torch.randn(32, 32, 6)
        behave_occ = torch.randn(32, 32)
        mask = torch.ones_like(behave_occ).bool()
        centerline = torch.randn(32, 150, 3)
        s = time()
        loss = contigency_loss(joint_plan,
        plan, pred, score, mask, behave_occ,
        centerline,)
        print(loss, time()-s)

if __name__ == '__main__':
    test_contigency_loss()