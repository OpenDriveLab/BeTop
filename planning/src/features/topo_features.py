'''
Behavioral Topology (BeTop): https://arxiv.org/abs/2409.18031
'''
'''
Pipeline developed upon planTF: 
https://arxiv.org/pdf/2309.10443
'''
import numpy as np 
import torch 


def segments_intersect(line1_start, line1_end, line2_start, line2_end):
    #calculating intersection given arbitary shape

    # Calculating the differences
    dx1 = line1_end[..., 0] - line1_start[..., 0]
    dy1 = line1_end[..., 1] - line1_start[..., 1]
    dx2 = line2_end[..., 0] - line2_start[..., 0]
    dy2 = line2_end[..., 1] - line2_start[..., 1]

    # Calculating determinants
    det = dx1 * dy2 - dx2 * dy1
    det_mask = det != 0

    # Checking if lines are parallel or coincident
    # close_mask = torch.logical_or(torch.abs(line1_end[..., 0]-line2_end[..., 0]) > 3, torch.abs(line1_start[..., 0]-line2_start[..., 0]) > 3)
    parallel_mask = torch.logical_not(det_mask)
    # parallel_mask = torch.logical_and(parallel_mask, close_mask)

    # Calculating intersection parameters
    t1 = ((line2_start[..., 0] - line1_start[...,  0]) * dy2 
          - (line2_start[..., 1] - line1_start[..., 1]) * dx2) / det
    t2 = ((line2_start[..., 0] - line1_start[..., 0]) * dy1 
          - (line2_start[..., 1] - line1_start[..., 1]) * dx1) / det

    # Checking intersection conditions
    intersect_mask = torch.logical_and(
        torch.logical_and(t1 >= 0, t1 <= 1),
        torch.logical_and(t2 >= 0, t2 <= 1)
    )

    # Handling parallel or coincident lines
    intersect_mask[parallel_mask] = False

    return intersect_mask

def judge_overlap(src_s, src_e, tar_s, tar_e,
    src_y_s, src_y_e, tar_y_s, tar_y_e):
    """
    first calculate the intersection:
    https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    then judge which one is overlapped above
    """
    x1, x2, x3, x4 = src_s[..., 0], src_e[..., 0], tar_s[..., 0], tar_e[..., 0]
    y1, y2, y3, y4 = src_s[..., 1], src_e[..., 1], tar_s[..., 1], tar_e[..., 1]
    z1, z2, z3, z4 = src_y_s, src_y_e, tar_y_s, tar_y_e

    inter_x = (x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4) 
    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    inter_x = inter_x / (det + 1e-8)

    inter_z_src = (inter_x - x2) / (x1 - x2 + 1e-8) * (z1 - z2) + z2
    inter_z_tar = (inter_x - x4) / (x3 - x4 + 1e-8) * (z3 - z4) + z4

    return (inter_z_src >= inter_z_tar).float() - (inter_z_src < inter_z_tar).float()


def judge_briad_indicater(src_traj, tar_traj, src_mask, tgt_mask):
    """
    judge the braid indication according to src and ter trajectories:
    src_traj, tar_traj: [b, T, 2]
    return res [b] containing {-1, 0, 1}
    """
    b, t, _ = src_traj.shape
    traj_t = torch.linspace(0, 1, t).to(src_traj.device)
    traj_t = traj_t[None, :].expand(b, -1)

    src_xt = torch.stack([src_traj[:, :, 0], traj_t], dim=-1)
    tar_xt = torch.stack([tar_traj[:, :, 0], traj_t], dim=-1)

    src_start, src_end = src_xt[:, :-1, :2], src_xt[:, 1:, :2]
    tar_start, tar_end = tar_xt[:, :-1, :2], tar_xt[:, 1:, :2]

    src_start_m, src_end_m = src_mask[:, :-1], src_mask[:, 1:]
    src_seg_valid = torch.logical_and(src_start_m, src_end_m)
    tgt_start_m, tgt_end_m = tgt_mask[:, :-1], tgt_mask[:, 1:]
    tgt_seg_valid = torch.logical_and(tgt_start_m, tgt_end_m)
    inter_valid = torch.logical_and(src_seg_valid, tgt_seg_valid)

    # raw braids :[b, T]
    raw_briad_mask = segments_intersect(src_start, src_end, tar_start, tar_end)
    raw_briad_mask = torch.logical_and(raw_briad_mask, inter_valid)
    # raw_briad_mask = raw_briad_mask.float()
    # print(raw_briad_mask[:10, 4::5])
    # assert 1==0

    src_y_start, src_y_end = src_traj[:, :-1, 1], src_traj[:, 1:, 1]
    tar_y_start, tar_y_end = tar_traj[:, :-1, 1], tar_traj[:, 1:, 1]

    dist = torch.sqrt((src_y_start - tar_y_start)**2 + (src_start[..., 0] - tar_start[..., 0])**2)
    far_mask = torch.logical_and(dist < 20, inter_valid)
    # near_mask = torch.logical_and(dist < 5, inter_valid)

    # y_mask = torch.logical_and(src_y_start - 3 < tar_y_start, dist < 20)
    raw_briad_mask = torch.logical_and(raw_briad_mask,  far_mask)
    # raw_briad_mask = torch.logical_or(raw_briad_mask,  near_mask)
    return torch.any(raw_briad_mask, dim=-1)

    overlap_mask = judge_overlap(src_start, src_end, tar_start, tar_end,
        src_y_start, src_y_end, tar_y_start, tar_y_end)
    
    # [B, T-1]
    # discard the multi-intersection cases:
    multi_case = (raw_briad_mask.sum(-1) > 1).float()
    res_mask = raw_briad_mask * overlap_mask
    multi_val = res_mask.sum(-1) * multi_case
    filtering = (multi_val >=0).float() - (multi_val < 0).float()

    return (1 - multi_case) * res_mask.sum(-1) + multi_case * filtering

def create_batched_combination_trajs(src_trajs, pos=None, head=None):
    """
    src trajs: [b, a, T, d]
    return [b, a, a, 2, T, d]
    """
    b, a, t, d = src_trajs.shape
    blank_traj = torch.zeros_like(src_trajs)
    src = torch.stack([src_trajs, blank_traj], dim=2)[:, :, None, :, :]
    tgt = torch.stack([blank_traj, src_trajs], dim=2)[:, None, :, :, :]
    res = src + tgt
    # print(res.shape)

    mask = res.sum(-1) != 0
    # unmask ego current:
    mask[:, 0, :, 0, 0] = True
    mask[:, :, 0, 1, 0] = True

    if pos is not None:
        x, y = res[..., 0] - pos[:, :, None, None, None, 0], res[..., 1] - pos[:, :, None, None, None, 1]
        cos, sin = torch.cos(head)[:, :, None, None, None], torch.sin(head)[:, :, None, None, None]
        new_x = cos * x + sin * y 
        new_y = -sin *x + cos * y 
        return torch.stack([new_x, new_y], dim=-1), mask
    return res, mask

def generate_behavior_braids(src_trajs, pos, head):
    """
    generating the behavior_braids label for interacted trajs
    inputs: src trajs: [b, a, T, d], pos [b, a, 2], head [b, a]
    return src_braids: [b, a, a]
    """
    #make full combinations:s
    combinated_trajs, mask = create_batched_combination_trajs(src_trajs, pos, head)
    combinated_trajs = combinated_trajs * mask.unsqueeze(-1).float()

    # transformed to ego heading as Y:
    combinated_trajs[..., [0, 1]] = combinated_trajs[..., [1, 0]]
    combinated_trajs[..., 0] = -combinated_trajs[..., 0]
    
    # combinated_trajs = combinated_trajs[..., 4::5, :]
    b, a, a, _, t, d = combinated_trajs.shape
    # sel = combinated_trajs[:, 1, 2, :, :, :]
    # print(torch.sum(sel[:, 0] - src_trajs[:, 2]))
    # print(torch.sum(sel[:, 1] - src_trajs[:, 1]))
    
    combinated_trajs = combinated_trajs.view(b*a*a, 2, t, d)
    # mask = mask[..., 4::5]
    combined_mask = mask.view(b*a*a, 2, t)
    src_mask, tgt_mask = combined_mask[:, 0], combined_mask[:, 1]

    src_comb_trajs, tat_comb_trajs = combinated_trajs[:, 0], combinated_trajs[:, 1]

    # calculating the braids:
    braids = judge_briad_indicater(src_comb_trajs, tat_comb_trajs, src_mask, tgt_mask)
    braids = braids.reshape(b, a, a)
    return braids

def generate_map_briads(src_trajs, map_centers, src_mask, map_mask):
    """
    generating the behavior_braids label for source trajs with map
    inputs: src trajs: [b, a, T, d], map_centers [b, m, l, 2]
    src_mask: [b, a, T], map_mask [b, m, l]
    return map_braids: [b, a, m]
    """
    #[ b, a, t, m, l]
    fvalid_mask = src_mask[:, :, :, None, None] * map_mask[:, None, None, :, :]
    invalid_dist = (1 - fvalid_mask.float()) * 10e6
    #[ b, a, t, m, l]
    dist = torch.linalg.norm(src_trajs[:, :, :, None, None, :2] - map_centers[:, None, None, :, :, :2], dim=-1)
    dist += invalid_dist
    min_dist = torch.min(dist, dim=-1)[0]
    #[b, a, t], adding invalid_dist for not valid element
    map_min_dist, min_dist_ind = torch.min(min_dist, dim=-1)
    mask = torch.zeros_like(min_dist)
    mask.scatter_(3, min_dist_ind.unsqueeze(-1), 1.)[..., 0]
    mask = mask.bool()
    mask = torch.logical_and(mask, (map_min_dist < 3)[..., None])
    valid_mask = torch.any(fvalid_mask, dim=-1)
    return torch.logical_and(torch.any(mask, dim=-2), torch.any(valid_mask,dim=-2))


if __name__ == '__main__':
    from time import time
    src_traj = torch.randn((32, 6, 80, 2))
    src_map = torch.randn((32, 200, 2))
    map_o = generate_map_briads(src_traj, src_map)




    
    