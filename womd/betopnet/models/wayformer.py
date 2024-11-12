'''
Behavioral Topology (BeTop): https://arxiv.org/abs/2409.18031
'''
'''
Mostly from UniTraj (ECCV'24): https://arxiv.org/abs/2403.15098
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
from easydict import EasyDict

from betopnet.models.utils.base_model import BaseModel
from betopnet.utils import common_utils, loss_utils, motion_utils
from .encoder import build_encoder
from .decoder import build_decoder


def init(module, weight_init, bias_init, gain=1):
    '''
    This function provides weight and bias initializations for linear layers.
    '''
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class Wayformer(BaseModel):

    def __init__(self, config):
        super(Wayformer, self).__init__()
        self.config = config
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.fisher_information = None

        encoder_config, decoder_config = self.separate_config(config)

        self.road_pts_lin = nn.Sequential(init_(nn.Linear(self.map_attr, self.d_k)))
        # INPUT ENCODERS
        self.agents_dynamic_encoder = nn.Sequential(init_(nn.Linear(self.k_attr, self.d_k)))

        self.perceiver_encoder = build_encoder(encoder_config)
        self.perceiver_decoder = build_decoder(decoder_config)

        self.agents_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, 1, (self._M + 1), self.d_k)),
            requires_grad=True
        )

        self.temporal_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, self.past_T, 1, self.d_k)),
            requires_grad=True
        )

        self.prob_predictor = nn.Sequential(init_(nn.Linear(self.d_k, 1)))
        self.output_model = nn.Sequential(init_(nn.Linear(self.d_k, 5 * self.T)))
        self.selu = nn.SELU(inplace=True)
        self.criterion = Criterion(self.config, self.object_type)

        self.fisher_information = None
        self.optimal_params = None
    

    def separate_config(self, config):
        self.object_type = ['TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST']
        self.map_attr = config['num_map_feature']
        self.k_attr = config['num_agent_feature']
        self.d_k = config['hidden_size']
        self._M = config['max_num_agents']  # num agents without the ego-agent
        self.c = config['num_modes']
        self.T = config['future_len']
        self.L_enc = config['num_encoder_layers']
        self.dropout = config['dropout']
        self.num_heads = config['tx_num_heads']
        self.L_dec = config['num_decoder_layers']
        self.tx_hidden_size = config['tx_hidden_size']
        self.use_map_img = config['use_map_image']
        self.use_map_lanes = config['use_map_lanes']
        self.past_T = config['past_len']
        self.max_points_per_lane = config['max_points_per_lane']
        self.max_num_roads = config['max_num_roads']
        self.num_queries_enc = config['num_queries_enc']
        self.num_queries_dec = config['num_queries_dec']

        encoder_config = dict(
            NAME='WayformerEncoder',
            num_latents=self.num_queries_enc,
            num_latent_channels=self.d_k,
            num_cross_attention_qk_channels=self.d_k,
            num_cross_attention_v_channels=self.d_k,
            num_self_attention_qk_channels=self.d_k,
            num_self_attention_v_channels=self.d_k
            )
        encoder_config = EasyDict(encoder_config)

        decoder_config = dict(
            NAME='WayformerDecoder',
            num_latent_channels=self.d_k,
            num_queries_dec=self.num_queries_dec,
            hidden_size=config['hidden_size'],
            )
        decoder_config = EasyDict(decoder_config)

        return encoder_config, decoder_config
    
    def process_observations(self, ego, agents):
        '''
        :param observations: (B, T, N+2, A+1) where N+2 is [ego, other_agents, env]
        :return: a tensor of only the agent dynamic states, active_agent masks and env masks.
        '''
        # ego stuff
        ego_tensor = ego[:, :, :self.k_attr]
        env_masks_orig = ego[:, :, -1]
        env_masks = (1.0 - env_masks_orig).to(torch.bool)
        env_masks = env_masks.unsqueeze(1).repeat(1, self.num_queries_dec, 1).view(ego.shape[0] * self.num_queries_dec,
                                                                                   -1)

        # Agents stuff
        temp_masks = torch.cat((torch.ones_like(env_masks_orig.unsqueeze(-1)), agents[:, :, :, -1]), dim=-1)
        opps_masks = (1.0 - temp_masks).to(torch.bool)  # only for agents.
        opps_tensor = agents[:, :, :, :self.k_attr]  # only opponent states

        return ego_tensor, opps_tensor, opps_masks, env_masks
    
    def _forward(self, batch_dict):
        '''
        :param ego_in: [B, T_obs, k_attr+1] with last values being the existence mask.
        :param agents_in: [B, T_obs, M-1, k_attr+1] with last values being the existence mask.
        :param roads: [B, S, P, map_attr+1] representing the road network if self.use_map_lanes or
                      [B, 3, 128, 128] image representing the road network if self.use_map_img or
                      [B, 1, 1] if self.use_map_lanes and self.use_map_img are False.
        :return:
            pred_obs: shape [c, T, B, 5] c trajectories for the ego agents with every point being the params of
                                        Bivariate Gaussian distribution.
            mode_probs: shape [B, c] mode probability predictions P(z|X_{1:T_obs})
        '''

        input_dict = batch_dict['input_dict']
        obj_trajs, obj_trajs_mask = input_dict['obj_trajs'].cuda(), input_dict['obj_trajs_mask'].cuda() 
        map_polylines, map_polylines_mask = input_dict['map_polylines'].cuda(), input_dict['map_polylines_mask'].cuda() 

        obj_trajs_last_pos = input_dict['obj_trajs_last_pos'].cuda() 
        map_polylines_center = input_dict['map_polylines_center'].cuda() 
        track_index_to_predict = input_dict['track_index_to_predict']

        num_center_objects = obj_trajs.shape[0]
        ego_in = obj_trajs[torch.arange(num_center_objects), track_index_to_predict]
        ego_in_mask = obj_trajs_mask[torch.arange(num_center_objects), track_index_to_predict]

        agent_dist = torch.linalg.norm(obj_trajs_last_pos, dim=-1)

        agt_mask = obj_trajs_mask.float().sum(-1) !=0
        agent_dist[~agt_mask] = 10000000.0
        agent_dist[torch.arange(num_center_objects), track_index_to_predict] = 10000000.0
        agent_topk_dist, base_agent_idxs = agent_dist.topk(k=min(self._M, agent_dist.shape[1]), dim=-1, largest=False)  # (num_center_objects, topk)
        less_agent_mask = agent_topk_dist < 10000000.0
        agents_in = obj_trajs[torch.arange(num_center_objects)[:, None], base_agent_idxs]
        agent_mask = obj_trajs_mask[torch.arange(num_center_objects)[:, None], base_agent_idxs]
        agent_mask = torch.logical_and(agent_mask, less_agent_mask[:, :, None])
        if agent_mask.shape[1] < self._M:
            agent_mask = F.pad(agent_mask, (0, 0, 0, self._M - agent_dist.shape[1]))
            agents_in = F.pad(agents_in, (0, 0, 0, 0, 0, self._M - agent_dist.shape[1]))

        agents_in = torch.cat([agents_in, agent_mask[:, :, :, None].type_as(obj_trajs)], dim=-1)
        agents_in = agents_in.permute(0, 2, 1, 3) #[b, t, a, d]

        ego_in = torch.cat([ego_in, ego_in_mask[:, :, None]], dim=-1)

        map_dist = torch.linalg.norm(map_polylines_center, dim=-1)
        mp_mask = map_polylines_mask.float().sum(-1) !=0
        map_dist[~mp_mask] = 10000000.0
        map_topk_dist, base_map_idxs = map_dist.topk(k=min(self.max_num_roads, map_dist.shape[1]), dim=-1, largest=False)  # (num_center_objects, topk)
        less_map_mask = map_topk_dist < 10000000.0
        roads = map_polylines[torch.arange(num_center_objects)[:, None], base_map_idxs]
        map_polylines_mask = map_polylines_mask[torch.arange(num_center_objects)[:, None], base_map_idxs]
        map_polylines_mask = torch.logical_and(map_polylines_mask, less_map_mask[:, :, None])
        if map_polylines_mask.shape[1] < self.max_num_roads:
            map_polylines_mask = F.pad(map_polylines_mask, (0, 0, 0, self.max_num_roads - map_dist.shape[1]))
            roads = F.pad(roads, (0, 0, 0, 0, 0, self.max_num_roads - map_dist.shape[1]))
        roads = torch.cat([roads, map_polylines_mask[:, :, :, None]], dim=-1)

        B = ego_in.size(0)
        num_agents = agents_in.shape[2] + 1
        # Encode all input observations (k_attr --> d_k)
        ego_tensor, _agents_tensor, opps_masks_agents, env_masks = self.process_observations(ego_in, agents_in)
        agents_tensor = torch.cat((ego_tensor.unsqueeze(2), _agents_tensor), dim=2)
        agents_emb = self.selu(self.agents_dynamic_encoder(agents_tensor))
        agents_emb = (agents_emb + self.agents_positional_embedding[:, :,
                                   :num_agents] + self.temporal_positional_embedding).view(B, -1, self.d_k)
        road_pts_feats = self.selu(self.road_pts_lin(roads[:, :self.max_num_roads, :, :self.map_attr]).view(B, -1,
                                                                                                            self.d_k))# + self.map_positional_embedding
        mixed_input_features = torch.concat([agents_emb, road_pts_feats], dim=1)
        opps_masks_roads = (1.0 - roads[:, :self.max_num_roads, :, -1]).to(torch.bool)
        mixed_input_masks = torch.concat([opps_masks_agents.view(B, -1), opps_masks_roads.view(B, -1)], dim=1)
        # Process through Wayformer's encoder
        context = self.perceiver_encoder(mixed_input_features, mixed_input_masks)

        # Wazformer-Ego Decoding
        out_seq = self.perceiver_decoder(context)

        out_dists = self.output_model(out_seq[:, :self.c]).reshape(B, self.c, self.T, -1)

        # Mode prediction
        mode_probs = self.prob_predictor(out_seq[:, :self.c]).reshape(B, self.c)

        batch_dict['pred_scores'] = mode_probs
        batch_dict['pred_trajs'] = out_dists

        if len(np.argwhere(np.isnan(out_dists.detach().cpu().numpy()))) > 1:
            breakpoint()
        return batch_dict

    def forward(self, batch):
        model_input = {}
        inputs = batch['input_dict']
        if self.training:
            self.forward_ret_dict = dict()
            self.forward_ret_dict['center_gt_trajs_mask'] = inputs['center_gt_trajs_mask'].cuda()
            self.forward_ret_dict['center_objects_type'] = inputs['center_objects_type']

        output = self._forward(batch)
        ground_truth = torch.cat([inputs['center_gt_trajs'][..., :2], inputs['center_gt_trajs_mask'].unsqueeze(-1)],
                                 dim=-1)
        # output['dataset_name'] = inputs['dataset_name']
        output['pred_scores'] = F.softmax(output['pred_scores'], dim=-1)
        if self.training:
            loss, tb_dict, disp_dict = self.criterion(output, ground_truth, inputs['center_gt_final_valid_idx'],
            forward_ret_dict=self.forward_ret_dict)
            tb_dict.update({'loss': loss.item()})
            disp_dict.update({'loss': loss.item()})
            return loss, tb_dict, disp_dict
        return output

       
class Criterion(nn.Module):
    def __init__(self, config, object_type=None):
        super(Criterion, self).__init__()
        self.config = config
        self.object_type = object_type
        pass

    def forward(self, out, gt, center_gt_final_valid_idx, forward_ret_dict):
        return self.nll_loss_gmm_direct(out['pred_scores'], out['pred_trajs'], gt,
                                        center_gt_final_valid_idx, forward_ret_dict=forward_ret_dict)

    def nll_loss_gmm_direct(self, pred_scores, pred_trajs, gt_trajs, center_gt_final_valid_idx,
                            pre_nearest_mode_idxs=None,
                            timestamp_loss_weight=None, use_square_gmm=False, log_std_range=(-1.609, 5.0),
                            rho_limit=0.5, forward_ret_dict=None):
        """
        GMM Loss for Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
        Written by Shaoshuai Shi

        Args:
            pred_scores (batch_size, num_modes):
            pred_trajs (batch_size, num_modes, num_timestamps, 5 or 3)
            gt_trajs (batch_size, num_timestamps, 3):
            timestamp_loss_weight (num_timestamps):
        """
        if use_square_gmm:
            assert pred_trajs.shape[-1] == 3
        else:
            assert pred_trajs.shape[-1] == 5

        batch_size = pred_trajs.shape[0]

        gt_valid_mask = gt_trajs[..., -1]

        if pre_nearest_mode_idxs is not None:
            nearest_mode_idxs = pre_nearest_mode_idxs
        else:
            distance = (pred_trajs[:, :, :, 0:2] - gt_trajs[:, None, :, :2]).norm(dim=-1)
            distance = (distance * gt_valid_mask[:, None, :]).sum(dim=-1)

            nearest_mode_idxs = distance.argmin(dim=-1)
        nearest_mode_bs_idxs = torch.arange(batch_size).type_as(nearest_mode_idxs)  # (batch_size, 2)

        nearest_trajs = pred_trajs[nearest_mode_bs_idxs, nearest_mode_idxs]  # (batch_size, num_timestamps, 5)
        res_trajs = gt_trajs[..., :2] - nearest_trajs[:, :, 0:2]  # (batch_size, num_timestamps, 2)
        dx = res_trajs[:, :, 0]
        dy = res_trajs[:, :, 1]

        if use_square_gmm:
            log_std1 = log_std2 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
            std1 = std2 = torch.exp(log_std1)  # (0.2m to 150m)
            rho = torch.zeros_like(log_std1)
        else:
            log_std1 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
            log_std2 = torch.clip(nearest_trajs[:, :, 3], min=log_std_range[0], max=log_std_range[1])
            std1 = torch.exp(log_std1)  # (0.2m to 150m)
            std2 = torch.exp(log_std2)  # (0.2m to 150m)
            rho = torch.clip(nearest_trajs[:, :, 4], min=-rho_limit, max=rho_limit)

        gt_valid_mask = gt_valid_mask.type_as(pred_scores)
        if timestamp_loss_weight is not None:
            gt_valid_mask = gt_valid_mask * timestamp_loss_weight[None, :]

        # -log(a^-1 * e^b) = log(a) - b
        reg_gmm_log_coefficient = log_std1 + log_std2 + 0.5 * torch.log(1 - rho ** 2)  # (batch_size, num_timestamps)
        reg_gmm_exp = (0.5 * 1 / (1 - rho ** 2)) * (
                (dx ** 2) / (std1 ** 2) + (dy ** 2) / (std2 ** 2) - 2 * rho * dx * dy / (
                std1 * std2))  # (batch_size, num_timestamps)

        reg_loss = ((reg_gmm_log_coefficient + reg_gmm_exp) * gt_valid_mask).sum(dim=-1)

        loss_cls = (F.cross_entropy(input=pred_scores, target=nearest_mode_idxs, reduction='none'))

        tb_dict = {}
        disp_dict = {}
        tb_dict[f'loss_reg_gmm'] = reg_loss.mean().item()
        tb_dict[f'loss_reg_cls'] = loss_cls.mean().item()

        disp_dict = motion_utils.get_ade_of_each_category(
                    pred_trajs=pred_trajs[..., :2],
                    gt_trajs=gt_trajs[:, :, 0:2], gt_trajs_mask=forward_ret_dict['center_gt_trajs_mask'],
                    object_types=forward_ret_dict['center_objects_type'],
                    valid_type_list=self.object_type,
                    post_tag=f'',
                    pre_tag=''
                )

        return (reg_loss + loss_cls).mean(), tb_dict, disp_dict

        