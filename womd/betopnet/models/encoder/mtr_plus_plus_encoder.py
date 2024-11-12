'''
Behavioral Topology (BeTop): https://arxiv.org/abs/2409.18031
'''
'''
(Deprecated) Reproduced MTR++ Encoder: https://arxiv.org/abs/2306.17770
'''


import numpy as np
import torch
import torch.nn as nn

from betopnet.models.utils.transformer import transformer_encoder_layer, position_encoding_utils

from functools import partial
from betopnet.models.utils import polyline_encoder
from betopnet.utils import common_utils
from betopnet.ops.knn import knn_utils
from betopnet.ops.grouping.grouping_utils import grouping_operation


class MTRPPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_cfg = config

        # build polyline encoders
        self.agent_polyline_encoder = self.build_polyline_encoder(
            in_channels=self.model_cfg.NUM_INPUT_ATTR_AGENT + 1,
            hidden_dim=self.model_cfg.NUM_CHANNEL_IN_MLP_AGENT,
            num_layers=self.model_cfg.NUM_LAYER_IN_MLP_AGENT,
            out_channels=self.model_cfg.D_MODEL
        )
        self.map_polyline_encoder = self.build_polyline_encoder(
            in_channels=self.model_cfg.NUM_INPUT_ATTR_MAP,
            hidden_dim=self.model_cfg.NUM_CHANNEL_IN_MLP_MAP,
            num_layers=self.model_cfg.NUM_LAYER_IN_MLP_MAP,
            num_pre_layers=self.model_cfg.NUM_LAYER_IN_PRE_MLP_MAP,
            out_channels=self.model_cfg.D_MODEL
        )

        self.joint_encode = self.model_cfg.JOINT_DECODE

        # build transformer encoder layers
        self.use_local_attn = self.model_cfg.get('USE_LOCAL_ATTN', False)
        self_attn_layers = []
        for _ in range(self.model_cfg.NUM_ATTN_LAYERS):
            self_attn_layers.append(self.build_transformer_encoder_layer(
                d_model=self.model_cfg.D_MODEL,
                nhead=self.model_cfg.NUM_ATTN_HEAD,
                dropout=self.model_cfg.get('DROPOUT_OF_ATTN', 0.1),
                normalize_before=False,
                use_local_attn=self.use_local_attn,
                use_rel_pos=self.joint_encode
            ))

        self.self_attn_layers = nn.ModuleList(self_attn_layers)
        self.num_out_channels = self.model_cfg.D_MODEL

    def build_polyline_encoder(self, in_channels, hidden_dim, num_layers, num_pre_layers=1, out_channels=None):
        ret_polyline_encoder = polyline_encoder.PointNetPolylineEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_pre_layers=num_pre_layers,
            out_channels=out_channels
        )
        return ret_polyline_encoder

    def build_transformer_encoder_layer(self, d_model, nhead, dropout=0.1, normalize_before=False, use_local_attn=False,
        ctx_rpe_query=None, ctx_rpe_key=None, ctx_rpe_value=None, use_rel_pos=False):
        single_encoder_layer = transformer_encoder_layer.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout,
            normalize_before=normalize_before, use_local_attn=use_local_attn,
            ctx_rpe_query=ctx_rpe_query, ctx_rpe_key=ctx_rpe_key, ctx_rpe_value=ctx_rpe_value,
            use_rel_pos=use_rel_pos
        )
        return single_encoder_layer

    def apply_global_attn(self, x, x_mask, x_pos):
        """

        Args:
            x (batch_size, N, d_model):
            x_mask (batch_size, N):
            x_pos (batch_size, N, 3):
        """
        assert torch.all(x_mask.sum(dim=-1) > 0)

        batch_size, N, d_model = x.shape
        x_t = x.permute(1, 0, 2)
        x_mask_t = x_mask.permute(1, 0, 2)
        x_pos_t = x_pos.permute(1, 0, 2)
 
        pos_embedding = position_encoding_utils.gen_sineembed_for_position(x_pos_t, hidden_dim=d_model)

        for k in range(len(self.self_attn_layers)):
            x_t = self.self_attn_layers[k](
                src=x_t,
                src_key_padding_mask=~x_mask_t,
                pos=pos_embedding
            )
        x_out = x_t.permute(1, 0, 2)  # (batch_size, N, d_model)
        return x_out

    def apply_local_attn(self, x, x_mask, x_pos, num_of_neighbors):
        """

        Args:
            x (batch_size, N, d_model):
            x_mask (batch_size, N):
            x_pos (batch_size, N, 3):
        """
        assert torch.all(x_mask.sum(dim=-1) > 0)
        batch_size, N, d_model = x.shape

        x_stack_full = x.view(-1, d_model)  # (batch_size * N, d_model)
        x_mask_stack = x_mask.view(-1)
        x_pos_stack_full = x_pos.view(-1, 3)
        batch_idxs_full = torch.arange(batch_size).type_as(x)[:, None].repeat(1, N).view(-1).int()  # (batch_size * N)

        # filter invalid elements
        x_stack = x_stack_full[x_mask_stack]
        x_pos_stack = x_pos_stack_full[x_mask_stack]
        batch_idxs = batch_idxs_full[x_mask_stack]

        # knn
        batch_offsets = common_utils.get_batch_offsets(batch_idxs=batch_idxs, bs=batch_size).int()  # (batch_size + 1)
        batch_cnt = batch_offsets[1:] - batch_offsets[:-1]

        # pos_stack = x_pos_stack[:, :2].contiguous()
        index_pair = knn_utils.knn_batch_mlogk(
            x_pos_stack, x_pos_stack,  batch_idxs, batch_offsets, num_of_neighbors
        )  # (num_valid_elems, K)

        pos_embedding = None

        if self.joint_encode:
            query_attn_pos = grouping_operation(
                x_pos_stack, batch_cnt, index_pair, batch_cnt).permute(0, 2, 1).contiguous()
                # (num_valid_elems, K, 3)
            rel_attn_pos = query_attn_pos - x_pos_stack[:, None, :]
            heading =  x_pos_stack[..., 2]
            heading = heading[..., None]

            cos, sin = torch.cos(heading), torch.sin(heading)
            rel_x, rel_y, rel_heading = rel_attn_pos[..., 0], rel_attn_pos[..., 1], rel_attn_pos[..., 2]
            new_x = cos * rel_x + sin * rel_y
            new_y = -sin * rel_x + cos * rel_y
            rel_heading = common_utils.wrap_to_pi(rel_heading)
            rel_pos = torch.stack([new_x, new_y, torch.cos(rel_heading), torch.sin(rel_heading)], dim=-1)

            pos_embedding = position_encoding_utils.gen_sineembed_for_position(rel_pos[:, :, :4], 
                hidden_dim=d_model)

            # positional encoding
            x_pos_stack = torch.stack([x_pos_stack[:, 0], x_pos_stack[:, 1], 
                torch.cos(x_pos_stack[:, 2]), torch.sin(x_pos_stack[:, 2])], dim=-1)
        else:
            x_pos_stack = x_pos_stack[:, :2]

        abs_pos_embedding = position_encoding_utils.gen_sineembed_for_position(x_pos_stack[None], 
            hidden_dim=d_model)[0]
        
        # local attn
        output = x_stack
        for k in range(len(self.self_attn_layers)):
            output = self.self_attn_layers[k](
                src=output,
                pos=abs_pos_embedding,
                index_pair=index_pair,
                query_batch_cnt=batch_cnt,
                key_batch_cnt=batch_cnt,
                index_pair_batch=batch_idxs,
                rpe_distance=pos_embedding
            )

        ret_full_feature = torch.zeros_like(x_stack_full)  # (batch_size * N, d_model)
        ret_full_feature[x_mask_stack] = output

        ret_full_feature = ret_full_feature.view(batch_size, N, d_model)
        return ret_full_feature
    
    def cal_map_polyline_center(self, map_polylines, map_polylines_center, polyline_mask):
        """
        map_polylines :[num_center_obj, num_poly_lines, len_seg, 9]
        [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]
        """
        # seg_center = torch.mean(map_polylines[..., :2], dim=-2)
        batch, num_poly, poly_len = polyline_mask.shape
        seg_center = map_polylines_center[..., :2]
        dist = torch.linalg.norm(map_polylines[..., :2] - seg_center[..., None, :2], dim=-1)
        dist[polyline_mask==0] = 10000
        min_ind = torch.min(dist, dim=-1)[1] #[b, num_poly]
        min_poly = map_polylines[torch.arange(batch)[:, None, None], torch.arange(num_poly)[None, :, None], min_ind.unsqueeze(-1)][:, :, 0]
        center_angle = common_utils.wrap_to_pi(torch.atan2(min_poly[..., 4], min_poly[..., 3].clamp(min=1e-3)))
        return seg_center, center_angle
    
    def traj_rotate_along_z(self, obj_trajs):
        """
        traj: [batch, t, 29]
        x.center_x, x.center_y, x.center_z, x.length, x.width, x.height, x.heading,
        x.velocity_x, x.velocity_y, x.valid
        should rotate cx, cy, dx, dy, 
        """
        curr_xy, curr_head = obj_trajs[..., -1, :2].clone() , obj_trajs[..., -1, -1].clone()
        traj_xy = obj_trajs[..., :2] - curr_xy[..., None, :2]

      
        rotated_xy = common_utils.poly_left_hand_rotations(traj_xy, curr_head)
        rotated_vxy = common_utils.poly_left_hand_rotations(obj_trajs[..., [24, 25]], curr_head)

        vel_pre = torch.roll(rotated_vxy, shifts=1, dims=-2)
        acce = (rotated_vxy - vel_pre) / 0.1  # (num_centered_objects, num_objects, num_timestamps, 2)
        acce[:, :, 0, :] = acce[:, :, 1, :]

        obj_trajs[..., :2] = rotated_xy
        obj_trajs[..., [24, 25]] = rotated_vxy
        obj_trajs[..., [26, 27]] = acce
        full_heading = obj_trajs[..., -1].clone()
        rel_head = common_utils.wrap_to_pi(full_heading - curr_head[..., None])
        obj_trajs[..., -1] = rel_head
        obj_trajs[..., 22] = torch.sin(rel_head)
        obj_trajs[..., 23] = torch.cos(rel_head)

        curr_pos = torch.stack([curr_xy[..., 0], curr_xy[..., 1], curr_head], dim=-1)
        return obj_trajs[..., :-1], curr_pos
    
    def poly_rotate_along_z(self, polylines, poly_center, polyline_mask):
        """
        map_polylines :[num_center_obj, num_poly_lines, len_seg, 9]
        [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]
        """
        seg_center, center_angle = self.cal_map_polyline_center(polylines, poly_center, polyline_mask)
        curr_xy = polylines[..., :2] - seg_center[..., None, :2]
        prev_xy = polylines[..., [7, 8]] - seg_center[..., None, :2]

        poly_xy = common_utils.poly_left_hand_rotations(curr_xy, center_angle)
        dir_xy = common_utils.poly_left_hand_rotations(polylines[..., [3, 4]], center_angle)
        prev_xy = common_utils.poly_left_hand_rotations(prev_xy, center_angle)

        polylines[..., :2] = poly_xy

        polylines[..., [3, 4]] = dir_xy
        polylines[..., [7, 8]] = prev_xy

        curr_pos = torch.stack([seg_center[..., 0], seg_center[..., 1], center_angle], dim=-1)

        return polylines, curr_pos
    

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
              input_dict:
        """
        input_dict = batch_dict['input_dict']
        obj_trajs, obj_trajs_mask = input_dict['obj_trajs'].cuda(), input_dict['obj_trajs_mask'].cuda() 
        map_polylines, map_polylines_mask = input_dict['map_polylines'].cuda(), input_dict['map_polylines_mask'].cuda() 

        obj_trajs_last_pos = input_dict['obj_trajs_last_pos'].cuda() 
        map_polylines_center = input_dict['map_polylines_center'].cuda() 
        # print( obj_trajs_last_pos[0])

        #make respective rotations:
        if self.joint_encode:
            map_polylines, map_polylines_center = self.poly_rotate_along_z(map_polylines, map_polylines_center, map_polylines_mask)
            obj_trajs, obj_trajs_last_pos = self.traj_rotate_along_z(obj_trajs)
        
        track_index_to_predict = input_dict['track_index_to_predict']

        assert obj_trajs_mask.dtype == torch.bool and map_polylines_mask.dtype == torch.bool

        num_center_objects, num_objects, num_timestamps, _ = obj_trajs.shape
        num_polylines = map_polylines.shape[1]


        # apply polyline encoder
        obj_trajs_in = torch.cat((obj_trajs, obj_trajs_mask[:, :, :, None].type_as(obj_trajs)), dim=-1)
        obj_polylines_feature = self.agent_polyline_encoder(obj_trajs_in, obj_trajs_mask)  # (num_center_objects, num_objects, C)
        map_polylines_feature = self.map_polyline_encoder(map_polylines, map_polylines_mask)  # (num_center_objects, num_polylines, C)

        # apply self-attn
        obj_valid_mask = (obj_trajs_mask[..., -1] > 0)  # (num_center_objects, num_objects)
        map_valid_mask = (map_polylines_mask.sum(dim=-1) > 0)  # (num_center_objects, num_polylines)

        global_token_feature = torch.cat((obj_polylines_feature, map_polylines_feature), dim=1) 
        global_token_mask = torch.cat((obj_valid_mask, map_valid_mask), dim=1) 
        global_token_pos = torch.cat((obj_trajs_last_pos, map_polylines_center), dim=1) 

        if self.use_local_attn:
            global_token_feature = self.apply_local_attn(
                x=global_token_feature, x_mask=global_token_mask, x_pos=global_token_pos,
                num_of_neighbors=self.model_cfg.NUM_OF_ATTN_NEIGHBORS
            )
        else:
            global_token_feature = self.apply_global_attn(
                x=global_token_feature, x_mask=global_token_mask, x_pos=global_token_pos
            )

        obj_polylines_feature = global_token_feature[:, :num_objects]
        map_polylines_feature = global_token_feature[:, num_objects:]
        assert map_polylines_feature.shape[1] == num_polylines

        # organize return features
        if self.joint_encode:
            center_objects_feature = obj_polylines_feature[torch.arange(num_center_objects)[:, None], track_index_to_predict]
            batch_dict['center_obj_pos'] = obj_trajs_last_pos[torch.arange(num_center_objects)[:, None], track_index_to_predict]
        else:
            center_objects_feature = obj_polylines_feature[torch.arange(num_center_objects), track_index_to_predict]
            batch_dict['center_obj_pos'] = obj_trajs_last_pos[torch.arange(num_center_objects), track_index_to_predict]


        batch_dict['center_objects_feature'] = center_objects_feature
        batch_dict['obj_feature'] = obj_polylines_feature
        batch_dict['map_feature'] = map_polylines_feature
        batch_dict['obj_mask'] = obj_valid_mask
        batch_dict['map_mask'] = map_valid_mask
        batch_dict['obj_pos'] = obj_trajs_last_pos
        batch_dict['map_pos'] = map_polylines_center

        return batch_dict