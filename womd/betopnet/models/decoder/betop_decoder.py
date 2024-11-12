'''
Behavioral Topology (BeTop): https://arxiv.org/abs/2409.18031
'''
'''
Pipeline developed upon Motion Transformer (MTR): 
https://arxiv.org/abs/2209.13508
'''

import copy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from betopnet.models.utils.transformer import transformer_decoder_layer
from betopnet.models.utils.transformer import position_encoding_utils
from betopnet.models.utils.transformer.topo_attention import (
    agent_topo_indexing, map_topo_indexing, apply_topo_attention
)
from betopnet.models.decoder.topo_decoder import TopoFuser, TopoDecoder

from betopnet.models.utils import common_layers
from betopnet.utils import common_utils, loss_utils, motion_utils, topo_utils
from betopnet.config import cfg

import numpy as np


class BeTopDecoder(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        self.model_cfg = config
        self.object_type = self.model_cfg.OBJECT_TYPE
        self.num_future_frames = self.model_cfg.NUM_FUTURE_FRAMES
        self.num_motion_modes = self.model_cfg.NUM_MOTION_MODES
        self.end_to_end = self.model_cfg.get('END_TO_END', False)
        self.d_model = self.model_cfg.D_MODEL
        self.num_decoder_layers = self.model_cfg.NUM_DECODER_LAYERS

        self.num_inter_layers = 2
        self.distinct_anchors = self.model_cfg.get('DISTINCT_ANCHORS', True)
        self.multi_step = 1

        self.num_topo = self.model_cfg.get('NUM_TOPO', 0)

        self.type_dict = {
            'TYPE_VEHICLE':0 , 'TYPE_PEDESTRIAN':1 , 'TYPE_CYCLIST':2
        }

        # define the cross-attn layers
        self.in_proj_center_obj = nn.Sequential(
            nn.Linear(in_channels, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )

        actor_d_model = self.model_cfg.get('ACTOR_D_MODEL', self.d_model)
        # build agent decoders
        self.in_proj_obj, self.obj_decoder_layers = self.build_transformer_decoder(
            in_channels=in_channels,
            d_model=actor_d_model,
            nhead=self.model_cfg.NUM_ATTN_HEAD,
            dropout=self.model_cfg.get('DROPOUT_OF_ATTN', 0.1),
            num_decoder_layers=self.num_decoder_layers,
            use_local_attn=True
        )
        # build agent decoder MLPs
        self.build_agent_decoder_mlp(actor_d_model)


        map_d_model = self.model_cfg.get('MAP_D_MODEL', self.d_model)
        # build map decoders
        self.in_proj_map, self.map_decoder_layers = self.build_transformer_decoder(
            in_channels=in_channels,
            d_model=map_d_model,
            nhead=self.model_cfg.NUM_ATTN_HEAD,
            dropout=self.model_cfg.get('DROPOUT_OF_ATTN', 0.1),
            num_decoder_layers=self.num_decoder_layers,
            use_local_attn=True
        )
        # build map decoder MLPs
        self.build_map_decoder_mlp(map_d_model)
        
        # define the dense future prediction layers
        self.build_dense_future_prediction_layers(
            hidden_dim=self.d_model, 
            num_future_frames=self.num_future_frames,
            actor_d_model=actor_d_model
        )

        # build Topo decoders, fusers and cross-layers:
        self.build_topo_layers(
            self.d_model, 
            map_d_model,actor_d_model, 
            0.1, self.num_decoder_layers)

        # define the motion query
        self.intention_points, self.intention_query, \
            self.intention_query_mlps = self.build_motion_query(self.d_model)
        if self.end_to_end:
            self.agent_type_embed = nn.Embedding(3, self.d_model)
            self.agent_init_embed = nn.Embedding(6, self.d_model)

        # define the motion head
        temp_layer = common_layers.build_mlps(
            c_in=self.d_model*2 + map_d_model, 
            mlp_channels=[self.d_model, self.d_model], 
            ret_before_act=True)

        self.query_feature_fusion_layers = nn.ModuleList(
            [copy.deepcopy(temp_layer) for _ in range(self.num_decoder_layers)])

        self.motion_reg_heads, self.motion_cls_heads, self.motion_vel_heads = self.build_motion_head(
            in_channels=self.d_model, hidden_size=self.d_model, num_decoder_layers=self.num_decoder_layers
        )

        self.forward_ret_dict = {}
    
    
    def build_agent_decoder_mlp(self, actor_d_model):
        '''
        Building the decoder mlps to align with decoder dim
        for dimension expansion / reduction
        '''
        if actor_d_model != self.d_model:
            temp_layer = nn.Linear(self.d_model, actor_d_model)
            self.actor_query_content_mlps = nn.ModuleList([
                copy.deepcopy(temp_layer) for _ in range(self.num_decoder_layers)])
            temp_r_layer = nn.Linear(actor_d_model, self.d_model)
            self.actor_query_content_mlps_reverse = nn.ModuleList([
                copy.deepcopy(temp_r_layer) for _ in range(self.num_decoder_layers)])
            self.actor_query_embed_mlps = nn.Linear(self.d_model, actor_d_model)

            temp_layer = nn.Linear(self.d_model, actor_d_model)
            self.topo_actor_query_content_mlps = nn.ModuleList([
                copy.deepcopy(temp_layer) for _ in range(self.num_decoder_layers)])
        else:
            self.actor_query_content_mlps_reverse = [None] * self.num_decoder_layers
            self.actor_query_content_mlps = [None] * self.num_decoder_layers
            self.actor_query_embed_mlps = None
        
            self.topo_actor_query_content_mlps = [None] * self.num_decoder_layers
    

    def build_map_decoder_mlp(self, map_d_model):
        '''
        Building the decoder mlps to align with decoder dim
        for dimension expansion / reduction
        '''
        if map_d_model != self.d_model:
            temp_layer = nn.Linear(self.d_model, map_d_model)
            self.map_query_content_mlps = nn.ModuleList([
                copy.deepcopy(temp_layer) for _ in range(self.num_decoder_layers)])
            self.map_query_embed_mlps = nn.Linear(self.d_model, map_d_model)

            temp_layer = nn.Linear(self.d_model, map_d_model)
            self.topo_map_content_mlps = nn.ModuleList([
                copy.deepcopy(temp_layer) for _ in range(self.num_decoder_layers)])
        else:
            self.map_query_content_mlps = [None] * self.num_decoder_layers
            self.map_query_embed_mlps = None
            self.topo_map_content_mlps = [None] * self.num_decoder_layers

    
    def build_topo_layers(self, d_model, d_map_model, actor_d_model, dropout=0.1, num_decoder_layers=1):

        self.actor_topo_fusers = nn.ModuleList(
            [TopoFuser(actor_d_model, actor_d_model//2, dropout) for _ in range(num_decoder_layers)]
            )
        
        self.map_topo_fusers = nn.ModuleList(
            [TopoFuser(d_map_model, d_map_model//2, dropout) for _ in range(num_decoder_layers)]
            )
        
        self.actor_topo_decoders = nn.ModuleList(
            [TopoDecoder(actor_d_model//2, dropout, self.multi_step) for _ in range(num_decoder_layers)]
            )
        
        self.map_topo_decoders = nn.ModuleList(
            [TopoDecoder(d_map_model//2, dropout, self.multi_step) for _ in range(num_decoder_layers)]
            )


    def build_dense_future_prediction_layers(self, hidden_dim, num_future_frames, actor_d_model):
        self.obj_pos_encoding_layer = common_layers.build_mlps(
            c_in=2, mlp_channels=[hidden_dim, hidden_dim, hidden_dim], ret_before_act=True, without_norm=True
        )
        self.dense_future_head = common_layers.build_mlps(
            c_in=hidden_dim + actor_d_model,
            mlp_channels=[hidden_dim, hidden_dim, num_future_frames * 7], ret_before_act=True
        )

        self.future_traj_mlps = common_layers.build_mlps(
            c_in=4 * self.num_future_frames, mlp_channels=[hidden_dim, 
            hidden_dim, hidden_dim], ret_before_act=True, without_norm=True
        )
        self.traj_fusion_mlps = common_layers.build_mlps(
            c_in=hidden_dim + actor_d_model, mlp_channels=[hidden_dim, 
                hidden_dim, actor_d_model], ret_before_act=True, without_norm=True
        )


    def build_transformer_decoder(self, in_channels, d_model, 
        nhead, dropout=0.1, num_decoder_layers=1, use_local_attn=False):
        in_proj_layer = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        decoder_layer = transformer_decoder_layer.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout,
            activation="relu", normalize_before=False, keep_query_pos=False,
            rm_self_attn_decoder=False, use_local_attn=use_local_attn
        )
        decoder_layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_decoder_layers)])
        return in_proj_layer, decoder_layers


    def build_motion_query(self, d_model):

        intention_points = intention_query = intention_query_mlps = None
        intention_points_file = cfg.ROOT_DIR / self.model_cfg.INTENTION_POINTS_FILE
        # for End-to-end decoding, use the 6 K-means anchors instead
        with open(intention_points_file, 'rb') as f:
            intention_points_dict = pickle.load(f)

        intention_points = {}
        for cur_type in self.object_type:
            cur_intention_points = intention_points_dict[cur_type]
            cur_intention_points = torch.from_numpy(cur_intention_points).float().view(-1, 2).cuda()
            intention_points[cur_type] = cur_intention_points

        intention_query_mlps = common_layers.build_mlps(
            c_in=d_model, mlp_channels=[d_model, d_model], ret_before_act=True
        )
        return intention_points, intention_query, intention_query_mlps


    def build_motion_head(self, in_channels, hidden_size, num_decoder_layers):
        motion_reg_head =  common_layers.build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_size, hidden_size, self.num_future_frames * 7], ret_before_act=True
        )
        motion_cls_head =  common_layers.build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_size, hidden_size, 1], ret_before_act=True
        )

        motion_reg_heads = nn.ModuleList([copy.deepcopy(motion_reg_head) for _ in range(num_decoder_layers)])
        motion_cls_heads = nn.ModuleList([copy.deepcopy(motion_cls_head) for _ in range(num_decoder_layers)])
        motion_vel_heads = None 
        return motion_reg_heads, motion_cls_heads, motion_vel_heads


    def apply_dense_future_prediction(self, obj_feature, obj_mask, obj_pos):
        num_center_objects, num_objects, _ = obj_feature.shape

        # dense future prediction
        obj_pos_valid = obj_pos[obj_mask][..., 0:2]
        obj_feature_valid = obj_feature[obj_mask]
        obj_pos_feature_valid = self.obj_pos_encoding_layer(obj_pos_valid)
        obj_fused_feature_valid = torch.cat((obj_pos_feature_valid, obj_feature_valid), dim=-1)

        pred_dense_trajs_valid = self.dense_future_head(obj_fused_feature_valid)
        pred_dense_trajs_valid = pred_dense_trajs_valid.view(pred_dense_trajs_valid.shape[0], 
                self.num_future_frames, 7)

        temp_center = pred_dense_trajs_valid[:, :, 0:2] + obj_pos_valid[:, None, 0:2]
        pred_dense_trajs_valid = torch.cat((temp_center, pred_dense_trajs_valid[:, :, 2:]), dim=-1)

        # future feature encoding and fuse to past obj_feature
        obj_future_input_valid = pred_dense_trajs_valid[:, :, [0, 1, -2, -1]].flatten(start_dim=1, end_dim=2) 
        obj_future_feature_valid = self.future_traj_mlps(obj_future_input_valid)

        obj_full_trajs_feature = torch.cat((obj_feature_valid, obj_future_feature_valid), dim=-1)
        obj_feature_valid = self.traj_fusion_mlps(obj_full_trajs_feature)

        ret_obj_feature = torch.zeros_like(obj_feature)
        ret_obj_feature[obj_mask] = obj_feature_valid

        ret_pred_dense_future_trajs = obj_feature.new_zeros(num_center_objects, 
        num_objects, self.num_future_frames, 7)
        ret_pred_dense_future_trajs[obj_mask] = pred_dense_trajs_valid
        self.forward_ret_dict['pred_dense_trajs'] = ret_pred_dense_future_trajs

        return ret_obj_feature, ret_pred_dense_future_trajs


    def get_motion_query(self, center_objects_type):
        num_center_objects = len(center_objects_type)

        intention_points = torch.stack([
            self.intention_points[center_objects_type[obj_idx]]
            for obj_idx in range(num_center_objects)], dim=0)
        # (num_query, num_center_objects, 2)
        intention_points = intention_points.permute(1, 0, 2)  

        if self.end_to_end:
            # use embeddings
            agent_type = np.array(
                [self.type_dict[center_objects_type[obj_idx]] 
                for obj_idx in range(num_center_objects)])
    
            agent_type = torch.from_numpy(agent_type).cuda().int()
            embed_type = self.agent_type_embed(agent_type)
            intention_query = self.agent_init_embed.weight
            intention_query = intention_query[:, None, :].repeat(1, num_center_objects, 1) 
            intention_query = intention_query + embed_type[None, :, :] 
        else:
            intention_query = position_encoding_utils.gen_sineembed_for_position(
                intention_points, hidden_dim=self.d_model)

        # (num_query, num_center_objects, D)
        intention_query = self.intention_query_mlps(
            intention_query.view(-1, self.d_model)).view(
                -1, num_center_objects, self.d_model)  
 
        return intention_query, intention_points
    
    def apply_topo_reasoning(
        self, 
        query_feat, kv_feat,
        prev_topo_feat,
        fuse_layer, 
        decoder_layer,
        query_content_pre_mlp=None,
        center_gt_positive_idx=None,
        full_preds=False
        ):
        """
        performing synergistic Topology reasoning
        Args:
            query_feat, kv_feat  [M, B, D], [B, N, D]
            prev_topo_feat, [B, M, N, D]
            fuse_layer, decoder layer: Topo decoders
            center_gt_positive_idx / full_preds:
            Efficient decoding for train-time reasoning 
        """
        
        if query_content_pre_mlp is not None:
            query_feat = query_content_pre_mlp(query_feat)

        query_feat = query_feat.permute(1, 0, 2)
        b = query_feat.shape[0]
        if self.training and not full_preds:
            query_feat = query_feat[torch.arange(b), center_gt_positive_idx][:, None]
     
        src = query_feat
        tgt = kv_feat 
        
        topo_feat = fuse_layer(src, tgt, prev_topo_feat)
        topo_pred = decoder_layer(topo_feat)

        if self.training and not full_preds:
            single_topo_pred = topo_pred
        else:
            single_topo_pred = topo_pred[torch.arange(b), center_gt_positive_idx][:, None]

        return topo_feat, single_topo_pred, topo_pred

    def apply_cross_attention(
        self, query_feat, kv_feat, kv_mask,
        query_pos_feat, kv_pos_feat, 
        pred_query_center, topo_indexing,
        attention_layer,
        query_feat_pre_mlp=None,
        query_embed_mlp=None,
        query_feat_pos_mlp=None,
        is_first=False,
        ): 

        """
        Applying the TopoAttention cross attnetion function
        Args:
            query_feat, query_pos_feat, query_searching_feat  [M, B, D]
            kv_feat, kv_pos_feat  [B, N, D]
            kv_mask [B, N]
            topo_indexing [B, N, N_top]
            attention_layer (func): LocalTransformer Layer (as in EQNet and MTR)
            query_feat_pre_mlp, query_embed_mlp, query_feat_pos_mlp (nn.Linear):
            projections to align decoder dimension
            is_first (bool): whether to concat query pos feature (as in MTR) 
        Returns:
            query_feat: (B, M, D)
        """

        if query_feat_pre_mlp is not None:
            query_feat = query_feat_pre_mlp(query_feat)
        if query_embed_mlp is not None:
            query_pos_feat = query_embed_mlp(query_pos_feat)
        
        d_model = query_feat.shape[-1]
        query_searching_feat = position_encoding_utils.gen_sineembed_for_position(
            pred_query_center, hidden_dim=d_model)
        
        query_feat = apply_topo_attention(
            attention_layer,
            query_feat,
            query_pos_feat,
            query_searching_feat,
            kv_feat,
            kv_pos_feat,
            kv_mask,
            topo_indexing,
            is_first
        )

        if query_feat_pos_mlp is not None:
            query_feat = query_feat_pos_mlp(query_feat)

        return query_feat

    def get_center_gt_idx(
        self, 
        layer_idx,
        pred_scores=None, 
        pred_trajs=None,
        pred_list=None,
        prev_trajs=None,
        prev_dist=None,
        ):
        """
        Calculating GT modality index
        Full: calculating by final displacement of anchors
        E2E: calculating by Winner-Take-All average displacement
        """
        if self.training:
            center_gt_trajs = self.forward_ret_dict['center_gt_trajs'].cuda()
            center_gt_trajs_mask = self.forward_ret_dict['center_gt_trajs_mask'].cuda()
            center_gt_final_valid_idx = self.forward_ret_dict['center_gt_final_valid_idx'].long()
            intention_points = self.forward_ret_dict['intention_points']
            num_center_objects = center_gt_trajs.shape[0]

            if self.end_to_end:
                center_gt_trajs_m = center_gt_trajs_mask.float()[:, None]
                # (num_center_objects, num_query, T)
                dist = (pred_trajs[:, :, :, :2] - center_gt_trajs[:, None, :, :2]).norm(dim=-1)  
                dist = dist * center_gt_trajs_m
                dist = dist.sum(-1) / (center_gt_trajs_m.sum(-1) + (center_gt_trajs_m.sum(-1)==0).float())
                center_gt_positive_idx = dist.argmin(dim=-1)  # (num_center_objects)
                return center_gt_positive_idx

            center_gt_goals = center_gt_trajs[torch.arange(num_center_objects), center_gt_final_valid_idx, 0:2] 
            # (num_center_objects, num_query)
            dist = (center_gt_goals[:, None, :] - intention_points).norm(dim=-1) 
            center_gt_positive_idx = dist.argmin(dim=-1)  # (num_center_objects)

            if (layer_idx//self.num_inter_layers) * self.num_inter_layers - 1 < 0:
                anchor_trajs = intention_points.unsqueeze(-2)
                select_mask = None
                if pred_list is None:
                    return center_gt_positive_idx, anchor_trajs, dist, select_mask
                if self.distinct_anchors:
                    center_gt_positive_idx, select_mask = motion_utils.select_distinct_anchors(
                        dist, pred_scores, pred_trajs, anchor_trajs
                    )
                return center_gt_positive_idx, anchor_trajs, dist, select_mask

            if self.distinct_anchors:
                # Evolving & Distinct Anchors
                if pred_list is None:
                    # For efficient topo reasoning:
                    unique_layers = set(
                        [(i//self.num_inter_layers)* self.num_inter_layers
                            for i in range(self.num_decoder_layers)]
                    )
                    if layer_idx in unique_layers:
                        anchor_trajs = pred_trajs
                        dist = ((center_gt_trajs[:, None, :, 0:2] - anchor_trajs[..., 0:2]).norm(dim=-1) * \
                             center_gt_trajs_mask[:, None]).sum(dim=-1) 
                    else:
                        anchor_trajs, dist = prev_trajs, prev_dist
                else:
                    anchor_trajs, dist = motion_utils.get_evolving_anchors(
                        layer_idx, self.num_inter_layers, pred_list, 
                        center_gt_goals, intention_points, 
                        center_gt_trajs, center_gt_trajs_mask, 
                        )

                center_gt_positive_idx, select_mask = motion_utils.select_distinct_anchors(
                    dist, pred_scores, pred_trajs, anchor_trajs
                )
        else:
            center_gt_positive_idx = None
            anchor_trajs, dist = None, None
            select_mask=None

        return center_gt_positive_idx, anchor_trajs, dist, select_mask

    def apply_transformer_decoder(
        self, center_objects_feature, center_objects_type,
        obj_feature, obj_mask, obj_pos, 
        map_feature, map_mask, map_pos):

        intention_query, intention_points = self.get_motion_query(center_objects_type)
        query_content = torch.zeros_like(intention_query)
        # (num_center_objects, num_query, 2)
        self.forward_ret_dict['intention_points'] = intention_points.permute(1, 0, 2)  

        dim = query_content.shape[-1]
        num_center_objects = query_content.shape[1]
        num_query = query_content.shape[0]

        # (num_query, num_center_objects, C)
        center_objects_feature = center_objects_feature[None, :, :].repeat(num_query, 1, 1)  

        base_map_idxs = None
        map_topo_feat = None
        actor_topo_feat = None
        center_gt_positive_idx = None
        pred_scores, pred_trajs = None, None
        anchor_trajs, anchor_dist = None, None

        pred_waypoints = intention_points.permute(1, 0, 2)[:, :, None, :]
        dynamic_query_center = intention_points

        map_pos_p = map_pos.permute(1, 0, 2)[:, :, 0:2]
        map_pos_embed = position_encoding_utils.gen_sineembed_for_position(map_pos_p, 
            hidden_dim=map_feature.shape[-1])

        obj_pos_p = obj_pos.permute(1, 0, 2)[:, :, 0:2]
        obj_pos_embed = position_encoding_utils.gen_sineembed_for_position(obj_pos_p,
             hidden_dim=obj_feature.shape[-1])

        pred_list = []

        for layer_idx in range(self.num_decoder_layers):
            
            if not self.end_to_end:
                center_gt_positive_idx, anchor_trajs, anchor_dist, _ = self.get_center_gt_idx(
                    layer_idx, pred_scores, pred_trajs, prev_trajs=anchor_trajs, prev_dist=anchor_dist
                )

            # apply BeTop reasoning / indexing
            actor_topo_feat, actor_topo_preds, full_actor_topo_preds  = self.apply_topo_reasoning(
                query_feat=query_content, kv_feat=obj_feature,
                prev_topo_feat=actor_topo_feat,
                fuse_layer=self.actor_topo_fusers[layer_idx], 
                decoder_layer=self.actor_topo_decoders[layer_idx],
                query_content_pre_mlp=self.topo_actor_query_content_mlps[layer_idx],
                center_gt_positive_idx=center_gt_positive_idx,
                full_preds=True
            )
            pred_agent_topo_idx = agent_topo_indexing(
                full_actor_topo_preds, obj_mask, max_agents=self.num_topo)
            
            map_topo_feat, map_topo_preds, full_map_topo_preds = self.apply_topo_reasoning(
                query_feat=query_content, kv_feat=map_feature,
                prev_topo_feat=map_topo_feat,
                fuse_layer=self.map_topo_fusers[layer_idx], 
                decoder_layer=self.map_topo_decoders[layer_idx],
                query_content_pre_mlp=self.topo_map_content_mlps[layer_idx],
                center_gt_positive_idx=center_gt_positive_idx,
                full_preds=False
            )
            pred_map_topo_idxs, base_map_idxs = map_topo_indexing(
                map_pos=map_pos, map_mask=map_mask,
                pred_waypoints=pred_waypoints,
                base_region_offset=self.model_cfg.CENTER_OFFSET_OF_MAP,
                num_waypoint_polylines=self.model_cfg.NUM_WAYPOINT_MAP_POLYLINES,
                num_base_polylines=self.model_cfg.NUM_BASE_MAP_POLYLINES,
                base_map_idxs=base_map_idxs,
                num_query=num_query
            )

            #apply Agent/Map TopoAttention decoder:
            agent_query_feature = self.apply_cross_attention(
                query_feat=query_content, kv_feat=obj_feature, kv_mask=obj_mask,
                query_pos_feat=intention_query, kv_pos_feat=obj_pos_embed, 
                pred_query_center=dynamic_query_center, topo_indexing=pred_agent_topo_idx,
                attention_layer=self.obj_decoder_layers[layer_idx],
                query_feat_pre_mlp=self.actor_query_content_mlps[layer_idx],
                query_embed_mlp=self.actor_query_embed_mlps,
                query_feat_pos_mlp=self.actor_query_content_mlps_reverse[layer_idx],
                is_first=layer_idx==0,    
            ) 

            map_query_feature = self.apply_cross_attention(
                query_feat=query_content, kv_feat=map_feature, kv_mask=map_mask,
                query_pos_feat=intention_query, kv_pos_feat=map_pos_embed, 
                pred_query_center=dynamic_query_center, topo_indexing=pred_map_topo_idxs,
                attention_layer=self.map_decoder_layers[layer_idx],
                query_feat_pre_mlp=self.map_query_content_mlps[layer_idx],
                query_embed_mlp=self.map_query_embed_mlps,
                is_first=layer_idx==0,   
            ) 

            # Motion prediction
            query_feature = torch.cat([
                center_objects_feature, agent_query_feature, map_query_feature
                ], dim=-1)
         
            query_content = self.query_feature_fusion_layers[layer_idx](
                query_feature.flatten(start_dim=0, end_dim=1)
            ).view(num_query, num_center_objects, -1) 

            query_content_t = query_content.permute(1, 0, 2).contiguous().view(num_center_objects * num_query, -1)
            pred_scores = self.motion_cls_heads[layer_idx](query_content_t).view(num_center_objects, num_query)
            if self.motion_vel_heads is not None:
                pred_trajs = self.motion_reg_heads[layer_idx](query_content_t).view(num_center_objects, num_query, self.num_future_frames, 5)
                pred_vel = self.motion_vel_heads[layer_idx](query_content_t).view(num_center_objects, num_query, self.num_future_frames, 2)
                pred_trajs = torch.cat((pred_trajs, pred_vel), dim=-1)
            else:
                pred_trajs = self.motion_reg_heads[layer_idx](query_content_t).view(num_center_objects, num_query, self.num_future_frames, 7)

            pred_list.append([pred_scores, pred_trajs, actor_topo_preds, map_topo_preds])
            # update
            pred_waypoints = pred_trajs.detach().clone()[:, :, :, 0:2]
            dynamic_query_center = pred_trajs.detach().clone()[:, :, -1, 0:2].contiguous().permute(1, 0, 2)  

        assert len(pred_list) == self.num_decoder_layers
        return pred_list
    
    def build_topo_gt(self, gt_trajs, gt_valid_mask, multi_step=1):
        gt_trajs = gt_trajs * gt_valid_mask[..., None].float()
        map_pos = self.forward_ret_dict['map_polylines']
        map_mask = self.forward_ret_dict['map_mask']
        polyline_mask = self.forward_ret_dict['map_polylines_mask']

        tgt_trajs = self.forward_ret_dict['obj_trajs_future_state'].cuda()
        tgt_trajs_mask = self.forward_ret_dict['obj_trajs_future_mask'].cuda()
        tgt_trajs = tgt_trajs * tgt_trajs_mask[..., None].float()

        actor_topo = topo_utils.generate_behavior_braids(gt_trajs[:, None, :, :2], tgt_trajs[..., :2], 
                gt_valid_mask[:, None], tgt_trajs_mask, multi_step)
        actor_topo_mask = torch.any(tgt_trajs_mask, dim=-1)[:, None, :]

        map_topo = topo_utils.generate_map_briads(gt_trajs[:, None, :, :2], map_pos[:, :, :, :2], 
                gt_valid_mask[:, None, :], polyline_mask, multi_step)
        map_topo_mask = map_mask[:, None, :]

        return actor_topo, actor_topo_mask, map_topo, map_topo_mask
    
    def topo_loss(
        self, 
        actor_topo, actor_topo_mask, map_topo, map_topo_mask,
        actor_topo_pred, map_topo_pred,
        ):

        actor_topo_loss = loss_utils.topo_loss(actor_topo_pred, actor_topo.detach(), 
            actor_topo_mask.float().detach(), top_k=True, top_k_ratio=0.25)
        map_topo_loss = loss_utils.topo_loss(map_topo_pred, map_topo[..., None].detach(), 
            map_topo_mask.float().detach(), top_k=True, top_k_ratio=0.25)

        return  actor_topo_loss, map_topo_loss

    def get_decoder_loss(self, tb_pre_tag=''):
        center_gt_trajs = self.forward_ret_dict['center_gt_trajs'].cuda()
        center_gt_trajs_mask = self.forward_ret_dict['center_gt_trajs_mask'].cuda()
        center_gt_final_valid_idx = self.forward_ret_dict['center_gt_final_valid_idx'].long()
        assert center_gt_trajs.shape[-1] == 4

        pred_list = self.forward_ret_dict['pred_list']
        intention_points = self.forward_ret_dict['intention_points']  # (num_center_objects, num_query, 2)

        num_center_objects = center_gt_trajs.shape[0]
        center_gt_goals = center_gt_trajs[torch.arange(num_center_objects), center_gt_final_valid_idx, 0:2]  # (num_center_objects, 2)
        
        actor_topo, actor_topo_mask, map_topo, map_topo_mask = self.build_topo_gt(
            center_gt_trajs, center_gt_trajs_mask, self.multi_step)
    
        tb_dict = {}
        disp_dict = {}
        total_loss = 0
        for layer_idx in range(self.num_decoder_layers):
     
            pred_scores, pred_trajs, actor_topo_preds, map_topo_preds = pred_list[layer_idx]  
            
            assert pred_trajs.shape[-1] == 7
            pred_trajs_gmm, pred_vel = pred_trajs[:, :, :, 0:5], pred_trajs[:, :, :, 5:7]

            center_gt_positive_idx,_,_,select_mask = self.get_center_gt_idx(
                    layer_idx, pred_scores, pred_trajs, pred_list=pred_list
                )

            loss_a_topo, loss_m_topo = self.topo_loss(
                actor_topo, actor_topo_mask, map_topo, map_topo_mask,
                actor_topo_preds, map_topo_preds,
            )
            loss_topo =  loss_a_topo + loss_m_topo

            loss_reg_gmm, center_gt_positive_idx = loss_utils.nll_loss_gmm_direct(
                pred_scores=pred_scores, pred_trajs=pred_trajs_gmm,
                gt_trajs=center_gt_trajs[:, :, 0:2], gt_valid_mask=center_gt_trajs_mask,
                pre_nearest_mode_idxs=center_gt_positive_idx,
                timestamp_loss_weight=None, use_square_gmm=False,
            )

            pred_vel = pred_vel[torch.arange(num_center_objects), center_gt_positive_idx]
            loss_reg_vel = F.l1_loss(pred_vel, center_gt_trajs[:, :, 2:4], reduction='none')
            loss_reg_vel = (loss_reg_vel * center_gt_trajs_mask[:, :, None]).sum(dim=-1).sum(dim=-1)

            bce_target = torch.zeros_like(pred_scores)
            bce_target[torch.arange(num_center_objects), center_gt_positive_idx] = 1.0
            loss_cls = F.binary_cross_entropy_with_logits(input=pred_scores, target=bce_target, reduction='none')
            loss_cls = (loss_cls * select_mask).sum(dim=-1)

            # total loss
            weight_cls = self.model_cfg.LOSS_WEIGHTS.get('cls', 1.0)
            weight_reg = self.model_cfg.LOSS_WEIGHTS.get('reg', 1.0)
            weight_vel = self.model_cfg.LOSS_WEIGHTS.get('vel', 0.2)
            weight_top = self.model_cfg.LOSS_WEIGHTS.get('top', 100)

            layer_loss = loss_reg_gmm * weight_reg + loss_reg_vel * weight_vel +\
                loss_cls.sum(dim=-1) * weight_cls + weight_top * loss_topo
           
            layer_loss = layer_loss.mean()
            total_loss += layer_loss
            tb_dict[f'{tb_pre_tag}loss_layer{layer_idx}'] = layer_loss.item()
            tb_dict[f'{tb_pre_tag}loss_layer{layer_idx}_reg_gmm'] = loss_reg_gmm.mean().item() * weight_reg
            tb_dict[f'{tb_pre_tag}loss_layer{layer_idx}_reg_vel'] = loss_reg_vel.mean().item() * weight_vel
            tb_dict[f'{tb_pre_tag}loss_layer{layer_idx}_cls'] = loss_cls.mean().item() * weight_cls
            tb_dict[f'{tb_pre_tag}loss_layer{layer_idx}_top'] = loss_topo.mean().item() * weight_top
   
            if layer_idx + 1 == self.num_decoder_layers:
                layer_tb_dict_ade = motion_utils.get_ade_of_each_category(
                    pred_trajs=pred_trajs_gmm[:, :, :, 0:2],
                    gt_trajs=center_gt_trajs[:, :, 0:2], gt_trajs_mask=center_gt_trajs_mask,
                    object_types=self.forward_ret_dict['center_objects_type'],
                    valid_type_list=self.object_type,
                    post_tag=f'_layer_{layer_idx}',
                    pre_tag=tb_pre_tag
                )
                tb_dict.update(layer_tb_dict_ade)
                disp_dict.update(layer_tb_dict_ade)

        total_loss = total_loss / self.num_decoder_layers

        return total_loss, tb_dict, disp_dict

    def get_dense_future_prediction_loss(self, tb_pre_tag='', tb_dict=None, disp_dict=None):
        obj_trajs_future_state = self.forward_ret_dict['obj_trajs_future_state'].cuda()
        obj_trajs_future_mask = self.forward_ret_dict['obj_trajs_future_mask'].cuda()
        pred_dense_trajs = self.forward_ret_dict['pred_dense_trajs'] 
        assert pred_dense_trajs.shape[-1] == 7
        assert obj_trajs_future_state.shape[-1] == 4

        pred_dense_trajs_gmm, pred_dense_trajs_vel = pred_dense_trajs[:, :, :, 0:5], pred_dense_trajs[:, :, :, 5:7]

        loss_reg_vel = F.l1_loss(pred_dense_trajs_vel, obj_trajs_future_state[:, :, :, 2:4], reduction='none')
        loss_reg_vel = (loss_reg_vel * obj_trajs_future_mask[:, :, :, None]).sum(dim=-1).sum(dim=-1)

        num_center_objects, num_objects, num_timestamps, _ = pred_dense_trajs.shape
        fake_scores = pred_dense_trajs.new_zeros((num_center_objects, num_objects)).view(-1, 1)  # (num_center_objects * num_objects, 1)

        temp_pred_trajs = pred_dense_trajs_gmm.contiguous().view(num_center_objects * num_objects, 1, num_timestamps, 5)
        temp_gt_idx = torch.zeros(num_center_objects * num_objects).cuda().long()  # (num_center_objects * num_objects)
        temp_gt_trajs = obj_trajs_future_state[:, :, :, 0:2].contiguous().view(num_center_objects * num_objects, num_timestamps, 2)
        temp_gt_trajs_mask = obj_trajs_future_mask.view(num_center_objects * num_objects, num_timestamps)
        loss_reg_gmm, _ = loss_utils.nll_loss_gmm_direct(
            pred_scores=fake_scores, pred_trajs=temp_pred_trajs, gt_trajs=temp_gt_trajs, gt_valid_mask=temp_gt_trajs_mask,
            pre_nearest_mode_idxs=temp_gt_idx,
            timestamp_loss_weight=None, use_square_gmm=False,
        )
        loss_reg_gmm = loss_reg_gmm.view(num_center_objects, num_objects)

        loss_reg = loss_reg_vel + loss_reg_gmm

        obj_valid_mask = obj_trajs_future_mask.sum(dim=-1) > 0

        loss_reg = (loss_reg * obj_valid_mask.float()).sum(dim=-1) / torch.clamp_min(obj_valid_mask.sum(dim=-1), min=1.0)
        loss_reg = loss_reg.mean()

        if tb_dict is None:
            tb_dict = {}
        if disp_dict is None:
            disp_dict = {}

        tb_dict[f'{tb_pre_tag}loss_dense_prediction'] = loss_reg.item()
        return loss_reg, tb_dict, disp_dict

    def get_loss(self, tb_pre_tag=''):

        loss_decoder, tb_dict, disp_dict = self.get_decoder_loss(tb_pre_tag=tb_pre_tag)
        loss_dense_prediction, tb_dict, disp_dict = self.get_dense_future_prediction_loss(tb_pre_tag=tb_pre_tag, tb_dict=tb_dict, disp_dict=disp_dict)

        total_loss = loss_decoder + loss_dense_prediction
        tb_dict[f'{tb_pre_tag}loss'] = total_loss.item()
        disp_dict[f'{tb_pre_tag}loss'] = total_loss.item()

        return total_loss, tb_dict, disp_dict

    def generate_final_prediction(self, pred_list):
   
        pred_scores, pred_trajs, _, _ = pred_list[-1]
        pred_scores = torch.sigmoid(pred_scores)

        if self.num_motion_modes != num_query:
            assert num_query > self.num_motion_modes
            pred_trajs_final, pred_scores_final, selected_idxs = motion_utils.inference_distance_nms(
                pred_scores, pred_trajs
            )
        else:
            pred_trajs_final = pred_trajs
            pred_scores_final = pred_scores
            selected_idxs = None
   
        return pred_scores_final, pred_trajs_final, selected_idxs

    def forward(self, batch_dict):
        input_dict = batch_dict['input_dict']
        obj_feature, obj_mask, obj_pos = batch_dict['obj_feature'], batch_dict['obj_mask'], batch_dict['obj_pos']
        map_feature, map_mask, map_pos = batch_dict['map_feature'], batch_dict['map_mask'], batch_dict['map_pos']
        center_objects_feature = batch_dict['center_objects_feature']
        num_center_objects, num_objects, _ = obj_feature.shape
        num_polylines = map_feature.shape[1]

        # input projection
        center_objects_feature = self.in_proj_center_obj(center_objects_feature)
        obj_feature_valid = self.in_proj_obj(obj_feature[obj_mask])
        obj_feature = obj_feature.new_zeros(num_center_objects, num_objects, obj_feature_valid.shape[-1])
        obj_feature[obj_mask] = obj_feature_valid

        map_feature_valid = self.in_proj_map(map_feature[map_mask])
        map_feature = map_feature.new_zeros(num_center_objects, num_polylines, map_feature_valid.shape[-1])
        map_feature[map_mask] = map_feature_valid

        # dense future prediction
        obj_feature, pred_dense_future_trajs = self.apply_dense_future_prediction(
            obj_feature=obj_feature, obj_mask=obj_mask, obj_pos=obj_pos
        )
        # decoder layers
        if self.training:
            self.forward_ret_dict['center_gt_trajs'] = input_dict['center_gt_trajs']
            self.forward_ret_dict['center_gt_trajs_mask'] = input_dict['center_gt_trajs_mask']
            self.forward_ret_dict['center_gt_final_valid_idx'] = input_dict['center_gt_final_valid_idx']
        
        pred_list = self.apply_transformer_decoder(
            center_objects_feature=center_objects_feature,
            center_objects_type=input_dict['center_objects_type'],
            obj_feature=obj_feature, obj_mask=obj_mask, obj_pos=obj_pos,
            map_feature=map_feature, map_mask=map_mask, map_pos=map_pos
        )

        self.forward_ret_dict['pred_list'] = pred_list

        if not self.training:
            pred_scores, pred_trajs, selected_idxs = self.generate_final_prediction(pred_list=pred_list)
            batch_dict['pred_scores'] = pred_scores
            batch_dict['pred_trajs'] = pred_trajs
            batch_dict['selected_idxs'] = selected_idxs

        else:
            self.forward_ret_dict['obj_trajs_future_state'] = input_dict['obj_trajs_future_state']
            self.forward_ret_dict['obj_trajs_future_mask'] = input_dict['obj_trajs_future_mask']

            self.forward_ret_dict['center_objects_type'] = input_dict['center_objects_type']

            self.forward_ret_dict['map_pos'] = map_pos
            self.forward_ret_dict['map_mask'] = map_mask

            self.forward_ret_dict['map_polylines'] = batch_dict['map_polylines']
            self.forward_ret_dict['map_polylines_mask'] = batch_dict['map_polylines_mask']
            self.forward_ret_dict['map_mask'] = batch_dict['map_mask']

        return batch_dict
