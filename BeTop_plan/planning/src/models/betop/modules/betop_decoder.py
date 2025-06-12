'''
Behavioral Topology (BeTop): https://arxiv.org/abs/2409.18031
'''
'''
Pipeline developed upon planTF: 
https://arxiv.org/pdf/2309.10443
'''

import sys

import copy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.betop.modules.utils.transformer import transformer_decoder_layer
from src.models.betop.modules.utils.transformer import position_encoding_utils
from src.models.betop.modules.utils.transformer.topo_attention import (
    agent_topo_indexing, apply_topo_attention
)
from src.models.betop.modules.decoder.topo_decoder import TopoFuser, TopoDecoder

from src.models.betop.modules.utils import common_layers
import numpy as np


class BeTopDecoder(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        self.model_cfg = config
        self.num_future_frames = self.model_cfg.get("NUM_FUTURE_FRAMES", 80)
        self.num_motion_modes = self.model_cfg.get("NUM_MOTION_MODES", 6)
        
        self.end_to_end = self.model_cfg.get('END_TO_END', True)
        self.multi_agent = self.model_cfg.get('MULTI_AGENT', False)
        self.one_out = self.model_cfg.get('ONE_OUT', False)

        self.d_model = self.model_cfg.D_MODEL
        self.num_decoder_layers = self.model_cfg.NUM_DECODER_LAYERS

        self.num_inter_layers = 2
        self.distinct_anchors = self.model_cfg.get('DISTINCT_ANCHORS', False)
        self.multi_step = 1

        self.num_topo = self.model_cfg.get('NUM_TOPO', 0)

        actor_d_model = self.model_cfg.get('ACTOR_D_MODEL', self.d_model)
        # build agent decoders
        self.in_proj_obj, self.obj_decoder_layers = self.build_transformer_decoder(
            in_channels=in_channels,
            d_model=actor_d_model,
            nhead=self.model_cfg.NUM_ATTN_HEAD,
            dropout=self.model_cfg.get('DROPOUT_OF_ATTN', 0.1),
            num_decoder_layers=self.num_decoder_layers,
            use_local_attn=False
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
            use_local_attn=False
        )
        # build map decoder MLPs
        self.build_map_decoder_mlp(map_d_model)
        
        # define the dense future prediction layers
        # self.build_dense_future_prediction_layers(
        #     hidden_dim=self.d_model, 
        #     num_future_frames=self.num_future_frames,
        #     actor_d_model=actor_d_model
        # )

        # build Topo decoders, fusers and cross-layers:
        self.build_topo_layers(
            self.d_model, 
            map_d_model,actor_d_model, 
            0.1, self.num_decoder_layers)

        # define the motion query
        # self.intention_points, self.intention_query, \
        if not self.multi_agent:
            self.intention_query_mlps = common_layers.build_mlps(
                c_in=self.d_model, mlp_channels=[self.d_model, self.d_model], ret_before_act=True
            )


        # if self.multi_agent:
        #     self.agent_init_embed = nn.Embedding(11, self.d_model)
        if not self.multi_agent:
            self.agent_init_embed = nn.Embedding(6, self.d_model)
        # self.ego_init_embed = nn.Embedding(6, self.d_model)

        # define the motion head
        temp_layer = common_layers.build_mlps(
            c_in=self.d_model*2 + map_d_model, 
            mlp_channels=[self.d_model, self.d_model], 
            ret_before_act=True)

        self.query_feature_fusion_layers = nn.ModuleList(
            [copy.deepcopy(temp_layer) for _ in range(self.num_decoder_layers)])
    
    
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
        if self.one_out:
            self.actor_topo_decoders = nn.ModuleList(
            [TopoDecoder(actor_d_model//2, dropout, self.multi_step)]
            )
        
            self.map_topo_decoders = nn.ModuleList(
                [TopoDecoder(d_map_model//2, dropout, self.multi_step)]
                )
        else:
            
            self.actor_topo_decoders = nn.ModuleList(
                [TopoDecoder(actor_d_model//2, dropout, self.multi_step) for _ in range(num_decoder_layers)]
                )
            
            self.map_topo_decoders = nn.ModuleList(
                [TopoDecoder(d_map_model//2, dropout, self.multi_step) for _ in range(num_decoder_layers)]
                )


    def build_dense_future_prediction_layers(self, hidden_dim, num_future_frames, actor_d_model):

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
        # in_proj_layer = nn.Sequential(
        #     nn.Linear(in_channels, d_model),
        #     nn.ReLU(),
        #     nn.Linear(d_model, d_model),
        # )
        in_proj_layer = None
        
        decoder_layer = transformer_decoder_layer.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout,
            activation="relu", normalize_before=False, keep_query_pos=False, attn_drop=0.,
            rm_self_attn_decoder=False, use_local_attn=use_local_attn,rm_pos=True, drop_path=0.1
        )
        init_decoder = transformer_decoder_layer.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout,
            activation="relu", normalize_before=False, keep_query_pos=False, attn_drop=0.,
            rm_self_attn_decoder=False, use_local_attn=use_local_attn, is_first=True,
            rm_pos=True, drop_path=0.1
        )
        decoder_layers = nn.ModuleList([init_decoder]+[copy.deepcopy(decoder_layer) for _ in range(num_decoder_layers-1)])
        return in_proj_layer, decoder_layers



    def apply_dense_future_prediction(self, obj_feature, prediction_feature):

        # future feature encoding and fuse to past obj_feature
        # obj_future_input_valid = prediction_feature[:, :, :, :].flatten(start_dim=2, end_dim=3).detach() 
        # obj_future_feature_valid = self.future_traj_mlps(obj_future_input_valid)

        # obj_full_trajs_feature = torch.cat((obj_feature, obj_future_feature_valid), dim=-1)
        # obj_feature_valid = self.traj_fusion_mlps(obj_full_trajs_feature)

        return obj_feature


    def get_motion_query(self, num_center_objects):

        intention_query = self.agent_init_embed.weight
        # ego_intention_query = self.ego_init_embed.weight
        # ego_intention_query = ego_intention_query[:, None, :]
        
        # (num_query, 10, C)
        # intention_query = intention_query[:, None, :].repeat(1, 10, 1) 
        # full_intention_query = torch.cat([ego_intention_query, intention_query], dim=1)
        intention_query = intention_query[:, None, :].repeat(1, num_center_objects, 1)
    
        # (num_query, num_center_objects, D)
        intention_query = self.intention_query_mlps(
            intention_query.view(-1, self.d_model)).view(
                -1, num_center_objects, self.d_model)  
 
        return intention_query
    
    def apply_topo_reasoning(
        self, 
        query_feat, kv_feat,
        prev_topo_feat,
        fuse_layer, 
        decoder_layer,
        query_content_pre_mlp=None,
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
     
        src = query_feat
        tgt = kv_feat 
        
        topo_feat = fuse_layer(src, tgt, prev_topo_feat)
        topo_pred = decoder_layer(topo_feat)

        single_topo_pred = topo_pred

        return topo_feat, single_topo_pred, topo_pred

    def apply_cross_attention(
        self, query_feat, kv_feat, kv_mask,
        query_pos_feat, kv_pos_feat, 
        pred_query_center, topo_indexing,
        attention_layer,
        query_feat_pre_mlp=None,
        query_embed_mlp=None,
        query_feat_pos_mlp=None,
        first_query_feat=None,
        is_first=False,
        sa_padding_mask=None,
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
        if first_query_feat is None:
            query_searching_feat = position_encoding_utils.gen_sineembed_for_position(
                pred_query_center, hidden_dim=d_model)
        else:
            query_searching_feat = first_query_feat
        
        
        query_feat = apply_topo_attention(
            attention_layer,
            query_feat,
            query_pos_feat,
            query_searching_feat,
            kv_feat,
            kv_pos_feat,
            kv_mask,
            topo_indexing,
            is_first,
            sa_padding_mask
        )

        if query_feat_pos_mlp is not None:
            query_feat = query_feat_pos_mlp(query_feat)

        return query_feat


    def apply_transformer_decoder(
        self, center_objects_feature, #center_objects_type,
        obj_feature, obj_mask, obj_pos, 
        map_feature, map_mask, map_pos, obj_pos_embed_input):

        batch_size = center_objects_feature.shape[0]

        if not self.multi_agent:
            intention_query = self.get_motion_query(batch_size)
        else:
            intention_query = obj_pos_embed_input.permute(1, 0, 2)
        query_content = torch.zeros_like(intention_query)
        # (num_center_objects, num_query, 2)

        num_center_objects = query_content.shape[1]
        num_query = query_content.shape[0]

        # (num_query, num_center_objects, C)
        if self.multi_agent:
            center_objects_feature = center_objects_feature.permute(1, 0, 2)
            # print(intention_query.shape, query_content.shape, center_objects_feature.shape)
            intention_query = intention_query[:center_objects_feature.shape[0]]
            intention_query += center_objects_feature
           
            query_content = query_content[:center_objects_feature.shape[0]]
            sa_padding_mask = ~obj_mask[:, :center_objects_feature.shape[0]]
        else:
            center_objects_feature = center_objects_feature[None, :, :].repeat(num_query, 1, 1)  
            sa_padding_mask= None
        
        query_content = intention_query

        map_topo_feat = None
        actor_topo_feat = None
        dynamic_query_center = None

        map_pos_p = map_pos.permute(1, 0, 2)[:, :, 0:2]
        map_pos_embed = position_encoding_utils.gen_sineembed_for_position(map_pos_p, 
            hidden_dim=map_feature.shape[-1])

        obj_pos_p = obj_pos.permute(1, 0, 2)[:, :, 0:2]
        obj_pos_embed = position_encoding_utils.gen_sineembed_for_position(obj_pos_p,
             hidden_dim=obj_feature.shape[-1])

        pred_list = []

        for layer_idx in range(self.num_decoder_layers):

            # apply BeTop reasoning / indexing
            actor_topo_feat, actor_topo_preds, full_actor_topo_preds  = self.apply_topo_reasoning(
                query_feat=query_content, kv_feat=obj_feature,
                prev_topo_feat=actor_topo_feat,
                fuse_layer=self.actor_topo_fusers[layer_idx], 
                decoder_layer=self.actor_topo_decoders[0] if self.one_out else self.actor_topo_decoders[layer_idx],
                query_content_pre_mlp=self.topo_actor_query_content_mlps[layer_idx],
                full_preds=True
            )
            pred_agent_topo_idx = agent_topo_indexing(
                full_actor_topo_preds, obj_mask, max_agents=self.num_topo)
            
            map_topo_feat, map_topo_preds, full_map_topo_preds = self.apply_topo_reasoning(
                query_feat=query_content, kv_feat=map_feature,
                prev_topo_feat=map_topo_feat,
                fuse_layer=self.map_topo_fusers[layer_idx], 
                decoder_layer=self.map_topo_decoders[0]  if self.one_out else self.map_topo_decoders[layer_idx],
                query_content_pre_mlp=self.topo_map_content_mlps[layer_idx],
                full_preds=True
            )

            pred_map_topo_idxs, base_map_idxs = None, None

            #apply Agent/Map TopoAttention decoder:
            agent_query_feature = self.apply_cross_attention(
                query_feat=query_content, kv_feat=obj_feature, kv_mask=~obj_mask,
                query_pos_feat=intention_query, kv_pos_feat=obj_pos_embed, 
                pred_query_center=dynamic_query_center, topo_indexing=pred_agent_topo_idx,
                attention_layer=self.obj_decoder_layers[layer_idx],
                query_feat_pre_mlp=self.actor_query_content_mlps[layer_idx],
                query_embed_mlp=self.actor_query_embed_mlps,
                query_feat_pos_mlp=self.actor_query_content_mlps_reverse[layer_idx],
                first_query_feat=query_content,
                is_first=layer_idx==0,    
                sa_padding_mask=sa_padding_mask
            ) 

            map_query_feature = self.apply_cross_attention(
                query_feat=query_content, kv_feat=map_feature, kv_mask=~map_mask,
                query_pos_feat=intention_query, kv_pos_feat=map_pos_embed, 
                pred_query_center=dynamic_query_center, topo_indexing=pred_map_topo_idxs,
                attention_layer=self.map_decoder_layers[layer_idx],
                query_feat_pre_mlp=self.map_query_content_mlps[layer_idx],
                query_embed_mlp=self.map_query_embed_mlps,
                first_query_feat=query_content,
                is_first=layer_idx==0,   
                sa_padding_mask=sa_padding_mask
            ) 

            # Motion prediction
            query_feature = torch.cat([
                center_objects_feature, agent_query_feature, map_query_feature
                ], dim=-1)
         
            query_content = self.query_feature_fusion_layers[layer_idx](
                query_feature.flatten(start_dim=0, end_dim=1)
            ).view(center_objects_feature.shape[0], num_center_objects, -1) 

            query_content_t = query_content.permute(1, 0, 2).contiguous()
            pred_list.append([query_content_t, actor_topo_preds, map_topo_preds])

        assert len(pred_list) == self.num_decoder_layers
        return pred_list
    

    def forward(self, batch_dict):
   
        obj_feature, obj_mask, obj_pos = batch_dict['obj_feature'], batch_dict['obj_mask'], batch_dict['obj_pos']
        map_feature, map_mask, map_pos = batch_dict['map_feature'], batch_dict['map_mask'], batch_dict['map_pos']
        center_objects_feature = batch_dict['center_objects_feature']

        obj_pos_embed_input = batch_dict['obj_pos_embed_input']

        # dense future prediction
        obj_feature = self.apply_dense_future_prediction(
            obj_feature=obj_feature, prediction_feature=batch_dict['prediction'],
        )
        # decoder layers
        pred_list = self.apply_transformer_decoder(
            center_objects_feature=center_objects_feature,
            obj_feature=obj_feature, obj_mask=obj_mask, obj_pos=obj_pos,
            map_feature=map_feature, map_mask=map_mask, map_pos=map_pos,
            obj_pos_embed_input=obj_pos_embed_input
        )

        return pred_list

def test_dummy_betop():
    from easydict import EasyDict
    cfg = EasyDict(
        D_MODEL=128,
        NUM_DECODER_LAYERS=4,
        NUM_ATTN_HEAD=8,
        END_TO_END=True,
        NUM_TOPO=32
    )
    decoder = BeTopDecoder(128, cfg)
    decoder.cuda()

    batch = 4
    tested_batch_dict = dict(
        obj_feature=torch.randn(batch, 10, 128).cuda(),
        obj_mask=torch.randn(batch, 10).bool().cuda(),
        obj_pos=torch.randn(batch, 10, 2).cuda(),
        map_feature=torch.randn(batch, 256, 128).cuda(),
        map_mask=torch.randn(batch, 256).bool().cuda(),
        map_pos=torch.randn(batch, 256, 2).cuda(),
        center_objects_feature=torch.randn(batch, 128).cuda(),
        prediction=torch.randn(batch, 10, 80, 7).cuda(),
    )

    pred_list = decoder(tested_batch_dict)
    for l in pred_list:
        print(l[0].shape, l[1].shape, l[2].shape)

if __name__ == '__main__':
    test_dummy_betop()

