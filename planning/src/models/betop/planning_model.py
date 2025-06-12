'''
Behavioral Topology (BeTop): https://arxiv.org/abs/2409.18031
'''
'''
Pipeline developed upon planTF: 
https://arxiv.org/pdf/2309.10443
'''
import torch
import torch.nn as nn
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)

from src.models.betop.layers.common_layers import build_mlp
from src.models.betop.layers.transformer_encoder_layer import TransformerEncoderLayer

from src.models.betop.modules.agent_encoder import AgentEncoder
from src.models.betop.modules.map_encoder import MapEncoder
from src.models.betop.modules.trajectory_decoder import TrajectoryDecoder, PredTrajectoryDecoder, NewContigencyDecoder, ContigencyDecoder

from src.models.betop.modules.topo_decoder import OccupancyFuser, OccupancyDecoder

# no meaning, required by nuplan
trajectory_sampling = TrajectorySampling(num_poses=8, time_horizon=8, interval_length=1)
from time import time


class PlanningModel(TorchModuleWrapper):
    def __init__(
        self,
        dim=128,
        state_channel=6,
        polygon_channel=6,
        history_channel=9,
        history_steps=21,
        future_steps=80,
        encoder_depth=4,
        drop_path=0.2,
        drop_key=0.3,
        num_heads=8,
        num_modes=6,
        use_ego_history=False,
        state_attn_encoder=True,
        state_dropout=0.75,
        joint_pred=False,
        rel_pred=False,
        feature_builder=None,
        planner=None,
        occ_pred=False,
        conti_plan=False,
        traj_step=5,
        multi_pred=False,
        conti_loss=False,
        marginal_mode=6,
    ) -> None:
        super().__init__(

            feature_builders=[feature_builder],
            target_builders=[EgoTrajectoryTargetBuilder(trajectory_sampling)],
            future_trajectory_sampling=trajectory_sampling,
        )

        self.dim = dim
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.rel_pred = rel_pred
        self.conti_loss = conti_loss

        self.pos_emb = build_mlp(5 if self.rel_pred else 4, [dim] * 2)
        
        self.agent_encoder = AgentEncoder(
            state_channel=state_channel,
            history_channel=history_channel,
            dim=dim,
            hist_steps=history_steps,
            drop_path=drop_path,
            use_ego_history=use_ego_history,
            state_attn_encoder=state_attn_encoder,
            state_dropout=state_dropout,
            perspect_norm=rel_pred
        )

        self.map_encoder = MapEncoder(
            dim=dim,
            polygon_channel=polygon_channel,
            perspect_norm=rel_pred
        )


        self.encoder_blocks = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp, drop_key=drop_key)
            for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        )

        self.norm = nn.LayerNorm(dim)

        if conti_plan:
            self.trajectory_decoder = NewContigencyDecoder(
                embed_dim=dim,
                num_modes=num_modes,
                marginal_mode=marginal_mode,
                future_steps=future_steps,
                out_channels=4,
                top_trajs=6, traj_input=4, traj_step=traj_step, multi_agent=True
            )
        else:
            self.trajectory_decoder = TrajectoryDecoder(
                embed_dim=dim,
                num_modes=num_modes,
                future_steps=future_steps,
                out_channels=4,
            )
        self.joint_pred = joint_pred
        self.occ_pred = occ_pred
        self.conti_plan = conti_plan

        if self.occ_pred:
            self.rel_pos_emb = build_mlp(5, [dim] * 2)
            self.occ_fuser = nn.ModuleList([OccupancyFuser(dim, 0.1) for _ in range(encoder_depth)])
            self.occ_decoder = OccupancyDecoder(dim, 0.0)
        
        self.multi_pred = multi_pred
        if self.joint_pred:
            self.agent_predictor = PredTrajectoryDecoder(
                embed_dim=dim,
                num_modes=num_modes,
                future_steps=future_steps,
                out_channels=4,
            )
        else:
            if self.multi_pred:
                self.agent_predictor = PredTrajectoryDecoder(
                    embed_dim=dim,
                    num_modes=num_modes,
                    future_steps=future_steps,
                    out_channels=4,
                )
            else:
                self.agent_predictor = build_mlp(dim, [dim * 2, future_steps * 2], norm="ln")
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    
    def _cosine(self, v1, v2):
        ''' 
        input: [B, M, N, 2], [B, M, N, 2]
        output: [B, M, N]
        cos(<a,b>) = (a dot b) / |a||b|
        '''
        v1_norm = torch.linalg.norm(v1, dim=-1)
        v2_norm = torch.linalg.norm(v2, dim=-1)
        v1_x, v1_y = v1[..., 0], v1[..., 1]
        v2_x, v2_y = v2[..., 0], v2[..., 1]
        cos = (v1_x * v2_x + v1_y * v2_y) / (v1_norm * v2_norm + 1e-10)
        return cos

    def _sine(self, v1, v2):
        ''' input: [B, M, N, 2], [B, M, N, 2]
            output: [B, M, N]
            sin(<a,b>) = (a x b) / |a||b|
        '''
        v1_norm = torch.linalg.norm(v1, dim=-1)
        v2_norm = torch.linalg.norm(v2, dim=-1)
        v1_x, v1_y = v1[..., 0], v1[..., 1]
        v2_x, v2_y = v2[..., 0], v2[..., 1]
        sin = (v1_x * v2_y - v1_y * v2_x) / (v1_norm * v2_norm + 1e-10)
        return sin
    
    def build_rel_feature(self, pos, radius=100.0):
        position, vec = pos[..., :2], pos[..., -2:]
        pos_diff = position[:, :, None, :] - position[:, None, :, :]
        d_pos = torch.linalg.norm(pos_diff, dim=-1)
        d_pos = d_pos * 2 / radius

        # angle diff
        cos_a1 = self._cosine(vec[:, :, None, :2], vec[:,  None, :, :2])
        sin_a1 = self._sine(vec[:, :, None, :2], vec[:,  None, :, :2])

        cos_a2 = self._cosine(vec[:, :, None, :2], pos_diff)
        sin_a2 = self._sine(vec[:, :, None, :2], pos_diff)

        return torch.stack([cos_a1, sin_a1, cos_a2, sin_a2, d_pos], dim=-1)

    def forward(self, data):

        agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
        agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
        agent_mask = data["agent"]["valid_mask"][:, :, : self.history_steps]
        polygon_center = data["map"]["polygon_center"]
        polygon_mask = data["map"]["valid_mask"]            

        bs, A = agent_pos.shape[0:2]

        position = torch.cat([agent_pos, polygon_center[..., :2]], dim=1)
        angle = torch.cat([agent_heading, polygon_center[..., 2]], dim=1)
        pos = torch.cat(
            [position, torch.stack([angle.cos(), angle.sin()], dim=-1)], dim=-1
        )

        if self.rel_pred or self.occ_pred:
            val_agt = agent_mask.any(-1)
            val_map = polygon_mask.any(-1)
            val_f_mask = torch.cat([val_agt, val_map], dim=-1)
            val_rel_mask = val_f_mask[:, :, None] * val_f_mask[:, None, :]

            rel_pos = self.build_rel_feature(pos)
            rel_pos = rel_pos.detach() * val_rel_mask[..., None].float()

        if self.rel_pred:
            pos = rel_pos
        if self.occ_pred:
            occ_feat = self.rel_pos_emb(rel_pos[:, :A, :, :])

        pos_embed = self.pos_emb(pos)

        if not self.rel_pred:
            agent_key_padding = ~(agent_mask.any(-1))
            polygon_key_padding = ~(polygon_mask.any(-1))
        else:
            agent_key_padding = agent_mask.any(-1)
            polygon_key_padding = polygon_mask.any(-1)

        key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1)
        if self.rel_pred:
            key_padding_mask = key_padding_mask[:, :, None] * key_padding_mask[:, None, :]

        x_agent = self.agent_encoder(data)
        x_polygon = self.map_encoder(data)

        x = torch.cat([x_agent, x_polygon], dim=1)

        if not self.rel_pred:
            x = x + pos_embed
        else:
            b, length, d = x.shape
            edge = pos_embed

        i = 0
        for blk in self.encoder_blocks:
            if not self.rel_pred:
                x = blk(x, key_padding_mask=key_padding_mask)
            else:
                x, edge = blk(x, edge_mask=key_padding_mask, edge=edge)
            if self.occ_pred:
                occ_feat = self.occ_fuser[i](src_feat=x[:, :A], tgt_feat=x, prev_occ_feat=occ_feat)
                i+= 1

        if self.rel_pred:
            x = x.view(b, length, d)
            
        x = self.norm(x)

        if self.conti_plan:
            joint_plan, trajectory, probability = self.trajectory_decoder(x[:, 0])
        else:
            trajectory, probability = self.trajectory_decoder(x[:, 0])
        if self.joint_pred:
            prediction, pred_probability = self.agent_predictor(x[:, 1:A])
        else:
            if self.multi_pred:
                prediction, pred_probability = self.agent_predictor(x[:, 1:A])
            else:
                prediction = self.agent_predictor(x[:, 1:A]).view(bs, -1, self.future_steps, 2)
        
        if self.occ_pred:
            actor_o, actor_map_o = self.occ_decoder(occ_feat, A)

        out = {
            "trajectory": trajectory,
            "probability": probability,
            "prediction": prediction,
            'rel_pred': self.rel_pred,
            'occ_pred':self.occ_pred,
            'conti_plan':self.conti_plan,
            'conti_loss':self.conti_loss
        }
        
        if self.occ_pred:
            out['actor_occ'] = actor_o
            out['actor_map_occ'] = actor_map_o

        if self.joint_pred or self.multi_pred:
            out["pred_probability"] = pred_probability
        
        if self.conti_plan:
            out['joint_plan'] = joint_plan

        if not self.training:
            best_mode = probability.argmax(dim=-1)
            output_trajectory = trajectory[torch.arange(bs), best_mode]
            angle = torch.atan2(output_trajectory[..., 3], output_trajectory[..., 2])
            out["output_trajectory"] = torch.cat(
                [output_trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
            ),
            full_angle = torch.atan2(trajectory[..., 3], trajectory[..., 2])
            
            out["full_trajectory"] = torch.cat(
                [trajectory[..., :2], full_angle.unsqueeze(-1)], dim=-1
            )

        return out

