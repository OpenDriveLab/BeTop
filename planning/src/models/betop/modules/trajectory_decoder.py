'''
Behavioral Topology (BeTop): https://arxiv.org/abs/2409.18031
'''
'''
Pipeline developed upon planTF: 
https://arxiv.org/pdf/2309.10443
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryDecoder(nn.Module):
    def __init__(self, embed_dim, num_modes, future_steps, out_channels) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.out_channels = out_channels

        self.multimodal_proj = nn.Linear(embed_dim, num_modes * embed_dim)

        hidden = 2 * embed_dim
        self.loc = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, future_steps * out_channels),
        )
        self.pi = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        x = self.multimodal_proj(x).view(-1, self.num_modes, self.embed_dim)
        loc = self.loc(x).view(-1, self.num_modes, self.future_steps, self.out_channels)
        pi = self.pi(x).squeeze(-1)

        return loc, pi

class ContigencyDecoder(nn.Module):
    def __init__(self, embed_dim, num_modes, future_steps, out_channels,
        top_trajs=6, traj_input=3, traj_step=15):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_modes = num_modes
        self.future_steps = future_steps - traj_step
        self.out_channels = out_channels
        self.top_trajs = top_trajs
        self.traj_step = traj_step

        self.multimodal_proj = nn.Linear(embed_dim, num_modes * embed_dim)

        self.traj_enc = nn.Sequential(
            nn.Linear(traj_input, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

        hidden = 2 * embed_dim
        self.loc = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.future_steps * out_channels),
        )
        self.pi = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
    
    def forward(self, x, traj, score):
        b = score.shape[0]
        top_score, score_idx = torch.topk(score, k=self.top_trajs, dim=-1)
        traj = traj[torch.arange(b)[:, None], score_idx, :self.traj_step ,:]

        traj_feat = self.traj_enc(traj) #[b, top, d]
        traj_feat = torch.max(traj_feat, dim=-2)[0]

        x = self.multimodal_proj(x).view(-1, self.num_modes, self.embed_dim)
        traj_feat = traj_feat[:, :, None, :].repeat(1, 1, self.num_modes, 1)
        x = x[:, None, :, :].repeat(1, self.top_trajs, 1, 1)
        comb_feat = torch.cat([x, traj_feat], dim=-1).reshape(-1, self.top_trajs*self.num_modes, self.embed_dim*2)

        loc = self.loc(comb_feat).view(-1, self.top_trajs*self.num_modes, self.future_steps, self.out_channels)
        
        cos, sin = torch.cos(traj[..., -1])[...,None], torch.sin(traj[..., -1])[...,None]
        out_traj = torch.cat([traj[..., :2], cos, sin], dim=-1)
        
        out_traj = out_traj[:, :, None, :, :].repeat(1, 1, self.num_modes, 1, 1)
        out_traj = out_traj.reshape(b, self.top_trajs*self.num_modes, self.traj_step, self.out_channels)
        out_score = top_score[:, :, None].repeat(1, 1, self.num_modes)
        max_score = out_score[:, 0]
        out_score = out_score.reshape(b, self.top_trajs*self.num_modes)
        outs = torch.cat([out_traj, loc], dim=-2)
    
        max_traj = outs.reshape(b, self.top_trajs, self.num_modes, self.future_steps + self.traj_step, self.out_channels)
        max_traj = max_traj[:, 0]
        angle = torch.atan2(max_traj[..., 3], max_traj[..., 2])
        out_max_traj = torch.cat(
                [max_traj[..., :2], angle.unsqueeze(-1)], dim=-1
            )

        pi = self.pi(comb_feat).squeeze(-1)

        return outs, pi, out_score, out_max_traj, max_score


class PredTrajectoryDecoder(nn.Module):
    def __init__(self, embed_dim, num_modes, future_steps, out_channels) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.out_channels = out_channels

        self.multimodal_proj = nn.Linear(embed_dim, embed_dim * num_modes)

        hidden = 2 * embed_dim
        self.loc = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, future_steps * out_channels),
        )
        self.pi = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        b, a, d = x.shape
        x = self.multimodal_proj(x).view(b, a, self.num_modes, self.embed_dim)
        loc = self.loc(x).view(b, a, self.num_modes, self.future_steps, self.out_channels)
        pi = self.pi(x).squeeze(-1)
        return loc, pi


class NewContigencyDecoder(nn.Module):
    def __init__(self, embed_dim, num_modes, marginal_mode, future_steps, out_channels,
        top_trajs=6, traj_input=6, traj_step=10, multi_agent=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_modes = num_modes
        self.marginal_mode = marginal_mode
        self.future_steps = future_steps - traj_step
        self.out_channels = out_channels
        self.top_trajs = top_trajs
        self.traj_step = traj_step
        self.multi_agent = multi_agent
        
        if self.multi_agent:
            self.multimodal_proj = nn.Linear(embed_dim, self.num_modes * embed_dim)
        else:
            self.multimodal_proj = nn.Linear(embed_dim, embed_dim)

        if marginal_mode != num_modes and marginal_mode!=1:
            self.marginal_proj = nn.Linear(embed_dim, marginal_mode * embed_dim)
            self.marginal_enc = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, embed_dim),
                )

        hidden = 2 * embed_dim if self.marginal_mode != 1 else embed_dim
        if self.marginal_mode != 1:
            self.traj_enc = nn.Sequential(
                nn.Linear(traj_input, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, embed_dim),
            )

            self.loc_1 = nn.Sequential(
                nn.Linear(embed_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, self.traj_step * out_channels),
            )
            self.loc = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, self.future_steps * out_channels),
            )
        else:
            self.loc_1 = nn.Sequential(
                nn.Linear(embed_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, future_steps * out_channels),
            )

        self.pi = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
    
    def forward(self, x):
        b = x.shape[0]
        x = self.multimodal_proj(x).view(-1, self.num_modes, self.embed_dim)
        if self.marginal_mode != 1:
            traj = self.loc_1(x).view(-1, self.num_modes, self.traj_step, self.out_channels)
        else:
            traj = self.loc_1(x).view(-1, self.num_modes, self.traj_step+self.future_steps, self.out_channels)
        feat_traj = traj.clone()

        if self.marginal_mode != 1:
            feat_traj[... ,:2] = torch.cumsum(feat_traj[..., :2], dim=-1)

            traj_feat = self.traj_enc(traj.detach()) #[b, top, d]
            traj_feat = torch.max(traj_feat, dim=-2)[0]

            traj_feat = traj_feat[:, :, None, :].repeat(1, 1, self.marginal_mode, 1)
            if self.marginal_mode != self.num_modes:
                x = self.marginal_proj(x)
                x = x.view(-1, self.num_modes, self.marginal_mode, self.embed_dim)
                x = self.marginal_enc(x)
            else:
                x = x[:, None, :, :].repeat(1, self.num_modes, 1, 1)
    
            comb_feat = torch.cat([x, traj_feat], dim=-1).reshape(-1, self.num_modes*self.marginal_mode, self.embed_dim*2)

            loc = self.loc(comb_feat).view(-1, self.num_modes*self.marginal_mode, self.future_steps, self.out_channels)
            out_traj = traj[:, :, None, :, :].repeat(1, 1, self.marginal_mode, 1, 1)
            out_traj = out_traj.reshape(b, self.num_modes*self.marginal_mode, self.traj_step, self.out_channels)
            outs = torch.cat([out_traj, loc], dim=-2)
            outs[... ,:2] = torch.cumsum(outs[..., :2], dim=-1)
        else:
            outs = traj
            comb_feat = x

        pi = self.pi(comb_feat).squeeze(-1)

        return feat_traj, outs, pi

def test_conti():
    
    dec = NewContigencyDecoder(128, 6, 12, 80, 4,
        top_trajs=6, traj_input=4, traj_step=5)
    
    x_feat = torch.randn((4, 128))
    traj = torch.randn((4, 15, 40, 3))
    score = torch.randn((4, 15))

    outs, pi, prob = dec(x_feat)
    print(outs.shape, pi.shape, prob.shape)

if __name__ == '__main__':
    test_conti()