'''
Behavioral Topology (BeTop): https://arxiv.org/abs/2409.18031
'''

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 

class MLP(nn.Module):
    def __init__(self, in_dim, dims, out_dim, activation=F.relu):
        super(MLP, self).__init__()
        self.in_layer = nn.Sequential(
            nn.Linear(in_dim, dims[0]),
            nn.ReLU(inplace=True)
        )
        if len(dims) > 0:
            self.layers = nn.ModuleList([
                    nn.Sequential(
                    nn.Linear(dims[i], dims[i]),
                    nn.ReLU(inplace=True)
                    ) for i in range(len(dims))
                ])
        else:
            self.layers = None
        self.out_layer = nn.Linear(dims[-1], out_dim)
        self.activation = activation
    
    def forward(self, x):
        x = self.in_layer(x)
        if self.layers is not None:
            for layer in self.layers:
                x = layer(x)
        x = self.out_layer(x)
        return x


class SelfGNN(nn.Module):
    def __init__(self, in_features, out_features, edge_weight=0.5):
        super(SelfGNN, self).__init__()
        self.edge_weight = edge_weight

        if self.edge_weight != 0:
            self.weight_forward = torch.Tensor(in_features, out_features)
            self.weight_forward = nn.Parameter(nn.init.xavier_uniform_(self.weight_forward))
            self.weight_backward = torch.Tensor(in_features, out_features)
            self.weight_backward = nn.Parameter(nn.init.xavier_uniform_(self.weight_backward))

        self.weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        self.edge_weight = edge_weight

    def forward(self, x, adj):

        support_loop = torch.matmul(x, self.weight)
        output = support_loop

        if self.edge_weight != 0:
            support_forward = torch.matmul(x, self.weight_forward)
            output_forward = torch.matmul(adj, support_forward)
            output += self.edge_weight * output_forward

            support_backward = torch.matmul(x, self.weight_backward)
            output_backward = torch.matmul(adj.permute(0, 2, 1), support_backward)
            output += self.edge_weight * output_backward

        return output

class CrossGNN(nn.Module):
    def __init__(self, in_features, out_features, edge_weight=0.5):
        super(CrossGNN, self).__init__()
        self.edge_weight = edge_weight
        self.weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
    
    def forward(self, tgt, adj):
        support = torch.matmul(tgt, self.weight)
        adj = adj * self.edge_weight
        output = torch.matmul(adj, support)
        return output

class CrossOccLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, edge_weight=0.5):
        super(CrossOccLayer, self).__init__()

        self.cross_gnn = CrossGNN(in_features, in_features, edge_weight)
        self.ffn = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.ln = nn.LayerNorm(out_features)
    
    def forward(self, src, tgt, edge):
        fitlered_tgt = self.cross_gnn(tgt, edge)
        occ_feat = self.ffn(fitlered_tgt)
        return self.ln(src + occ_feat)


class JFPScorer(nn.Module):
    def __init__(self, input_dim, dim, occ_step=1):
        super(JFPScorer, self).__init__()
        self.src_mlp = MLP(input_dim*2, [dim], dim//2)
        self.tgt_mlp = MLP(input_dim*2, [dim], dim//2)
        self.fuse_mlp = MLP(dim, [dim//2], occ_step)
    
    def left_hand_rotate(self, traj, curr):
        cos, sin = torch.cos(curr[:, -1])[:, None, None], torch.sin(curr[:, -1])[:, None, None]
        x, y = traj[:, :, :, 0], traj[:, :, :, 1]
        new_x = cos * x - sin * y + curr[:, 0, None, None]
        new_y = sin * x + cos * y + curr[:, 1, None, None]
        new_traj = traj.clone()
        new_traj[..., 0] = new_x
        new_traj[..., 1] = new_y 
        return new_traj


    def forward(self, pred_traj, curr):
        '''
        pred_traj: [B(b*2), m, T, 7]
        src_curr: [B(b*2), 3]
        '''
        b, m, t, d = pred_traj.shape
        pred_traj = pred_traj.reshape(b//2, 2, m, t, d)
        curr = curr.reshape(b//2, 2, 3)
        src_traj, tgt_traj = pred_traj[:, 0], pred_traj[:, 1]
        src_curr, tgt_curr = curr[:, 0], curr[:, 1]


        src_tgt_traj = self.left_hand_rotate(tgt_traj, src_curr)
        tgt_src_traj = self.left_hand_rotate(src_traj, tgt_curr)

        src_inputs = torch.cat([src_traj, src_tgt_traj], dim=-1)
        tgt_inputs = torch.cat([tgt_traj, tgt_src_traj], dim=-1)

        src_feat = torch.max(self.src_mlp(src_inputs), dim=-2)[0]
        tgt_feat = torch.max(self.tgt_mlp(tgt_inputs), dim=-2)[0]

        src_feat = src_feat[:, :, None, :].repeat(1, 1, m, 1)
        tgt_feat = tgt_feat[:, None, :, :].repeat(1, m, 1, 1)
        feat = torch.cat([src_feat, tgt_feat], dim=-1)
        jfp_graph = self.fuse_mlp(feat)[..., 0]
        return jfp_graph


class TopoFuser(nn.Module):
    def __init__(self, input_dim, dim, drop=0.1):
        super(TopoFuser, self).__init__()
        self.src_mlp = MLP(input_dim, [dim], dim//2)
        self.tgt_mlp = MLP(input_dim, [dim], dim//2)
    
    def forward(self, src_feat, tgt_feat, prev_occ_feat=None):
        """
        src_feat, tgt_feat :[b, len_src, d], [b, len_tgt, d]
        prev_occ_feat: [b, len_src, len_tgt, d]
        return occ_feat[b, len_src, len_tgt, d]
        """

        src_feat = self.src_mlp(src_feat)
        tgt_feat = self.tgt_mlp(tgt_feat)
        # broadcast the source and target feature:
        len_src, len_tgt = src_feat.shape[1], tgt_feat.shape[1]
        # [b, len_src, len_tgt, d//2]
        
        src = src_feat.unsqueeze(2).repeat(1, 1, len_tgt, 1)
        tgt = tgt_feat.unsqueeze(1).repeat(1, len_src, 1, 1)
        agt_inter_feat = torch.cat([src, tgt], dim=-1)

        if prev_occ_feat is not None:
            agt_inter_feat = agt_inter_feat + prev_occ_feat

        return agt_inter_feat

class TopoDecoder(nn.Module):
    def __init__(self, dim, drop=0.1, multi_step=1):
        super(TopoDecoder, self).__init__()
        self.decoder = MLP(dim, [dim], multi_step)
    
    def forward(self, occ_feat):
        # [b, a+m, a+m, 1]
        out = self.decoder(occ_feat)
        # return out[..., 0]
        return out
        
