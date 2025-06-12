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
import numpy as np 

class MLP(nn.Module):
    def __init__(self, in_dim, dims, out_dim, activation=F.relu):
        super(MLP, self).__init__()
        self.in_layer = nn.Sequential(
            nn.Linear(in_dim, dims[0]),
            nn.ReLU(inplace=True)
        )
        self.layers = nn.ModuleList([
                nn.Sequential(
                nn.Linear(dims[i], dims[i]),
                nn.ReLU(inplace=True)
                ) for i in range(len(dims))
            ])
        self.out_layer = nn.Linear(dims[-1], out_dim)
        self.activation = activation
    
    def forward(self, x):
        x = self.in_layer(x)
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


class OccupancyFuser(nn.Module):
    def __init__(self, dim, drop=0.1):
        super(OccupancyFuser, self).__init__()
        self.src_mlp = MLP(dim, [dim], dim//2)
        self.tgt_mlp = MLP(dim, [dim], dim//2)

        self.fuser = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(dim, dim),
            nn.Dropout(drop),
        )
        self.ln = nn.LayerNorm(dim)
    
    def forward(self, src_feat, tgt_feat, prev_occ_feat):
        """
        src_feat, tgt_feat :[b, len_src, d], [b, len_tgt, d]
        prev_occ_feat: [b, len_src, len_tgt, d]
        return occ_feat[b, A, A, d]
        """

        src_feat = self.src_mlp(src_feat)
        tgt_feat = self.tgt_mlp(tgt_feat)
        # broadcast the source and target feature:
        len_src, len_tgt = src_feat.shape[1], tgt_feat.shape[1]
        # [b, len_src, len_tgt, d//2]
        src = src_feat.unsqueeze(2).repeat(1, 1, len_tgt, 1)
        tgt = tgt_feat.unsqueeze(1).repeat(1, len_src, 1, 1)
        agt_inter_feat = torch.cat([src, tgt, prev_occ_feat], dim=-1)
        agt_inter_feat = self.fuser(agt_inter_feat)
        occ_feat = self.ln(agt_inter_feat + prev_occ_feat)
        return occ_feat

class OccupancyDecoder(nn.Module):
    def __init__(self, dim, drop=0.1):
        super(OccupancyDecoder, self).__init__()
        self.decoder = MLP(dim, [dim], 1)
    
    def forward(self, occ_feat, A=33):
        # [b, a+m, a+m, 1]
        out = self.decoder(occ_feat)
        actor_o = out[:, :A, :A, :]
        actor_map_o = out[:, :A, A:, :]
        # map_actor_o = out[:, A:, :A, :]
        return actor_o, actor_map_o#, map_actor_o
        

def test_agt_layer():
    agt_input = torch.randn((4, 33, 256))
    tgt_input = torch.randn((4, 55, 256))
    agt_adj = torch.randn((4, 33, 55))

    # layer = BehaviorMLP(256, 256)
    layer = CrossGNN(256, 256)
    out = layer(tgt_input, agt_adj)
    print(out.shape)

if __name__ == "__main__":
    test_agt_layer()
