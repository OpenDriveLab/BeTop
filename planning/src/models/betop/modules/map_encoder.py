'''
Behavioral Topology (BeTop): https://arxiv.org/abs/2409.18031
'''
'''
Pipeline developed upon planTF: 
https://arxiv.org/pdf/2309.10443
'''
import torch
import torch.nn as nn

from ..layers.embedding import PointsEncoder


class MapEncoder(nn.Module):
    def __init__(
        self,
        polygon_channel=6,
        dim=128,
        perspect_norm=False,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.perspect_norm = perspect_norm
        self.polygon_encoder = PointsEncoder(polygon_channel, dim)
        self.speed_limit_emb = nn.Sequential(
            nn.Linear(1, dim), nn.ReLU(), nn.Linear(dim, dim)
        )

        self.type_emb = nn.Embedding(3, dim)
        self.on_route_emb = nn.Embedding(2, dim)
        self.traffic_light_emb = nn.Embedding(4, dim)
        self.unknown_speed_emb = nn.Embedding(1, dim)
    
    def rotation(self, xy, heading):
        cos, sin = torch.cos(heading[:, :, None]), torch.sin(heading[:, :, None])
        x, y = xy[..., 0], xy[..., 1]
        new_x = cos *x + sin * y 
        new_y = -sin *x + cos * y
        return torch.stack([new_x, new_y], dim=-1)
    
    def plot_map(self, position):
        import matplotlib.pyplot as plt 
        for b in range(position.shape[0]):
            pos = position[b].cpu().numpy()
            a = pos.shape[0]
            plt.figure()
            for i in range(a):
                mask = pos[i].sum(-1) != 0
                plt.plot(pos[i, mask, 0], pos[i, mask, 1], color='k', alpha=0.8)
            plt.gca().set_aspect('equal')
            plt.tight_layout()
            plt.savefig(f'/home/liuhaochen/behavioro/test_rot/map_{b}.png') 

    def forward(self, data) -> torch.Tensor:
        polygon_center = data["map"]["polygon_center"]
        polygon_type = data["map"]["polygon_type"].long()
        polygon_on_route = data["map"]["polygon_on_route"].long()
        polygon_tl_status = data["map"]["polygon_tl_status"].long()
        polygon_has_speed_limit = data["map"]["polygon_has_speed_limit"]
        polygon_speed_limit = data["map"]["polygon_speed_limit"]
        point_position = data["map"]["point_position"]
        point_vector = data["map"]["point_vector"]
        
        point_orientation = data["map"]["point_orientation"]
        polygon_orientation = data['map']['polygon_orientation']
        valid_mask = data["map"]["valid_mask"]

        if self.perspect_norm:
            # norm to perspective axis:
            fvalid_mask = valid_mask.float().clone()
            curr_pos = polygon_center[:, :, :2]
            map_position = self.rotation(point_position[:, :, 0] - curr_pos[..., None, :], polygon_orientation.clone())
            map_position = map_position * fvalid_mask[..., None]
            # self.plot_map(map_position)
            
            point_orientation = point_orientation[:, :, 0, :] - polygon_orientation.clone()[..., None]
            point_orientation = point_orientation * fvalid_mask
            vector = self.rotation(point_vector[:, :, 0], polygon_orientation.clone())
            vector = vector * fvalid_mask[..., None]

            polygon_feature = torch.cat([
                map_position, vector, point_orientation.cos()[..., None], point_orientation.sin()[..., None]
            ], dim=-1)
         
        else:
            polygon_feature = torch.cat(
                [
                    point_position[:, :, 0] - polygon_center[..., None, :2],
                    point_vector[:, :, 0],
                    torch.stack(
                        [
                            point_orientation[:, :, 0].cos(),
                            point_orientation[:, :, 0].sin(),
                        ],
                        dim=-1,
                    ),
                ],
                dim=-1,
            )

        bs, M, P, C = polygon_feature.shape
        valid_mask = valid_mask.view(bs * M, P)
        polygon_feature = polygon_feature.reshape(bs * M, P, C)

        x_polygon = self.polygon_encoder(polygon_feature, valid_mask).view(bs, M, -1)

        x_type = self.type_emb(polygon_type)
        x_on_route = self.on_route_emb(polygon_on_route)
        x_tl_status = self.traffic_light_emb(polygon_tl_status)
        x_speed_limit = torch.zeros(bs, M, self.dim, device=x_polygon.device)
        x_speed_limit[polygon_has_speed_limit] = self.speed_limit_emb(
            polygon_speed_limit[polygon_has_speed_limit].unsqueeze(-1)
        )
        x_speed_limit[~polygon_has_speed_limit] = self.unknown_speed_emb.weight

        x_polygon += x_type + x_on_route + x_tl_status + x_speed_limit

        return x_polygon
