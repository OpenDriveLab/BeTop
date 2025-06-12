'''
Behavioral Topology (BeTop): https://arxiv.org/abs/2409.18031
'''
'''
Pipeline developed upon planTF: 
https://arxiv.org/pdf/2309.10443
'''
from contigency_plan import ContigencyPlanner
import torch 
import numpy as np

def convert_to_tensor(inputs):
    return inputs[None].cuda()

def build_dummy_inputs(t=80):

    vec_x = torch.linspace(0, t, t)
    vec_y = torch.zeros((t, 1)) #-torch.abs(torch.randn(t, 1) * 0.5)
    
    vec_traj = torch.cat([vec_x[:, None], vec_y], dim=-1)

    actor_curr = torch.tensor([[55., 0.]])
    actor_hw = torch.tensor([[2., 2.]])

    actor_xy = torch.cat(
        [
            torch.zeros((t, 1)),
            torch.cat([torch.zeros(75), torch.linspace(0, 5, 5)])[:, None],
            torch.ones((t, 1)) * torch.pi/2
        ], dim=-1
    )

    actor_xy_neg = torch.cat(
        [   
            torch.zeros((t, 1)),
            torch.cat([torch.zeros(75), torch.linspace(0, -5, 5)])[:, None],
            -torch.ones((t, 1)) * torch.pi/2
        ], dim=-1
    )

    actor_full = torch.stack([actor_xy, actor_xy_neg], dim=0)[None]
    prob = torch.tensor([0.6, 0.4])

    data = {
        'shape':convert_to_tensor(actor_hw),
        'position':convert_to_tensor(actor_curr),
        'prediction': convert_to_tensor(actor_full),
        'pred_probability':convert_to_tensor(prob),
        'output_trajectory':convert_to_tensor(vec_traj)
    }
    return data

def main():

    device = torch.device('cuda:0')
    planner = ContigencyPlanner(
        device, feature_len=9, horizon=80, branch_horizon=50,
        modal=2, num_neighbors=1, test=True)
    
    data = build_dummy_inputs()
    contigency_plan = planner.plan_simple(data)
    print(contigency_plan.shape)

    import matplotlib.pyplot as plt

    plt.figure()
    contigency_plan = contigency_plan[0].cpu().numpy()
    prob = data['pred_probability'][0].cpu().numpy()
    actor = data['prediction'][0].cpu().numpy()[0, :, :, :2]
    actor_current = data['position'][0].cpu().numpy()
    actor = actor + actor_current[0, None, None, :]
    for i in range(1):
        plt.plot(contigency_plan[i, :, 0], contigency_plan[i, :, 1], color='darkred', alpha=prob[i])
        plt.scatter(contigency_plan[i, :, 0], contigency_plan[i, :, 1], color='darkred', alpha=prob[i])

        plt.plot(actor[i, :, 0], actor[i, :, 1], color='navy', alpha=prob[i])
        plt.scatter(actor[i, :, 0], actor[i, :, 1], color='navy', alpha=prob[i])
    
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(f'/home/liuhaochen/plan_vanilla.png')
    plt.close()
    

if __name__ == "__main__":
    main()


