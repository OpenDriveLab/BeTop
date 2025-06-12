import torch
# torch.autograd.set_detect_anomaly(True)

from src.models.planTF.diff_plan_utils.lqr import BatchLQRTracker
from src.models.planTF.diff_plan_utils.kinematic_model import DiffKinematicBicycleModel

motion_model = DiffKinematicBicycleModel()
tracker = BatchLQRTracker()

import torch.nn as nn
from torch.optim import Adam

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
    def forward(self, x):
        return self.layer(x)

model = MLP()
optimizer = Adam(model.parameters(),lr=1e-4)


B = 64
from time import time
import torch.nn.functional as F

inputs = torch.randn((B, 41, 3))
gt =  torch.randn((B, 41, 3))

for _ in range(100):
    s = time()
    optimizer.zero_grad()
    traj = model(inputs)
    tracker.update(traj[..., :3])
    simulated_states = torch.zeros((traj.shape[0], traj.shape[1], 11)).to(traj.device)
    
    for time_idx in range(1, 5 + 1):
        command_states = tracker.track_trajectory(
                    time_idx -1,
                    time_idx,
                    simulated_states[:, time_idx - 1],
                )

        simulated_states[:, time_idx] = motion_model.propagate_state(
            states=simulated_states[:, time_idx - 1],
            command_states=command_states,
            sampling_time=0.1,
        )
        
    pred = torch.cat([simulated_states[:, :5, :3], traj[:, 5:, :3]], dim=1)

    loss = F.smooth_l1_loss(pred, gt)
    print(loss, time()-s)
    loss.backward()
    optimizer.step()
