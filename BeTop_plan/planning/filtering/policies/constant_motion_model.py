import numpy as np 

class ConstMotionModel:
    def __init__(self, horizon=80, dt=0.1, use_acc=False):
        self.horizon = horizon
        self.dt = dt
        self.use_acc = use_acc
    
    def cal_full_traj(self, cur_state, motion_traj):
        full_traj = np.concatenate([cur_state[:2][None], motion_traj], axis=0)
        diff = np.diff(full_traj, axis=0)
        angle = np.arctan2(diff[:, 1], diff[:, 0])
        ret_traj = np.concatenate([motion_traj, angle[:, None]], axis=1)
        return ret_traj
    
    def constant_a_model(self, cur_state):
        pos_xy = np.repeat(cur_state[:2][None], self.horizon, axis=0)
        speed = np.repeat(cur_state[[3, 4]][None], self.horizon, axis=0)
        acc = np.repeat(cur_state[[5, 6]][None], self.horizon, axis=0)

        cum_acc = np.cumsum(acc, axis=0)
        motion_speed = np.cumsum(speed + self.dt * cum_acc, axis=0)
        motion_traj = pos_xy + self.dt * motion_speed + 0.5 * (self.dt ** 2) * cum_acc

        return self.cal_full_traj(cur_state, motion_traj)
    
    def constant_v_model(self, cur_state):
        pos_xy = np.repeat(cur_state[:2][None], self.horizon, axis=0)
        speed = np.sqrt(cur_state[3]**2 + cur_state[4]**2)
        new_speed = np.array([speed * np.cos(cur_state[2]), speed * np.sin(cur_state[2])])

        new_speed = np.repeat(new_speed[None], self.horizon, axis=0)

        cum_speed = np.cumsum(new_speed, axis=0)
        motion_traj = pos_xy + self.dt * cum_speed 

        length = pos_xy.shape[0]
        angle = np.array([cur_state[2]] * length)

        return np.concatenate([motion_traj, angle[:, None]], axis=1)
    
    def compute_trajectory(self, cur_state):
        if self.use_acc:
            return self.constant_a_model(cur_state)
        return self.constant_v_model(cur_state)
