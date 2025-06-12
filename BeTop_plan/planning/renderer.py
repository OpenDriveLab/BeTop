'''
Behavioral Topology (BeTop): https://arxiv.org/abs/2409.18031
'''

import logging
import hydra
import torch
import numpy as np

from nuplan.planning.training.preprocessing.feature_collate import FeatureCollate
from nuplan.planning.training.preprocessing.feature_preprocessor import FeaturePreprocessor

from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.builders.scenario_builder import build_scenarios

from nuplan.planning.training.data_loader.scenario_dataset import ScenarioDataset
from torch.utils.data import DataLoader
from nuplan.planning.script.utils import set_default_path

from nuplan.planning.script.builders.scenario_builder import extract_scenarios_from_dataset
from nuplan.planning.script.builders.planner_builder import build_planners

from filtering.scenario_processor import MetricProcessor

logging.getLogger("numba").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

set_default_path()

def wrap_to_pi(theta):
    return (theta+np.pi) % (2*np.pi) - np.pi

def build_dataloader(cfg, worker, model):

    feature_builders = model.get_list_of_required_feature()
    # print(feature_builders)
    target_builders = model.get_list_of_computed_target()

    feature_preprocessor = FeaturePreprocessor(
        cache_path=cfg.cache.cache_path,
        force_feature_computation=cfg.cache.force_feature_computation,
        feature_builders=feature_builders,
        target_builders=target_builders,
    )

    scenarios = build_scenarios(cfg, worker, model)

    dataset = ScenarioDataset(
            scenarios=scenarios,
            feature_preprocessor=feature_preprocessor,
        )
    # dataset = dataset.to(torch.device('cuda'))
    
    return DataLoader(
            dataset=dataset, shuffle=False, 
            batch_size=4, num_workers=4, 
            collate_fn=FeatureCollate()
            )

import matplotlib.pyplot as plt
import matplotlib as mpl


def process_cuda_tensor(data):
    for k in data.keys():
        if isinstance(data[k], torch.Tensor):
            data[k] = data[k].cuda()
            continue
        for sk in data[k].keys():
            if isinstance(data[k][sk], torch.Tensor):
                data[k][sk] = data[k][sk].cuda()
    return data

def calibration(inputs, beta=2., alpha=5.):
    less = (inputs < 0).float()
    return less * beta * inputs + (1- less) * inputs * alpha

def soft_iou(inputs, preds):
    b, s, t = inputs.shape 
    inputs, preds = inputs.reshape(b, s*t), preds.reshape(b, s*t)
    inter = inputs * preds 
    uni =  inputs.sum(-1) + preds.sum(-1) - inter
    uni = uni + (uni==0).float()
    return (inter / uni).mean()

def post_process(out, data):
    max_score = out['probability']
    b = max_score.shape[0]
    #filter top k traj
    proposals_array = out['full_trajectory']
    full_array = proposals_array.clone()
    plan_probs, score_idx = torch.topk(max_score, k=10, dim=-1)
    top_score = max_score[torch.arange(b)[:, None], score_idx].softmax(-1)
    proposals_array = proposals_array[torch.arange(b)[:, None], score_idx, :]
    # print(out['prediction'].shape)
    pred = out['prediction'][:, :, :, :, :2]
    curr = data["agent"]["position"][:, 1:, 20, :2]
    pred = pred + curr[:, :, None, None, :]
    pred_score = torch.argmax(out['pred_probability'],dim=-1)
    n = pred_score.shape[-1]
    pred = pred[torch.arange(b)[:, None, None], torch.arange(n)[None, :, None],pred_score[:, :, None]][:, :, 0]
  
    return proposals_array.detach().cpu().numpy(), pred.detach().cpu().numpy(), top_score.detach().cpu().numpy(), full_array.detach().cpu().numpy(), score_idx.detach().cpu().numpy()


def plot_scneario(scenario, processor, planner):

    planner_input, planner_initialization = processor._get_planner_inputs(scenario)
    planner.initialize(planner_initialization)
    _ = planner.compute_planner_trajectory(planner_input)

    data = planner.input_data
    out = planner.plan_output
    valid_mask = data["agent"]["valid_mask"][:, :, -80 :]
    
    plan, pred, score, full_plan, score_idx = post_process(out, data)

    for b in range(1):
        # [a, m]
        agent_pos = data["agent"]["position"][b, :, :].cpu().numpy()
        agent_heading = wrap_to_pi(data["agent"]["heading"][b, :, :].cpu().numpy())
        agent_shape = data["agent"]["shape"][b, :, :].cpu().numpy()
        agent_mask = data["agent"]["valid_mask"][b, :, :].cpu().numpy()

        agent_valid = valid_mask.float().sum(-1) > 0
        comb_valid = agent_valid[:, None, :] * agent_valid[:, :, None]

        polygon_center = data["map"]["polygon_center"][b].cpu().numpy()  
        polygon_mask = data["map"]["valid_mask"][b].cpu().numpy()
        polygon_points = data['map']['point_position'][b].cpu().numpy()

        fig = plt.figure()
        dpi = 400

        fig.set_dpi(dpi)
        fig.set_tight_layout(True)
        
        ego_actor_occ = calibration(out['actor_occ'][b, 0, :, 0]).sigmoid()
        _, ego_top_idx = ego_actor_occ.topk(min(4,ego_actor_occ.shape[0]), dim=-1)
        ego_actor_occ = ego_actor_occ.cpu().detach().numpy()
        #plot agents:

        for i in range(agent_pos.shape[0]):
            color = 'firebrick' if i==0 else 'mediumseagreen'
            z = np.linspace(8, 0, 80)
            # print(pred.shape)
            if agent_mask[i, 20] != 0 :#and (braids[i, 0] !=0 or i==0):
                if i!=0:
                    box_alpha = np.clip(ego_actor_occ[i] * 5, 0, 0.5) + 0.5
                else:
                    box_alpha = 1.
                plt.plot(agent_pos[i, agent_mask[i], 0], agent_pos[i, agent_mask[i], 1], color=color, alpha=box_alpha)
                if i!=0:
                    if i in ego_top_idx[b]:
                        plt.scatter(pred[b, i-1, :, 0], pred[b, i-1, :, 1],  alpha=0.5,zorder=10, c=z, s=1.5, cmap='winter')
                        color = 'navy'
                    else:
                        plt.plot(pred[b, i-1, :, 0], pred[b, i-1, :, 1], alpha=0.3, color='darkblue', linewidth=1, zorder=8)
                else:
                    idx = 0
                    for m in range(full_plan.shape[1]):
                        if m in (score_idx[b]):
                            plt.scatter(full_plan[b, m, :, 0], full_plan[b, m, :, 1], alpha=score[b, idx], c=z, s=2, cmap='autumn', zorder=11)
                            idx += 1
                        else:
                            plt.plot(full_plan[b, m, :, 0], full_plan[b, m, :, 1], alpha=0.5, color='firebrick', linewidth=1, zorder=9)
                    
                x_center, y_center, heading = agent_pos[i, 20, 0], agent_pos[i, 20, 1], agent_heading[i, 20]
                agent_length, agent_width = agent_shape[i, 20, 0],  agent_shape[i, 20, 1]
                agent_bottom_right = (x_center - agent_length/2, y_center - agent_width/2)
            
                rect = plt.Rectangle(agent_bottom_right, agent_length, agent_width, linewidth=2, color=color, alpha=box_alpha, zorder=10 if color=='navy' else 15,
                                    transform=mpl.transforms.Affine2D().rotate_around(*(x_center, y_center), heading - np.pi/2) + plt.gca().transData)
                plt.gca().add_patch(rect)
        
        for i in range(polygon_center.shape[0]):
            if np.sum(polygon_mask[i]) !=0:
                for j in range(polygon_points.shape[1]):
                    if j==0:
                        plt.plot(polygon_points[i, j, polygon_mask[i], 0], polygon_points[i, j, polygon_mask[i], 1], 
                        color='whitesmoke', alpha=0.6, linewidth=0.8, zorder=2, linestyle='dotted')
                    plt.plot(polygon_points[i, j, polygon_mask[i], 0], polygon_points[i, j, polygon_mask[i], 1], 
                    color='gray', alpha=1.0, linewidth=5,  zorder=1)
                
        
        plt.gca().set_facecolor('k')
        plt.gca().margins(0)  
        plt.gca().set_aspect('equal')
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.tight_layout()
        plt.axis('off')
        scenario_id = scenario.token
        
        plt.savefig(f'/path/to/fig/save/plan_{scenario_id}.png', 
         bbox_inches='tight', pad_inches=0, edgecolor='none', facecolor='k')
        plt.close()

        
@hydra.main(config_path="./config", config_name="default_simulation")
def main(cfg):

    build_logger(cfg)

    worker = build_worker(cfg)

    scenarios = extract_scenarios_from_dataset(cfg, worker)
    print(len(scenarios))

    planner = build_planners(cfg.planner, scenarios[0])[0]

    processor = MetricProcessor(with_history=20)
    from tqdm import tqdm

    for scenario in tqdm(scenarios):
        plot_scneario(scenario, processor, planner)

if __name__ == "__main__":
    main()