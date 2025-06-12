'''
Behavioral Topology (BeTop): https://arxiv.org/abs/2409.18031
'''
'''
Pipeline developed upon planTF: 
https://arxiv.org/pdf/2309.10443
'''
import time
from typing import List, Optional, Type

import numpy as np
import torch
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    Observation,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
    PlannerReport,
)
from nuplan.planning.simulation.planner.planner_report import MLPlannerReport
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper

from src.feature_builders.common.utils import rotate_round_z_axis

from .planner_utils import global_trajectory_to_states, load_checkpoint

from src.planners.pdm_planner.abstract_pdm_planner import (
    AbstractPDMPlanner,
)
from src.planners.pdm_planner.observation.pdm_observation_pred import (
    PDMObservationPred,
)
from src.planners.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from src.planners.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
)
from src.planners.pdm_planner.utils.pdm_emergency_brake import (
    PDMEmergencyBrake,
)
from src.planners.pdm_planner.observation.pdm_observation_utils import (
    get_drivable_area_map,
)
from src.planners.pdm_planner.proposal.batch_idm_policy import (
    BatchIDMPolicy,
)
from src.planners.pdm_planner.utils.pdm_geometry_utils import (
    parallel_discrete_path,
)
from src.planners.pdm_planner.utils.pdm_path import PDMPath

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
import gc
import math
def wrap_to_pi(theta):
    return (theta+math.pi) % (2*math.pi) - math.pi

class BeTopImitationPlanner(AbstractPDMPlanner):
    """
    Long-term IL-based trajectory planner, with short-term RL-based trajectory tracker.
    """

    requires_scenario: bool = False

    def __init__(
        self,
        planner: TorchModuleWrapper,
        planner_ckpt: str = None,
        replan_interval: int = 1,
        use_gpu: bool = True,
        map_radius: float = 50,
    ) -> None:
        super(BeTopImitationPlanner, self).__init__(map_radius)  
        """
        Initializes the ML planner class.
        :param model: Model to use for inference.
        """
        if use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self._planner = planner
        self._planner_feature_builder = planner.get_list_of_required_feature()[0]
        self._planner_ckpt = planner_ckpt
        self._initialization: Optional[PlannerInitialization] = None

        self._future_horizon = 8.0
        self._step_interval = 0.1

        self._replan_interval = replan_interval
        self._last_plan_elapsed_step = replan_interval  # force plan at first step
        self._global_trajectory = None
        self._start_time = None

        # Runtime stats for the MLPlannerReport
        self._feature_building_runtimes: List[float] = []
        self._inference_runtimes: List[float] = []
        self.brake = False

        #for PDM score:
        trajectory_sampling  = TrajectorySampling(num_poses=80, interval_length=0.1)
        proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        self._observation = PDMObservationPred(trajectory_sampling, proposal_sampling, map_radius)
        self._simulator = PDMSimulator(proposal_sampling)
        self._scorer = PDMScorer(proposal_sampling)
        self._emergency_brake = PDMEmergencyBrake(trajectory_sampling)
    
    def model_initialize(self):
        print(self._planner_ckpt)
        torch.set_grad_enabled(False)
        if self._planner_ckpt is not None:
            print('loaded')
            self._planner.load_state_dict(load_checkpoint(self._planner_ckpt))
        self._planner.eval()
        self._planner = self._planner.to(self.device)
    
    def step_initialize(self, initialization: PlannerInitialization):
        self._initialization = initialization

        self._iteration = 0
        self._map_api = initialization.map_api
        self._load_route_dicts(initialization.route_roadblock_ids)
        gc.collect()
        rotate_round_z_axis(np.zeros((1, 2), dtype=np.float64), float(0.0))

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        torch.set_grad_enabled(False)

        if self._planner_ckpt is not None:
            self._planner.load_state_dict(load_checkpoint(self._planner_ckpt))

        self._planner.eval()
        self._planner = self._planner.to(self.device)
        self._initialization = initialization

        self._iteration = 0
        self._map_api = initialization.map_api
        self._load_route_dicts(initialization.route_roadblock_ids)
        gc.collect()

        # just to trigger numba compile, no actually meaning
        rotate_round_z_axis(np.zeros((1, 2), dtype=np.float64), float(0.0))

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def _planning(self, current_input: PlannerInput):
        self._start_time = time.perf_counter()
        planner_feature = self._planner_feature_builder.get_features_from_simulation(
            current_input, self._initialization
        )
        planner_feature_torch = planner_feature.collate(
            [planner_feature.to_feature_tensor().to_device(self.device)]
        )
        self._feature_building_runtimes.append(time.perf_counter() - self._start_time)

        out = self._planner.forward(planner_feature_torch.data)
        local_trajectory = out["output_trajectory"][0].cpu().numpy()

        return local_trajectory.astype(np.float64)
    
    def _update_proposal_manager(self, ego_state: EgoState):
        """
        Updates or initializes PDMProposalManager class
        :param ego_state: state of ego-vehicle
        """

        current_lane = self._get_starting_lane(ego_state)

        # TODO: Find additional conditions to trigger re-planning
        create_new_proposals = self._iteration == 0

        if create_new_proposals:
            self._get_proposal_paths(current_lane)
    
    def _get_proposal_paths(
        self, current_lane
    ):
        """
        Returns a list of path's to follow for the proposals. Inits a centerline.
        :param current_lane: current or starting lane of path-planning
        :return: lists of paths (0-index is centerline)
        """
        centerline_discrete_path = self._get_discrete_centerline(current_lane)
        self._centerline = PDMPath(centerline_discrete_path)
    
    def pdm_plan(
        self,
        current_input: PlannerInput,
        ):

        self._start_time = time.perf_counter()
        planner_feature = self._planner_feature_builder.get_features_from_simulation(
            current_input, self._initialization
        )
        planner_feature_torch = planner_feature.collate(
            [planner_feature.to_feature_tensor().to_device(self.device)]
        )
        self._feature_building_runtimes.append(time.perf_counter() - self._start_time)
        planner_out = self._planner.forward(planner_feature_torch.data)

        self.input_data = planner_feature_torch.data
        self.plan_output = planner_out

        ego_state, observation = current_input.history.current_state

        proposals_array = planner_out['full_trajectory'][0]
        max_score = planner_out['probability']
        b = max_score.shape[0]
        # filter top k traj
        top_k = min(10, max_score.shape[-1])
        plan_probs, score_idx = torch.topk(max_score, k=top_k, dim=-1)
        proposals_array = proposals_array[score_idx[0], :]
        
        proposals_array = proposals_array.cpu().numpy()
        proposals_array = proposals_array.astype(np.float64)
        num_modes = proposals_array.shape[0]
        arr_list = []
        for i in range(num_modes):
            temp_array = proposals_array[i]
            arr_list.append(
                self._get_global_trajectory(temp_array, current_input.history.ego_states[-1])
            )
        proposals_array = np.stack(arr_list, axis=0)

        ### add predictions ###
        data = planner_feature_torch.data
        n_agent = planner_out['prediction'].shape[1] 
        token = planner_feature_torch.data['agent_token'][0][:n_agent]

        token = planner_feature_torch.data['agent_token'][0]
        agent_pos = data["agent"]["position"][:, 1:, 20]
        angle = torch.atan2(planner_out['prediction'][..., 3], planner_out['prediction'][..., 2])
        agent_valid = data["agent"]["valid_mask"][:, 1:, : 21].any(-1)
        pred = planner_out['prediction'][..., :2] + agent_pos[:, :, None, None, :2]

        curr_angle = data["agent"]["heading"][:, 1:, 20]
        full_angle = angle + curr_angle[:, :, None, None]
        full_angle = wrap_to_pi(full_angle)

        out_pred = torch.cat([pred, full_angle[..., None]], dim=-1)[0]

        n, m, t, d = out_pred.shape 
        curr_pos = torch.stack([agent_pos[0, :, 0], agent_pos[0, :, 1], curr_angle[0]], dim=-1)
        curr_pos = curr_pos[:, None, :].repeat(1, m, 1)
        curr_pos = curr_pos[:, :, None, :]
        out_pred = torch.cat([curr_pos, out_pred], dim=-2)
        max_score = torch.argmax(planner_out['pred_probability'], dim=-1)
        out_pred = out_pred[torch.arange(n), max_score[0]]

        pred = out_pred.cpu().numpy()#.astype(np.float64)

        actor_pred = planner_out['actor_occ']
        ego_behave_occ = actor_pred[:, 0, 1:, 0].sigmoid()
        ego_behave_occ = ego_behave_occ[0].cpu().numpy() 

        valid = agent_valid[0].cpu().numpy()
        if np.sum(valid) == 0:
            token=None
            pred_input=None
            ego_behave_occ=None
        else:
            pred = out_pred[valid].cpu().numpy()
            token = token[valid]
            ego_behave_occ = ego_behave_occ[valid]
            pred_input = []
            
            for i in range(pred.shape[0]):
                ptemp_array = np.ascontiguousarray(pred[i])
                ptemp_array = ptemp_array.astype(np.float64)
                ptemp_array = self._get_global_trajectory(ptemp_array, current_input.history.ego_states[-1])
                pred_input.append(ptemp_array)

            pred_input = np.array(pred_input, dtype=np.float64)
            if len(token)==0:
                token=None
                pred_input=None
                ego_behave_occ=None

        self._observation.update(
            ego_state,
            observation,
            current_input.traffic_light_data,
            self._route_lane_dict,
            pred=pred_input,
            token=token,
            behave_occ=None
        )
        self._update_proposal_manager(ego_state)

        simulated_proposals_array = self._simulator.simulate_proposals(
            proposals_array, ego_state
        )

        # 5. Score proposals
        proposal_scores = self._scorer.score_proposals(
            simulated_proposals_array,
            ego_state,
            self._observation,
            self._centerline,
            self._route_lane_dict,
            self._drivable_area_map,
            self._map_api,
        )

        # 6.a Apply brake if emergency is expected
        trajectory = self._emergency_brake.brake_if_emergency(
            ego_state, proposal_scores, self._scorer
        )
        brake = True
        # trajectory = None

        if trajectory is None:
            brake = False
            if not planner_out['conti_plan']:
                plan_probs = planner_out['max_score'].softmax(-1)[0].cpu().numpy()
            else:
                plan_probs = plan_probs[0].softmax(-1).cpu().numpy()

            if self.simulation_metric == 'open_loop_boxes':
                full_score = plan_probs
            else:
                full_score = proposal_scores + 0.3 * plan_probs
            
            # print(proposal_scores, full_score)
            trajectory = proposals_array[np.argmax(full_score)]

        return trajectory, brake


    def compute_planner_trajectory(
        self, current_input: PlannerInput,
    ) -> AbstractTrajectory:
        """
        Infer relative trajectory poses from model and convert to absolute agent states wrapped in a trajectory.
        Inherited, see superclass.
        """
        gc.disable()
        ego_state, _ = current_input.history.current_state
        if self._iteration == 0:
            self._route_roadblock_correction(ego_state)

        # Update/Create drivable area polygon map
        self._drivable_area_map = get_drivable_area_map(
            self._map_api, ego_state, self._map_radius
        )

        if self._last_plan_elapsed_step >= self._replan_interval or (self.brake==True):
            self._global_trajectory, self.brake = self.pdm_plan(current_input)
            self._last_plan_elapsed_step = 0
        else:
            self._global_trajectory = self._global_trajectory[1:]

        if self.brake:
            trajectory = self._global_trajectory
        else:
            trajectory = InterpolatedTrajectory(
                trajectory=global_trajectory_to_states(
                    global_trajectory=self._global_trajectory,
                    ego_history=current_input.history.ego_states,
                    future_horizon=len(self._global_trajectory) * self._step_interval,
                    step_interval=self._step_interval,
                )
            )

        self._inference_runtimes.append(time.perf_counter() - self._start_time)
        self._iteration += 1
        self._last_plan_elapsed_step += 1

        return trajectory

    def generate_planner_report(self, clear_stats: bool = True) -> PlannerReport:
        """Inherited, see superclass."""
        report = MLPlannerReport(
            compute_trajectory_runtimes=self._compute_trajectory_runtimes,
            feature_building_runtimes=self._feature_building_runtimes,
            inference_runtimes=self._inference_runtimes,
        )
        if clear_stats:
            self._compute_trajectory_runtimes: List[float] = []
            self._feature_building_runtimes = []
            self._inference_runtimes = []

        return report

    def _get_global_trajectory(self, local_trajectory: np.ndarray, ego_state: EgoState):
        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading
        arr = np.ascontiguousarray(local_trajectory[..., :2])
        global_position = (
            rotate_round_z_axis(arr, -angle)
            + origin
        )
        global_heading = local_trajectory[..., 2] + angle

        global_trajectory = np.concatenate(
            [global_position, global_heading[..., None]], axis=1
        )

        return global_trajectory
