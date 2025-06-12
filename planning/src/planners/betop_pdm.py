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
    AbstractPlanner,
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

from src.planners.pdm_planner.proposal.pdm_generator import (
    PDMGenerator,
)
from src.planners.pdm_planner.proposal.pdm_proposal import (
    PDMProposalManager,
)

from src.planners.pdm_planner.utils.pdm_path import PDMPath

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
import gc

class PDMImitationPlanner(AbstractPDMPlanner):
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
        idm_policies=None,
    ) -> None:
        super(PDMImitationPlanner, self).__init__(map_radius)  
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

        self._generator = PDMGenerator(trajectory_sampling, proposal_sampling)
        self._observation = PDMObservationPred(trajectory_sampling, proposal_sampling, map_radius)
        self._simulator = PDMSimulator(proposal_sampling)
        self._scorer = PDMScorer(proposal_sampling)
        self._emergency_brake = PDMEmergencyBrake(trajectory_sampling)

        self._idm_policies: BatchIDMPolicy = idm_policies
        self._lateral_offsets: Optional[List[float]] = [-1.0, 1.0]

        self._proposal_manager: Optional[PDMProposalManager] = None

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
            proposal_paths: List[PDMPath] = self._get_proposal_paths(current_lane)

            self._proposal_manager = PDMProposalManager(
                lateral_proposals=proposal_paths,
                longitudinal_policies=self._idm_policies,
            )

        # update proposals
        self._proposal_manager.update(current_lane.speed_limit_mps)
    
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
        # 1. save centerline path (necessary for progress metric)
        output_paths: List[PDMPath] = [self._centerline]

        # 2. add additional paths with lateral offset of centerline
        if self._lateral_offsets is not None:
            for lateral_offset in self._lateral_offsets:
                offset_discrete_path = parallel_discrete_path(
                    discrete_path=centerline_discrete_path, offset=lateral_offset
                )
                output_paths.append(PDMPath(offset_discrete_path))

        return output_paths
    
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
        token = planner_feature_torch.data['agent_token'][0]
        # print(token.shape)
        self._feature_building_runtimes.append(time.perf_counter() - self._start_time)
        planner_out = self._planner.forward(planner_feature_torch.data)

        data = planner_feature_torch.data
        agent_pos = data["agent"]["position"][:, 1:, 20]

        angle = torch.atan2(planner_out['prediction'][..., 3], planner_out['prediction'][..., 2])
        agent_valid = data["agent"]["valid_mask"][:, 1:, : 21].any(-1)
        pred = planner_out['prediction'][..., :2] + agent_pos[:, :, None, None, :2]
        # full_pred = torch.cat([agent_pos[:, :, None, :2], pred], dim=-2)
        curr_angle = data["agent"]["heading"][:, 1:, 20]
        full_angle = angle + curr_angle[:, :, None, None]
        pred = torch.cat([pred, full_angle[..., None]], dim=-1)
        # print(pred.shape)
        curr_pos = torch.stack([agent_pos[..., 0], agent_pos[..., 1], curr_angle], dim=-1)
        b, n, m, t, d = pred.shape 
        curr_pos = curr_pos[:, :, None, :].repeat(1, 1, m, 1)
        # print(curr_pos.shape)
        pred = torch.cat([curr_pos[..., None, :], pred], dim=-2)
        # print(pred.shape)
        max_score = torch.argmax(planner_out['pred_probability'], dim=-1)
        out_pred = pred[0]
        out_pred = out_pred[torch.arange(n), max_score[0]]
        pred = out_pred.cpu().numpy().astype(np.float64)
        # print(pred_input.shape)
        valid = agent_valid[0].cpu().numpy()
        # pred_input = np.array(pred_input, dtype=np.float64)
        # print(pred_input.shape)

        pred = pred[valid]
        token = token[valid]

        # print(pred_input.shape, token.shape)

        pred_input = []
        for i in range(pred.shape[0]):
            temp_array = np.ascontiguousarray(pred[i])
            # print(pred[i][0, :2])
            temp_array = self._get_global_trajectory(temp_array, current_input.history.ego_states[-1])
            pred_input.append(temp_array)

        pred_input = np.array(pred_input, dtype=np.float64)

        ego_state, observation = current_input.history.current_state

        # 1. Environment forecast and observation update
        self._observation.update(
            ego_state,
            observation,
            current_input.traffic_light_data,
            self._route_lane_dict,
            pred_input,
            token
        )

        self._update_proposal_manager(ego_state)

        # 3. Generate/Unroll proposals
        proposals_array = self._generator.generate_proposals(
            ego_state, self._observation, self._proposal_manager
        )

        # if planner_out['conti_plan']:
        #     max_score = planner_out['max_score']
        #     # b = max_score.shape[0]
        #     #filter top k traj
        #     # plan_probs, score_idx = torch.topk(max_score, k=6, dim=-1)
        #     proposals_array = planner_out['max_traj']#proposals_array[torch.arange(b)[:, None], score_idx, :]
        
        # proposals_array = proposals_array[0].cpu().numpy()
        # proposals_array = proposals_array.astype(np.float64)
        # num_modes = proposals_array.shape[0]
        # arr_list = []
        # for i in range(num_modes):
        #     temp_array = proposals_array[i]
        #     arr_list.append(
        #         self._get_global_trajectory(temp_array, current_input.history.ego_states[-1])
        #     )
        # proposals_array = np.stack(arr_list, axis=0)

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

        if trajectory is None:
            trajectory = self._generator.generate_trajectory(np.argmax(proposal_scores))
        #     brake = False
        #     if not planner_out['conti_plan']:
        #         plan_probs = planner_out['max_score'][0].cpu().numpy()
        #     else:
        #         plan_probs = planner_out['max_score'][0].cpu().numpy()
        #     full_score = plan_probs + 0.5 * proposal_scores
        #     # print(proposal_scores, full_score)
        #     trajectory = proposals_array[np.argmax(full_score)]

        return trajectory, brake


    def compute_planner_trajectory(
        self, current_input: PlannerInput
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

        # if self.brake:
        trajectory = self._global_trajectory
        # else:
        #     trajectory = InterpolatedTrajectory(
        #         trajectory=global_trajectory_to_states(
        #             global_trajectory=self._global_trajectory,
        #             ego_history=current_input.history.ego_states,
        #             future_horizon=len(self._global_trajectory) * self._step_interval,
        #             step_interval=self._step_interval,
        #         )
        #     )

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
        # if local_trajectory.shape[-1] == 2:
        #     return global_position

        global_heading = local_trajectory[..., 2] + angle

        global_trajectory = np.concatenate(
            [global_position, global_heading[..., None]], axis=1
        )
        return global_trajectory
