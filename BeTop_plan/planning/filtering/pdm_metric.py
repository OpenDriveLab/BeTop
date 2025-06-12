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

from src.planners.pdm_planner.abstract_pdm_planner import (
    AbstractPDMPlanner,
)
from src.planners.pdm_planner.observation.pdm_observation import (
    PDMObservation,
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

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from typing import List, Optional, Tuple, Type

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import (
    StateSE2,
    TimeDuration,
    TimePoint,
    Point2D
)
from nuplan.planning.metrics.utils.state_extractors import (
    extract_ego_acceleration,
    extract_ego_yaw_rate,
)
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.scenario_utils import (
    sample_indices_with_time_horizon,
)
from nuplan.planning.simulation.history.simulation_history_buffer import (
    SimulationHistoryBuffer,
)
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
    SimulationIteration,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
    AbstractModelFeature,
)
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
    build_ego_features,
)
from shapely.geometry import Point

from src.planners.pdm_planner.pdm_closed_planner import (
    PDMClosedPlanner,
)
from src.planners.pdm_planner.utils.pdm_array_representation import (
    ego_states_to_state_array,
)
from src.planners.pdm_planner.utils.pdm_enums import (
    StateIndex,
)
from src.planners.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)
from src.planners.pdm_planner.utils.pdm_array_representation import (
    ego_state_to_state_array,
)
from src.planners.pdm_planner.utils.pdm_path import PDMPath

from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.abstract_map import AbstractMap, PolygonMapObject
from nuplan.common.maps.maps_datatypes import (
    SemanticMapLayer,
    TrafficLightStatusData,
    TrafficLightStatusType,
)

from src.planners.pdm_planner.utils.pdm_enums import (
    MultiMetricIndex,
    WeightedMetricIndex,
)


import shapely

class PDMMetric(object):
    """
    Long-term IL-based trajectory planner, with short-term RL-based trajectory tracker.
    """

    requires_scenario: bool = False

    def __init__(
        self,
        history_horizon: float = 2,
        future_horizon: float = 8,
        sample_interval: float = 0.1,
    ) -> None:  
        """
        Initializes the ML planner class.
        :param model: Model to use for inference.
        """

        self._future_horizon = 8.0
        self._step_interval = 0.1

        #for PDM score:
        trajectory_sampling  = TrajectorySampling(num_poses=80, interval_length=0.1)
        proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
     
        self._simulator = PDMSimulator(proposal_sampling)
        self._scorer = PDMScorer(proposal_sampling)
        self.proposal_sampling = proposal_sampling
        self.trajectory_sampling = trajectory_sampling
    
    def get_trajectory_as_array(
        self,
        trajectory: InterpolatedTrajectory,
        future_sampling: TrajectorySampling,
        start_time: TimePoint,
    ) -> npt.NDArray[np.float64]:
        """
        Interpolated trajectory and return as numpy array
        :param trajectory: nuPlan's InterpolatedTrajectory object
        :param future_sampling: Sampling parameters for interpolation
        :param start_time: TimePoint object of start
        :return: Array of interpolated trajectory states.
        """

        times_s = np.arange(
            0.0,
            future_sampling.time_horizon + future_sampling.interval_length,
            future_sampling.interval_length,
        )

        times_s += start_time.time_s
        times_us = [int(time_s * 1e6) for time_s in times_s]
        # print(len(times_us),trajectory.start_time.time_us/1e6, trajectory.end_time.time_us/1e6, start_time.time_s)
        times_us = np.clip(times_us, trajectory.start_time.time_us, trajectory.end_time.time_us)
        time_points = [TimePoint(time_us) for time_us in times_us]

        trajectory_ego_states: List[EgoState] = trajectory.get_state_at_times(time_points)
        # print(len(trajectory_ego_states))

        return ego_states_to_state_array(trajectory_ego_states)
    
    def metric_details(self, multi_metric, weight_metric):

        return dict(
            no_collision=multi_metric[MultiMetricIndex.NO_COLLISION],
            drivable_area=multi_metric[MultiMetricIndex.DRIVABLE_AREA],
            driving_direction=multi_metric[MultiMetricIndex.DRIVING_DIRECTION],
            progress=weight_metric[WeightedMetricIndex.PROGRESS],
            ttc=weight_metric[WeightedMetricIndex.TTC],
            comfortable=weight_metric[WeightedMetricIndex.COMFORTABLE]
        )
    
    def pdm_scoring(
        self,
        metric_data,
        plan_traj,
        ):

        ego_state = metric_data['ego_state']
        curr = np.zeros((1, 3))
        curr[0, :2] = ego_state.rear_axle.array
        curr[0, 2] = ego_state.rear_axle.heading
        # 1. Environment forecast and observation update
        if isinstance(plan_traj, InterpolatedTrajectory):
            plan_states = self.get_trajectory_as_array(plan_traj, self.trajectory_sampling, ego_state.time_point)
        else:
            plan_states = np.concatenate([curr, plan_traj[:40, :3]], axis=0, dtype=np.float64)
        ref_states = self.get_trajectory_as_array(metric_data['trajectory'], self.proposal_sampling, ego_state.time_point)
        # print(plan_states.shape)
        proposals_array = np.concatenate([
            ref_states[None, ..., :3], plan_states[None, :41, :3]
            ], axis=0)

        simulated_proposals_array = self._simulator.simulate_proposals(
            proposals_array, ego_state
        )

        # 5. Score proposals
        proposal_scores = self._scorer.score_proposals(
            simulated_proposals_array,
            ego_state,
            metric_data['observation'],
            metric_data['centerline'],
            metric_data['route_lane_ids'],
            metric_data['drivable_area_map'],
            None,
        )

        details = self.metric_details(self._scorer.ret_multi_metric, self._scorer.ret_weighted_metric)

        #0 for referenced traj, 1 for plan traj 
        return proposal_scores[0], proposal_scores[1], details