from __future__ import annotations
import pathlib
from typing import Any, Dict, Optional, Tuple

from nuplan.planning.training.experiments.cache_metadata_entry import CacheMetadataEntry
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
    SimulationIteration,
)
from nuplan.planning.simulation.history.simulation_history_buffer import (
    SimulationHistoryBuffer,
)

from src.planners.pdm_planner.pdm_closed_planner import (
    PDMClosedPlanner,
)
from src.planners.pdm_planner.proposal.batch_idm_policy import (
    BatchIDMPolicy,
)
from src.planners.pdm_planner.observation.pdm_observation_sim import (
    PDMObservation,
)

from .metric_caching_utils import StateInterpolator

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.static_object import StaticObject

import numpy as np

from nuplan.common.actor_state.tracked_objects_types import (
    AGENT_TYPES,
)
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.common.actor_state.tracked_objects import TrackedObjects

import pickle

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

from src.planners.pdm_planner.sim_pdm_closed_planner import (
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

import shapely


class MetricProcessor:

    def __init__(self, 
        save_path=None,
        with_history=1):
        """
        Initialize class.
        :param cache_path: Whether to cache features.
        :param force_feature_computation: If true, even if cache exists, it will be overwritten.
        """
        # self._cache_path = pathlib.Path(cache_path) if cache_path else None
        # self._force_feature_computation = force_feature_computation
        self.save_path = save_path

        # TODO: Add to some config
        self._future_sampling = TrajectorySampling(num_poses=50, interval_length=0.1)
        self._proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        self._map_radius = 100

        self.history_horizon = 2
        self.history_samples = 20

        self.future_horizon = 8
        self.future_samples = 80
        self.with_history = with_history

        self._pdm_closed = PDMClosedPlanner(
            trajectory_sampling=self._future_sampling,
            proposal_sampling=self._proposal_sampling,
            idm_policies=BatchIDMPolicy(
                speed_limit_fraction=[0.2, 0.4, 0.6, 0.8, 1.0],
                fallback_target_velocity=15.0,
                min_gap_to_lead_agent=1.0,
                headway_time=1.5,
                accel_max=1.5,
                decel_max=3.0,
            ),
            lateral_offsets=[-1.0, 1.0],
            map_radius=self._map_radius,
        )

        self.ego_params = get_pacifica_parameters()
        self.length = self.ego_params.length
        self.width = self.ego_params.width

        self.interested_objects_types = [
            TrackedObjectType.EGO,
            TrackedObjectType.VEHICLE,
            TrackedObjectType.PEDESTRIAN,
            TrackedObjectType.BICYCLE,
        ]

    def _get_planner_inputs(
        self, scenario: AbstractScenario
    ) -> Tuple[PlannerInput, PlannerInitialization]:
        """
        Creates planner input arguments from scenario object.
        :param scenario: scenario object of nuPlan
        :return: tuple of planner input and initialization objects
        """

        # Initialize Planner
        planner_initialization = PlannerInitialization(
            route_roadblock_ids=scenario.get_route_roadblock_ids(),
            mission_goal=scenario.get_mission_goal(),
            map_api=scenario.map_api,
        )
        
        ego_states = [scenario.initial_ego_state]
        observations = [scenario.initial_tracked_objects]
        if self.with_history > 1:
            past_ego_trajectory = scenario.get_ego_past_trajectory(
                iteration=0,
                time_horizon=self.history_horizon,
                num_samples=self.history_samples,
            )
            ego_states = list(past_ego_trajectory) + ego_states

            past_tracked_objects = [
                tracked_objects
                for tracked_objects in scenario.get_past_tracked_objects(
                    iteration=0,
                    time_horizon=self.history_horizon,
                    num_samples=self.history_samples,
                )
            ]
            observations = past_tracked_objects + observations

        history = SimulationHistoryBuffer.initialize_from_list(
            buffer_size=self.with_history+1,
            ego_states=ego_states,
            observations=observations,
            sample_interval=0.1,
        )

        planner_input = PlannerInput(
            iteration=SimulationIteration(index=0, time_point=scenario.start_time),
            history=history,
            traffic_light_data=list(scenario.get_traffic_light_status_at_iteration(self.with_history)),
        )

        return planner_input, planner_initialization

    def _interpolate_gt_observation(self, scenario: AbstractScenario) -> PDMObservation:

        # TODO: add to config
        state_size = 6  # (time, x, y, heading, velo_x, velo_y)

        time_horizon = 5.0  # [s]
        resolution_step = 0.5  # [s]
        interpolate_step = 0.1  # [s]

        scenario_step = scenario.database_interval  # [s]

        # sample detection tracks a 2Hz
        relative_time_s = (
            np.arange(0, (time_horizon * 1 / resolution_step) + 1, 1, dtype=float) * resolution_step
        )

        gt_indices = np.arange(
            0, int(time_horizon / scenario_step) + 1, int(resolution_step / scenario_step)
        )
        gt_detection_tracks = [
            scenario.get_tracked_objects_at_iteration(iteration=iteration)
            for iteration in gt_indices
        ]

        detection_tracks_states: Dict[str, Any] = {}
        unique_detection_tracks: Dict[str, Any] = {}

        for time_s, detection_track in zip(relative_time_s, gt_detection_tracks):

            for tracked_object in detection_track.tracked_objects:
                # log detection track
                token = tracked_object.track_token

                # extract states for dynamic and static objects
                tracked_state = np.zeros(state_size, dtype=np.float64)
                tracked_state[:4] = (
                    time_s,
                    tracked_object.center.x,
                    tracked_object.center.y,
                    tracked_object.center.heading,
                )

                if tracked_object.tracked_object_type in AGENT_TYPES:
                    # extract additional states for dynamic objects
                    tracked_state[4:] = (
                        tracked_object.velocity.x,
                        tracked_object.velocity.y,
                    )

                # found new object
                if token not in detection_tracks_states.keys():
                    detection_tracks_states[token] = [tracked_state]
                    unique_detection_tracks[token] = tracked_object

                # object already existed
                else:
                    detection_tracks_states[token].append(tracked_state)

        # create time interpolators
        detection_interpolators: Dict[str, StateInterpolator] = {}
        for token, states_list in detection_tracks_states.items():
            states = np.array(states_list, dtype=np.float64)
            detection_interpolators[token] = StateInterpolator(states)

        # interpolate at 10Hz
        interpolated_time_s = (
            np.arange(0, int(time_horizon / interpolate_step) + 1, 1, dtype=float)
            * interpolate_step
        )

        interpolated_detection_tracks = []
        for time_s in interpolated_time_s:
            interpolated_tracks = []
            for token, interpolator in detection_interpolators.items():
                initial_detection_track = unique_detection_tracks[token]
                interpolated_state = interpolator.interpolate(time_s)

                if interpolator.start_time == interpolator.end_time:
                    interpolated_tracks.append(initial_detection_track)

                elif interpolated_state is not None:

                    tracked_type = initial_detection_track.tracked_object_type
                    metadata = (
                        initial_detection_track.metadata
                    )  # copied since time stamp is ignored

                    oriented_box = OrientedBox(
                        StateSE2(*interpolated_state[:3]),
                        initial_detection_track.box.length,
                        initial_detection_track.box.width,
                        initial_detection_track.box.height,
                    )

                    if tracked_type in AGENT_TYPES:
                        velocity = StateVector2D(*interpolated_state[3:])

                        detection_track = Agent(
                            tracked_object_type=tracked_type,
                            oriented_box=oriented_box,
                            velocity=velocity,
                            metadata=initial_detection_track.metadata,  # simply copy
                        )
                    else:
                        detection_track = StaticObject(
                            tracked_object_type=tracked_type,
                            oriented_box=oriented_box,
                            metadata=metadata,
                        )

                    interpolated_tracks.append(detection_track)
            interpolated_detection_tracks.append(
                DetectionsTracks(TrackedObjects(interpolated_tracks))
            )

        # convert to pdm observation
        pdm_observation = PDMObservation(
            self._future_sampling,
            self._proposal_sampling,
            self._map_radius,
            observation_sample_res=1,
        )
        pdm_observation.update_detections_tracks(interpolated_detection_tracks)
        return pdm_observation
    
    def get_ego_gt(self, scenario: AbstractScenario):
        ego_cur_state = scenario.initial_ego_state

        # ego features
        past_ego_trajectory = scenario.get_ego_past_trajectory(
            iteration=0,
            time_horizon=self.history_horizon,
            num_samples=self.history_samples,
        )
        future_ego_trajectory = scenario.get_ego_future_trajectory(
            iteration=0,
            time_horizon=self.future_horizon,
            num_samples=self.future_samples,
        )
        ego_state_list = (
            list(past_ego_trajectory) + [ego_cur_state] + list(future_ego_trajectory)
        )

        present_idx = 20
        data = {}
        data["current_state"] = self._get_ego_current_state(
            ego_state_list[present_idx], ego_state_list[present_idx - 1]
        )
        ego_features = self._get_ego_features(ego_states=ego_state_list)
        data['trajectory'] = ego_features

        return data
    
    def _get_ego_current_state(self, ego_state: EgoState, prev_state: EgoState):

        steering_angle, yaw_rate = self.calculate_additional_ego_states(
            ego_state, prev_state
        )

        state = np.zeros(9, dtype=np.float64)
        state[0:2] = ego_state.rear_axle.array
        state[2] = ego_state.rear_axle.heading
        state[3] = ego_state.dynamic_car_state.rear_axle_velocity_2d.x
        state[4] = ego_state.dynamic_car_state.rear_axle_velocity_2d.y
        state[5] = ego_state.dynamic_car_state.rear_axle_acceleration_2d.x
        state[6] = ego_state.dynamic_car_state.rear_axle_acceleration_2d.y
        state[7] = steering_angle
        state[8] = yaw_rate
        return state
    
    def calculate_additional_ego_states(
        self, current_state: EgoState, prev_state: EgoState, dt=0.1
    ):
        cur_velocity = current_state.dynamic_car_state.rear_axle_velocity_2d.x
        angle_diff = current_state.rear_axle.heading - prev_state.rear_axle.heading
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        yaw_rate = angle_diff / 0.1

        if abs(cur_velocity) < 0.2:
            return 0.0, 0.0  # if the car is almost stopped, the yaw rate is unreliable
        else:
            steering_angle = np.arctan(
                yaw_rate * self.ego_params.wheel_base / abs(cur_velocity)
            )
            steering_angle = np.clip(steering_angle, -2 / 3 * np.pi, 2 / 3 * np.pi)
            yaw_rate = np.clip(yaw_rate, -0.95, 0.95)

            return steering_angle, yaw_rate
    
    def _get_ego_features(self, ego_states: List[EgoState]):
        """note that rear axle velocity and acceleration are in ego local frame,
        and need to be transformed to the global frame.
        """
        T = len(ego_states)

        position = np.zeros((T, 2), dtype=np.float64)
        heading = np.zeros((T), dtype=np.float64)
        velocity = np.zeros((T, 2), dtype=np.float64)
        acceleration = np.zeros((T, 2), dtype=np.float64)
        shape = np.zeros((T, 2), dtype=np.float64)
        valid_mask = np.ones(T, dtype=np.bool)

        for t, state in enumerate(ego_states):
            position[t] = state.rear_axle.array
            heading[t] = state.rear_axle.heading
        
        return np.concatenate(
            [position, heading[..., None]], axis=-1
        )


    def compute_metric_cache(self, scenario: AbstractScenario):

        # file_name = (
        #     self._cache_path
        #     / scenario.log_name
        #     / scenario.scenario_type
        #     / scenario.token
        #     / "metric_cache.pkl"
        # )

        # init and run PDM-Closed
        planner_input, planner_initialization = self._get_planner_inputs(scenario)
        self._pdm_closed.initialize(planner_initialization)
        pdm_closed_trajectory, max_score = self._pdm_closed.compute_planner_trajectory(planner_input)

        observation = self._interpolate_gt_observation(scenario)

        # save and dump features
        ret_dict =  dict(
            trajectory=pdm_closed_trajectory,
            ego_state=scenario.initial_ego_state,
            observation=observation,
            centerline=self._pdm_closed._centerline,
            route_lane_ids=self._pdm_closed._route_lane_dict,
            drivable_area_map=self._pdm_closed._drivable_area_map,
            org_score=max_score,
            scenario_info=dict(
                log_name=scenario.log_name,
                scenario_type=scenario.scenario_type,
                token=scenario.token,
            )
        )

        if self.save_path is not None:
            with open(self.save_path + f'/{scenario.token}.pkl', 'wb') as writer:
                pickle.dump(ret_dict, writer)

        return ret_dict

        # return metadata
        # return CacheMetadataEntry(file_name)