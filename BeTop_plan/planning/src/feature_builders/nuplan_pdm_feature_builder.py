from __future__ import annotations
'''
Behavioral Topology (BeTop): https://arxiv.org/abs/2409.18031
'''
'''
Pipeline developed upon planTF: 
https://arxiv.org/pdf/2309.10443
'''
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

from ..features.nuplan_feature import NuplanFeature
from .common.route_utils import route_roadblock_correction
from .common.utils import interpolate_polyline, rotate_round_z_axis

import shapely


class NuplanFeatureBuilder(AbstractFeatureBuilder):
    """Feature builder class for PDMOpen and PDMOffset."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        history_sampling: TrajectorySampling,
        planner: Optional[PDMClosedPlanner],
        centerline_samples: int = 120,
        centerline_interval: float = 1.0,
        
        radius: float = 100,
        history_horizon: float = 2,
        future_horizon: float = 8,
        sample_interval: float = 0.1,
        max_agents: int = 64,
        persepect_norm: bool = False
    ):
        """
        Constructor for PDMFeatureBuilder
        :param history_sampling: dataclass for storing trajectory sampling
        :param centerline_samples: number of centerline poses
        :param centerline_interval: interval of centerline poses [m]
        :param planner: PDMClosed planner for correction
        """
        assert (
            type(planner) == PDMClosedPlanner or planner is None
        ), f"PDMFeatureBuilder: Planner must be PDMClosedPlanner or None, but got {type(planner)}"

        self._trajectory_sampling = trajectory_sampling
        self._history_sampling = history_sampling
        self._centerline_samples = centerline_samples
        self._centerline_interval = centerline_interval

        self._planner = planner

        print('feature builder loaded!')

        self.radius = radius
        self.history_horizon = history_horizon
        self.future_horizon = future_horizon
        self.history_samples = int(self.history_horizon / sample_interval)
        self.future_samples = int(self.future_horizon / sample_interval)
        self.sample_interval = sample_interval
        self.ego_params = get_pacifica_parameters()
        self.length = self.ego_params.length
        self.width = self.ego_params.width
        self.max_agents = max_agents
        self.persepect_norm = persepect_norm

        self.interested_objects_types = [
            TrackedObjectType.EGO,
            TrackedObjectType.VEHICLE,
            TrackedObjectType.PEDESTRIAN,
            TrackedObjectType.BICYCLE,
        ]
        self.polygon_types = [
            SemanticMapLayer.LANE,
            SemanticMapLayer.LANE_CONNECTOR,
            SemanticMapLayer.CROSSWALK,
        ]

    def get_feature_type(self) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return NuplanFeature  # type: ignore

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "feature"

    def get_features_from_scenario(self, scenario: AbstractScenario):
        """Inherited, see superclass."""

        # print('get from scenario')
        
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

        past_ego_states = [
            ego_state
            for ego_state in scenario.get_ego_past_trajectory(
                iteration=0,
                time_horizon=self._history_sampling.time_horizon,
                num_samples=self._history_sampling.num_poses,
            )
        ] + [scenario.initial_ego_state]

        # agents features
        present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
        past_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_past_tracked_objects(
                iteration=0,
                time_horizon=self.history_horizon,
                num_samples=self.history_samples,
            )
        ]
        future_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_future_tracked_objects(
                iteration=0,
                time_horizon=self.future_horizon,
                num_samples=self.future_samples,
            )
        ]
        tracked_objects_list = (
            past_tracked_objects + [present_tracked_objects] + future_tracked_objects
        )

        current_input, initialization = self._get_planner_params_from_scenario(scenario)

        return self._compute_feature(
            present_idx=self.history_samples,
            ego_state_list=ego_state_list,
            tracked_objects_list=tracked_objects_list,
            route_roadblocks_ids=scenario.get_route_roadblock_ids(),
            map_api=scenario.map_api,
            mission_goal=scenario.get_mission_goal(),
            traffic_light_status=scenario.get_traffic_light_status_at_iteration(0),
            #PDM
            ego_states=past_ego_states,
            current_input=current_input, 
            initialization=initialization,
            )

    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> PDMFeature:
        """Inherited, see superclass."""

        history = current_input.history
        current_ego_state, _ = history.current_state
        past_ego_states = history.ego_states[:-1]

        indices = sample_indices_with_time_horizon(
            self._history_sampling.num_poses, self._history_sampling.time_horizon, history.sample_interval
        )
        past_ego_states = [past_ego_states[-idx] for idx in reversed(indices)] + [
            current_ego_state
        ]

        ######
        history = current_input.history
        tracked_objects_list = [
            observation.tracked_objects for observation in history.observations
        ]

        horizon = self.history_samples + 1

        return self._compute_feature(
            ego_states=past_ego_states,
            current_input=current_input, 
            initialization=initialization,
            #####
            present_idx=-1,
            ego_state_list=history.ego_states[-horizon:],
            tracked_objects_list=tracked_objects_list[-horizon:],
            route_roadblocks_ids=initialization.route_roadblock_ids,
            map_api=initialization.map_api,
            mission_goal=initialization.mission_goal,
            traffic_light_status=current_input.traffic_light_data)

    def _get_planner_params_from_scenario(
        self, scenario: AbstractScenario
    ) -> Tuple[PlannerInput, PlannerInitialization]:
        """
        Creates planner input arguments from scenario object.
        :param scenario: scenario object of nuPlan
        :return: tuple of planner input and initialization objects
        """

        buffer_size = int(2 / scenario.database_interval + 1)

        # Initialize Planner
        planner_initialization = PlannerInitialization(
            route_roadblock_ids=scenario.get_route_roadblock_ids(),
            mission_goal=scenario.get_mission_goal(),
            map_api=scenario.map_api,
        )

        history = SimulationHistoryBuffer.initialize_from_scenario(
            buffer_size=buffer_size,
            scenario=scenario,
            observation_type=DetectionsTracks,
        )

        planner_input = PlannerInput(
            iteration=SimulationIteration(index=0, time_point=scenario.start_time),
            history=history,
            traffic_light_data=list(scenario.get_traffic_light_status_at_iteration(0)),
        )

        return planner_input, planner_initialization

    def _compute_feature(
        self,
        ego_states: List[EgoState],
        current_input: PlannerInput,
        initialization: PlannerInitialization,

        present_idx: int,
        ego_state_list: List[EgoState],
        tracked_objects_list: List[TrackedObjects],
        route_roadblocks_ids: list[int],
        map_api: AbstractMap,
        mission_goal: StateSE2,
        traffic_light_status: List[TrafficLightStatusData] = None,
    ):
        """
        Creates PDMFeature dataclass based in ego history, and planner input
        :param ego_states: list of ego states
        :param current_input: planner input of current frame
        :param initialization: planner initialization of current frame
        :return: PDMFeature dataclass
        """

        ######################## planner inputs ########################
        present_ego_state = ego_state_list[present_idx]
        query_xy = present_ego_state.center

        route_roadblocks_ids = route_roadblock_correction(
            present_ego_state, map_api, route_roadblocks_ids
        )

        data = {}
        data["current_state"] = self._get_ego_current_state(
            ego_state_list[present_idx], ego_state_list[present_idx - 1]
        )

        ego_features = self._get_ego_features(ego_states=ego_state_list)
        agent_features, token = self._get_agent_features(
            query_xy=query_xy,
            present_idx=present_idx,
            tracked_objects_list=tracked_objects_list,
        )
        data['agent_token'] = token

        data["agent"] = {}
        for k in agent_features.keys():
            data["agent"][k] = np.concatenate(
                [ego_features[k][None, ...], agent_features[k]], axis=0
            )

        data["map"] = self._get_map_features(
            map_api=map_api,
            query_xy=query_xy,
            route_roadblock_ids=route_roadblocks_ids,
            traffic_light_status=traffic_light_status,
            radius=self.radius,
        )

        ######################## ref-plan inputs ########################
        current_ego_state: EgoState = ego_states[-1]
        current_pose: StateSE2 = current_ego_state.rear_axle

        # extract ego vehicle history states
        ego_position = get_ego_position(ego_states)
        ego_velocity = get_ego_velocity(ego_states)
        ego_acceleration = get_ego_acceleration(ego_states)

        # run planner
        self._planner.initialize(initialization)
        trajectory, score = self._planner.compute_full_planner_trajectory(
            current_input
        )

        planner_trajectory = convert_absolute_to_relative_se2_array(
            current_pose, trajectory
        )  # convert to relative coords

        # extract planner centerline
        centerline: PDMPath = self._planner._centerline
        current_progress: float = centerline.project(Point(*current_pose.array))
        centerline_progress_values = (
            np.arange(self._centerline_samples, dtype=np.float64)
            * self._centerline_interval
            + current_progress
        )  # distance values to interpolate
        planner_centerline = convert_absolute_to_relative_se2_array(
            current_pose,
            centerline.interpolate(centerline_progress_values, as_array=True),
        )  # convert to relative coords

        data['pdm'] = {}
        data['pdm']['centerline'] = planner_centerline
        data['pdm']['current_state'] = ego_state_to_state_array(ego_state_list[present_idx])
        data['pdm']['planner_trajectory'] = planner_trajectory
        data['pdm']['score'] = score

        return NuplanFeature.normalize(data, first_time=True, radius=self.radius, persepect_norm=self.persepect_norm)

    def _get_ego_current_state(self, ego_state: EgoState, prev_state: EgoState):
        steering_angle, yaw_rate = self.calculate_additional_ego_states(
            ego_state, prev_state
        )

        state = np.zeros(7, dtype=np.float64)
        state[0:2] = ego_state.rear_axle.array
        state[2] = ego_state.rear_axle.heading
        state[3] = ego_state.dynamic_car_state.rear_axle_velocity_2d.x
        state[4] = ego_state.dynamic_car_state.rear_axle_acceleration_2d.x
        state[5] = steering_angle
        state[6] = yaw_rate
        return state
    

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
            velocity[t] = rotate_round_z_axis(
                state.dynamic_car_state.rear_axle_velocity_2d.array,
                -state.rear_axle.heading,
            )
            acceleration[t] = rotate_round_z_axis(
                state.dynamic_car_state.rear_axle_acceleration_2d.array,
                -state.rear_axle.heading,
            )
            shape[t] = np.array([self.width, self.length])

        category = np.array(
            self.interested_objects_types.index(TrackedObjectType.EGO), dtype=np.int8
        )

        return {
            "position": position,
            "heading": heading,
            "velocity": velocity,
            "acceleration": acceleration,
            "shape": shape,
            "category": category,
            "valid_mask": valid_mask,
        }

    def _get_agent_features(
        self,
        query_xy: Point2D,
        present_idx: int,
        tracked_objects_list: List[TrackedObjects],
    ):
        present_tracked_objects = tracked_objects_list[present_idx]
        present_agents = present_tracked_objects.get_tracked_objects_of_types(
            self.interested_objects_types
        )
        N, T = min(len(present_agents), self.max_agents), len(tracked_objects_list)

        position = np.zeros((N, T, 2), dtype=np.float64)
        heading = np.zeros((N, T), dtype=np.float64)
        velocity = np.zeros((N, T, 2), dtype=np.float64)
        shape = np.zeros((N, T, 2), dtype=np.float64)
        category = np.zeros((N,), dtype=np.int8)
        valid_mask = np.zeros((N, T), dtype=np.bool)

        if N == 0:
            return {
                "position": position,
                "heading": heading,
                "velocity": velocity,
                "shape": shape,
                "category": category,
                "valid_mask": valid_mask,
            }, np.zeros((N,))

        agent_ids = np.array([agent.track_token for agent in present_agents])
        agent_cur_pos = np.array([agent.center.array for agent in present_agents])
        distance = np.linalg.norm(agent_cur_pos - query_xy.array[None, :], axis=1)
        agent_ids_sorted_list = agent_ids[np.argsort(distance)[: self.max_agents]]
        agent_ids_sorted = {agent_id: i for i, agent_id in enumerate(agent_ids_sorted_list)}

        for t, tracked_objects in enumerate(tracked_objects_list):
            for agent in tracked_objects.get_tracked_objects_of_types(
                self.interested_objects_types
            ):
                if agent.track_token not in agent_ids_sorted:
                    continue

                idx = agent_ids_sorted[agent.track_token]
                position[idx, t] = agent.center.array
                heading[idx, t] = agent.center.heading
                velocity[idx, t] = agent.velocity.array
                shape[idx, t] = np.array([agent.box.width, agent.box.length])
                valid_mask[idx, t] = True

                if t == present_idx:
                    category[idx] = self.interested_objects_types.index(
                        agent.tracked_object_type
                    )

        return {
            "position": position,
            "heading": heading,
            "velocity": velocity,
            "shape": shape,
            "category": category,
            "valid_mask": valid_mask,
        }, agent_ids_sorted_list

    def _get_map_features(
        self,
        map_api: AbstractMap,
        query_xy: Point2D,
        route_roadblock_ids: List[str],
        traffic_light_status: List[TrafficLightStatusData],
        radius: float,
        sample_points: int = 20,
    ):
        route_ids = set(int(route_id) for route_id in route_roadblock_ids)
        tls = {tl.lane_connector_id: tl.status for tl in traffic_light_status}

        map_objects = map_api.get_proximal_map_objects(
            query_xy,
            radius,
            [
                SemanticMapLayer.LANE,
                SemanticMapLayer.LANE_CONNECTOR,
                SemanticMapLayer.CROSSWALK,
            ],
        )
        lane_objects = (
            map_objects[SemanticMapLayer.LANE]
            + map_objects[SemanticMapLayer.LANE_CONNECTOR]
        )
        crosswalk_objects = map_objects[SemanticMapLayer.CROSSWALK]

        object_ids = [int(obj.id) for obj in lane_objects + crosswalk_objects]
        object_types = (
            [SemanticMapLayer.LANE] * len(map_objects[SemanticMapLayer.LANE])
            + [SemanticMapLayer.LANE_CONNECTOR]
            * len(map_objects[SemanticMapLayer.LANE_CONNECTOR])
            + [SemanticMapLayer.CROSSWALK]
            * len(map_objects[SemanticMapLayer.CROSSWALK])
        )

        M, P = len(lane_objects) + len(crosswalk_objects), sample_points
        point_position = np.zeros((M, 3, P, 2), dtype=np.float64)
        point_vector = np.zeros((M, 3, P, 2), dtype=np.float64)
        point_side = np.zeros((M, 3), dtype=np.int8)
        point_orientation = np.zeros((M, 3, P), dtype=np.float64)
        polygon_center = np.zeros((M, 3), dtype=np.float64)
        polygon_position = np.zeros((M, 2), dtype=np.float64)
        polygon_orientation = np.zeros(M, dtype=np.float64)
        polygon_type = np.zeros(M, dtype=np.int8)
        polygon_on_route = np.zeros(M, dtype=np.bool)
        polygon_tl_status = np.zeros(M, dtype=np.int8)
        polygon_speed_limit = np.zeros(M, dtype=np.float64)
        polygon_has_speed_limit = np.zeros(M, dtype=np.bool)

        for lane in lane_objects:
            object_id = int(lane.id)
            idx = object_ids.index(object_id)
            speed_limit = lane.speed_limit_mps

            centerline = self._sample_discrete_path(
                lane.baseline_path.discrete_path, sample_points + 1
            )
            left_bound = self._sample_discrete_path(
                lane.left_boundary.discrete_path, sample_points + 1
            )
            right_bound = self._sample_discrete_path(
                lane.right_boundary.discrete_path, sample_points + 1
            )
            edges = np.stack([centerline, left_bound, right_bound], axis=0)

            point_vector[idx] = edges[:, 1:] - edges[:, :-1]
            point_position[idx] = edges[:, :-1]
            point_orientation[idx] = np.arctan2(
                point_vector[idx, :, :, 1], point_vector[idx, :, :, 0]
            )
            point_side[idx] = np.arange(3)

            polygon_center[idx] = np.concatenate(
                [
                    centerline[int(sample_points / 2)],
                    [point_orientation[idx, 0, int(sample_points / 2)]],
                ],
                axis=-1,
            )
            polygon_position[idx] = centerline[0]
            polygon_orientation[idx] = point_orientation[idx, 0, 0]
            polygon_type[idx] = self.polygon_types.index(object_types[idx])
            polygon_on_route[idx] = int(lane.get_roadblock_id()) in route_ids
            polygon_tl_status[idx] = (
                tls[object_id] if object_id in tls else TrafficLightStatusType.UNKNOWN
            )
            polygon_has_speed_limit[idx] = speed_limit is not None
            polygon_speed_limit[idx] = (
                lane.speed_limit_mps if lane.speed_limit_mps else 0
            )

        for crosswalk in crosswalk_objects:
            idx = object_ids.index(int(crosswalk.id))
            edges = self._get_crosswalk_edges(crosswalk)
            point_vector[idx] = edges[:, 1:] - edges[:, :-1]
            point_position[idx] = edges[:, :-1]
            point_orientation[idx] = np.arctan2(
                point_vector[idx, :, :, 1], point_vector[idx, :, :, 0]
            )
            point_side[idx] = np.arange(3)
            polygon_center[idx] = np.concatenate(
                [
                    edges[0, int(sample_points / 2)],
                    [point_orientation[idx, 0, int(sample_points / 2)]],
                ],
                axis=-1,
            )
            polygon_position[idx] = edges[0, 0]
            polygon_orientation[idx] = point_orientation[idx, 0, 0]
            polygon_type[idx] = self.polygon_types.index(object_types[idx])
            polygon_on_route[idx] = False
            polygon_tl_status[idx] = TrafficLightStatusType.UNKNOWN
            polygon_has_speed_limit[idx] = False

        return {
            "point_position": point_position,
            "point_vector": point_vector,
            "point_orientation": point_orientation,
            "point_side": point_side,
            "polygon_center": polygon_center,
            "polygon_position": polygon_position,
            "polygon_orientation": polygon_orientation,
            "polygon_type": polygon_type,
            "polygon_on_route": polygon_on_route,
            "polygon_tl_status": polygon_tl_status,
            "polygon_has_speed_limit": polygon_has_speed_limit,
            "polygon_speed_limit": polygon_speed_limit,
        }

    def _sample_discrete_path(self, discrete_path: List[StateSE2], num_points: int):
        path = np.stack([point.array for point in discrete_path], axis=0)
        return interpolate_polyline(path, num_points)

    def _get_crosswalk_edges(
        self, crosswalk: PolygonMapObject, sample_points: int = 21
    ):
        bbox = shapely.minimum_rotated_rectangle(crosswalk.polygon)
        coords = np.stack(bbox.exterior.coords.xy, axis=-1)
        edge1 = coords[[3, 0]]  # right boundary
        edge2 = coords[[2, 1]]  # left boundary

        edges = np.stack([(edge1 + edge2) * 0.5, edge2, edge1], axis=0)  # [3, 2, 2]
        vector = edges[:, 1] - edges[:, 0]  # [3, 2]
        steps = np.linspace(0, 1, sample_points, endpoint=True)[None, :]
        points = edges[:, 0][:, None, :] + vector[:, None, :] * steps[:, :, None]

        return points

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
    



def get_ego_position(ego_states: List[EgoState]) -> npt.NDArray[np.float32]:
    """
    Creates an array of relative positions (x, y, θ)
    :param ego_states: list of ego states
    :return: array of shape (num_frames, 3)
    """
    ego_poses = build_ego_features(ego_states, reverse=True)
    return ego_poses


def get_ego_velocity(ego_states: List[EgoState]) -> npt.NDArray[np.float32]:
    """
    Creates an array of ego's velocities (v_x, v_y, v_θ)
    :param ego_states: list of ego states
    :return: array of shape (num_frames, 3)
    """
    v_x = np.asarray(
        [ego_state.dynamic_car_state.center_velocity_2d.x for ego_state in ego_states]
    )
    v_y = np.asarray(
        [ego_state.dynamic_car_state.center_velocity_2d.y for ego_state in ego_states]
    )
    v_yaw = extract_ego_yaw_rate(ego_states)
    return np.stack([v_x, v_y, v_yaw], axis=1)


def get_ego_acceleration(ego_states: List[EgoState]) -> npt.NDArray[np.float32]:
    """
    Creates an array of ego's accelerations (a_x, a_y, a_θ)
    :param ego_states: list of ego states
    :return: array of shape (num_frames, 3)
    """
    a_x = extract_ego_acceleration(ego_states, "x")
    a_y = extract_ego_acceleration(ego_states, "y")
    a_yaw = extract_ego_yaw_rate(ego_states, deriv_order=2, poly_order=3)
    return np.stack([a_x, a_y, a_yaw], axis=1)