from __future__ import annotations

import warnings
from typing import Any, Generator, List, Optional, Set, Tuple, Type, cast


from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import (
    StateVector2D,
    StateSE2,
    TimePoint,
)
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import (
    TrafficLightStatusType,
    TrafficLightStatusData,
    TrafficLightStatuses,
    Transform,
)

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    SensorChannel,
    Sensors,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.database.maps_db.gpkg_mapsdb import MAP_LOCATIONS

from navsim.planning.scenario_builder.navsim_scenario_utils import (
    annotations_to_detection_tracks,
    sample_future_indices,
)

from navsim.common.dataclasses import Scene

# TODO: Refactor
DUMMY_SCENARIO_TYPE = "unknown"
DUMMY_GOAL_STATE = StateSE2(0, 0, 0)

class NavSimScenario(AbstractScenario):
    # TODO: Refactor

    def __init__(
        self,
        scene: Scene,
        map_root: str,
        map_version: str,
        ego_vehicle_parameters: VehicleParameters = get_pacifica_parameters(),
    ) -> None:

        self._database_interval = 0.5  # interpolated to 10 Hz
        self._scene = scene

        # map attributes
        self._map_root = map_root
        self._map_version = map_version
        
        self._scene_data = scene.scene_metadata
        self._map_name = self._scene_data.map_name
        self._map_name = (
            self._map_name if self._map_name != "las_vegas" else "us-nv-las-vegas-strip"
        )
        
        self._initial_frame_idx = self._scene_data.num_history_frames - 1
        
        self._initial_lidar_token = self._scene.frames[self._initial_frame_idx].token
        self._log_name = self._scene_data.log_name
        self._route_roadblock_ids = self._scene.frames[self._initial_frame_idx].roadblock_ids

        self._time_points = [TimePoint(int(frame.timestamp)) for frame in self._scene.frames]
        self._future_sampling = TrajectorySampling(
            num_poses=len(self._time_points) + 1, interval_length=0.5
        )
        self._ego_vehicle_parameters = ego_vehicle_parameters

    def __reduce__(self) -> Tuple[Type[NavSimScenario], Tuple[Any, ...]]:
        """
        Hints on how to reconstruct the object when pickling.
        :return: Object type and constructor arguments to be used.
        """
        return (
            self.__class__,
            (
                self._scene,
                self._map_root,
                self._map_version,
                self._ego_vehicle_parameters,
            ),
        )

    @property
    def ego_vehicle_parameters(self) -> VehicleParameters:
        """Inherited, see superclass."""
        return self._ego_vehicle_parameters

    @property
    def token(self) -> str:
        """Inherited, see superclass."""
        return self._initial_lidar_token

    @property
    def log_name(self) -> str:
        """Inherited, see superclass."""
        # e.g. "2021.07.16.20.45.29_veh-35_01095_01486.db"
        return self._log_name

    @property
    def scenario_name(self) -> str:
        """Inherited, see superclass."""
        return self.token

    @property
    def scenario_type(self) -> str:
        """Inherited, see superclass."""
        return DUMMY_SCENARIO_TYPE  # TODO: avoid dummy

    @property
    def map_api(self) -> AbstractMap:
        """Inherited, see superclass."""
        assert self._map_name in MAP_LOCATIONS, f"Map location {self._map_name} not available!"
        map_api = get_maps_api(self._map_root, self._map_version, self._map_name)
        return map_api

    @property
    def map_root(self) -> str:
        """Get the map root folder."""
        return self._map_root

    @property
    def map_version(self) -> str:
        """Get the map version."""
        return self._map_version

    @property
    def database_interval(self) -> float:
        """Inherited, see superclass."""
        return self._database_interval

    def get_number_of_iterations(self) -> int:
        """Inherited, see superclass."""
        return len(self._scene.frames)

    def get_lidar_to_ego_transform(self) -> Transform:
        """Inherited, see superclass."""
        raise NotImplementedError

    def get_mission_goal(self) -> Optional[StateSE2]:
        """Inherited, see superclass."""
        return DUMMY_GOAL_STATE  # TODO: avoid dummy

    def get_route_roadblock_ids(self) -> List[str]:
        """Inherited, see superclass."""
        return cast(List[str], self._route_roadblock_ids)

    def get_expert_goal_state(self) -> StateSE2:
        """Inherited, see superclass."""
        return DUMMY_GOAL_STATE  # TODO: avoid dummy

    def get_time_point(self, iteration: int) -> TimePoint:
        """Inherited, see superclass."""
        
        frame_idx = self._initial_frame_idx + iteration 
        assert (
            0 <= frame_idx < self.get_number_of_iterations()
        ), f"Iteration {frame_idx} out of bound of {self.get_number_of_iterations()} iterations!"
        return self._time_points[frame_idx]

    def get_ego_state_at_iteration(self, iteration: int) -> EgoState:
        """Inherited, see superclass."""
        
        frame_idx = self._initial_frame_idx + iteration 
        assert (
            0 <= frame_idx < self.get_number_of_iterations()
        ), f"Iteration {frame_idx} out of bound of {self.get_number_of_iterations()} iterations!"

        rear_axle_velocity_2d = StateVector2D(
            *self._scene.frames[frame_idx].ego_status.ego_velocity
        )
        rear_axle_acceleration_2d = StateVector2D(
            *self._scene.frames[frame_idx].ego_status.ego_acceleration
        )
        return EgoState.build_from_rear_axle(
            StateSE2(*self._scene.frames[frame_idx].ego_status.ego_pose),
            tire_steering_angle=0.0,
            vehicle_parameters=self._ego_vehicle_parameters,
            time_point=self.get_time_point(iteration),
            rear_axle_velocity_2d=rear_axle_velocity_2d,
            rear_axle_acceleration_2d=rear_axle_acceleration_2d,
        )

    def get_tracked_objects_at_iteration(
        self,
        iteration: int,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> DetectionsTracks:
        """Inherited, see superclass."""
        frame_idx = self._initial_frame_idx + iteration 
        assert (
            0 <= frame_idx < self.get_number_of_iterations()
        ), f"Iteration is out of scenario: {frame_idx}!"

        if future_trajectory_sampling:
            warnings.warn(
                "NavSimScenario: TrajectorySampling in get_tracked_objects_at_iteration() not supported."
            )

        ego_state = self.get_ego_state_at_iteration(iteration)
        return annotations_to_detection_tracks(self._scene.frames[frame_idx].annotations, ego_state)

    def get_tracked_objects_within_time_window_at_iteration(
        self,
        iteration: int,
        past_time_horizon: float,
        future_time_horizon: float,
        filter_track_tokens: Optional[Set[str]] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> DetectionsTracks:
        """Inherited, see superclass."""
        assert (
            0 <= iteration < self.get_number_of_iterations()
        ), f"Iteration is out of scenario: {iteration}!"
        raise NotImplementedError

    def get_sensors_at_iteration(
        self, iteration: int, channels: Optional[List[SensorChannel]] = None
    ) -> Sensors:
        """Inherited, see superclass."""
        raise NotImplementedError

    def get_future_timestamps(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TimePoint, None, None]:
        """Inherited, see superclass."""
        indices = sample_future_indices(self._future_sampling, iteration, time_horizon, num_samples)
        for idx in indices:
            yield self.get_time_point(idx)

    def get_past_timestamps(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TimePoint, None, None]:
        """Inherited, see superclass."""
        # FIXME:
        yield self.get_time_point(0)

    def get_ego_past_trajectory(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[EgoState, None, None]:
        """Inherited, see superclass."""
        # FIXME:
        yield self.get_ego_state_at_iteration(0)

    def get_ego_future_trajectory(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[EgoState, None, None]:
        """Inherited, see superclass."""
        indices = sample_future_indices(self._future_sampling, iteration, time_horizon, num_samples)
        for idx in indices:
            yield self.get_ego_state_at_iteration(idx)

    def get_past_tracked_objects(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> Generator[DetectionsTracks, None, None]:
        """Inherited, see superclass."""
        # FIXME: add history stats
        yield self.get_tracked_objects_at_iteration(0)

    def get_future_tracked_objects(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> Generator[DetectionsTracks, None, None]:
        """Inherited, see superclass."""

        indices = sample_future_indices(self._future_sampling, iteration, time_horizon, num_samples)
        for idx in indices:
            yield self.get_tracked_objects_at_iteration(idx)

    def get_past_sensors(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
        channels: Optional[List[SensorChannel]] = None,
    ) -> Generator[Sensors, None, None]:
        """Inherited, see superclass."""
        raise NotImplementedError

    def get_traffic_light_status_at_iteration(
        self, iteration: int
    ) -> Generator[TrafficLightStatusData, None, None]:
        """Inherited, see superclass."""
        
        frame_idx = iteration + self._initial_frame_idx

        for lane_connector_id, is_red in self._scene.frames[frame_idx].traffic_lights:
            status = TrafficLightStatusType.RED if is_red else TrafficLightStatusType.GREEN
            yield TrafficLightStatusData(status, lane_connector_id, self.get_time_point(iteration))

    def get_past_traffic_light_status_history(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TrafficLightStatuses, None, None]:
        """
        Gets past traffic light status.

        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations.
        :param time_horizon [s]: the desired horizon to the past.
        :param num_samples: number of entries in the future, if None it will be deduced from the DB.
        :return: Generator object for traffic light history to the past.
        """
        # FIXME: add traffic light stats
        yield from []  # placeholder

    def get_future_traffic_light_status_history(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TrafficLightStatuses, None, None]:
        """
        Gets future traffic light status.

        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations.
        :param time_horizon [s]: the desired horizon to the future.
        :param num_samples: number of entries in the future, if None it will be deduced from the DB.
        :return: Generator object for traffic light history to the future.
        """
        # FIXME: add traffic light stats
        yield from []  # placeholder

    def get_scenario_tokens(self) -> List[str]:
        """Return the list of lidarpc tokens from the DB that are contained in the scenario."""
        raise NotImplementedError
