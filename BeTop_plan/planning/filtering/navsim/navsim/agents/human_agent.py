from typing import List
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Trajectory, Scene, SensorConfig

class HumanAgent(AbstractAgent):

    requires_scene = True

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(
            time_horizon=4, interval_length=0.5
        ),
    ):
        self._trajectory_sampling = trajectory_sampling

    def name(self) -> str:
        """Inherited, see superclass."""

        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        pass

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_no_sensors()

    def compute_trajectory(self, agent_input: AgentInput, scene: Scene) -> Trajectory:
        """
        Computes the ego vehicle trajectory.
        :param current_input: Dataclass with agent inputs.
        :return: Trajectory representing the predicted ego's position in future
        """
        return scene.get_future_trajectory(self._trajectory_sampling.num_poses)