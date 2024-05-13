from typing import Any, Dict, List
import numpy as np

from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance

class MultiPickAndPlace(Task):
    def __init__(
        self,
        sim: PyBullet,
        num_objects: int = 3,  # 여러 개의 객체를 처리하기 위해 객체 수를 추가
        reward_type: str = "sparse",
        distance_threshold: float = 0.05,
        goal_xy_range: float = 0.3,
        goal_z_range: float = 0.2,
        obj_xy_range: float = 0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.num_objects = num_objects
        self.object_size = 0.04
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, goal_z_range])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.objects = []
        self.targets = []
        for i in range(self.num_objects):
            obj_name = f"object_{i}"
            target_name = f"target_{i}"
            obj_pos = self._sample_object()
            self.objects.append(obj_name)
            self.targets.append(target_name)
            self.sim.create_box(
                body_name=obj_name,
                half_extents=np.ones(3) * self.object_size / 2,
                mass=1.0,
                position=obj_pos,
                rgba_color=np.random.rand(4),
            )
            self.sim.create_box(
                body_name=target_name,
                half_extents=np.ones(3) * self.object_size / 2,
                mass=0.0,
                ghost=True,
                position=self._sample_goal(),
                rgba_color=np.random.rand(4) * 0.5 + 0.5,  # 반투명
            )

    def get_obs(self) -> np.ndarray:
        obs = []
        for obj_name in self.objects:
            pos = self.sim.get_base_position(obj_name)
            rot = self.sim.get_base_rotation(obj_name)
            vel = self.sim.get_base_velocity(obj_name)
            ang_vel = self.sim.get_base_angular_velocity(obj_name)
            obs.extend([pos, rot, vel, ang_vel])
        return np.concatenate(obs)

    def is_success(self, achieved_goals: List[np.ndarray], desired_goals: List[np.ndarray]) -> bool:
        success = True
        for achieved_goal, desired_goal in zip(achieved_goals, desired_goals):
            if distance(achieved_goal, desired_goal) >= self.distance_threshold:
                success = False
                break
        return success

    def compute_reward(self, achieved_goals: List[np.ndarray], desired_goals: List[np.ndarray], info: Dict[str, Any]) -> float:
        rewards = 0
        for achieved_goal, desired_goal in zip(achieved_goals, desired_goals):
            d = distance(achieved_goal, desired_goal)
            if self.reward_type == "sparse":
                rewards += -np.array(d > self.distance_threshold, dtype=np.float32)
            else:
                rewards += -d.astype(np.float32)
        return rewards
