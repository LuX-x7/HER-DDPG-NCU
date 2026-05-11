# import numpy as np


# class ObstacleAwareReward:
#     """Sparse Fetch reward plus a soft safety penalty near obstacle boxes."""

#     def __init__(self, env, base_reward_func, weight=0.0, safe_distance=0.10):
#         self.base_reward_func = base_reward_func
#         self.weight = float(weight)
#         self.safe_distance = float(safe_distance)
#         self.boxes = self._read_obstacle_boxes(env)
#         self.dynamic_obs_dim = 12

#     def _read_obstacle_boxes(self, env):
#         boxes = []
#         unwrapped = env.unwrapped
#         if not hasattr(unwrapped, "model"):
#             return boxes

#         model = unwrapped.model
#         names = getattr(unwrapped, "_model_names", None)
#         for name in ("obstacle1", "obstacle2"):
#             try:
#                 body_id = names.body_name2id[name] if names is not None else model.body(name).id
#             except Exception:
#                 continue

#             geom_adr = int(model.body_geomadr[body_id])
#             geom_num = int(model.body_geomnum[body_id])
#             if geom_num <= 0:
#                 continue

#             boxes.append(
#                 {
#                     "center": model.body_pos[body_id].copy(),
#                     "half_size": model.geom_size[geom_adr].copy(),
#                 }
#             )
#         return boxes

#     def _centers_from_obs_next(self, info, batch_size):
#         if not info or "obs_next" not in info:
#             return None

#         obs_next = np.asarray(info["obs_next"])
#         if obs_next.ndim != 2 or obs_next.shape[1] < self.dynamic_obs_dim:
#             return None

#         obstacle_pos = obs_next[:, -self.dynamic_obs_dim : -self.dynamic_obs_dim + 6]
#         if obstacle_pos.shape[0] != batch_size:
#             return None
#         return obstacle_pos.reshape(batch_size, 2, 3)

#     def _aabb_distance(self, points, centers, half_size):
#         delta = np.maximum(np.abs(points - centers) - half_size, 0.0)
#         return np.linalg.norm(delta, axis=-1)

#     def __call__(self, achieved_goal, desired_goal, info):
#         reward = self.base_reward_func(achieved_goal, desired_goal, info)
#         if self.weight <= 0 or not self.boxes:
#             return reward

#         achieved_goal = np.asarray(achieved_goal)
#         single = achieved_goal.ndim == 1
#         points = achieved_goal.reshape(1, -1) if single else achieved_goal

#         dynamic_centers = self._centers_from_obs_next(info, points.shape[0])
#         penalty = np.zeros(points.shape[0], dtype=np.float32)
#         for idx, box in enumerate(self.boxes):
#             if dynamic_centers is not None and idx < dynamic_centers.shape[1]:
#                 centers = dynamic_centers[:, idx, :]
#             else:
#                 centers = np.broadcast_to(box["center"], points.shape)

#             distances = self._aabb_distance(points, centers, box["half_size"])
#             penalty += np.clip((self.safe_distance - distances) / self.safe_distance, 0.0, 1.0)

#         shaped = reward - self.weight * penalty
#         return shaped[0] if single else shaped


###第一版


import numpy as np


class ObstacleAwareReward:
    """
    Sparse Fetch reward + dense goal reward + obstacle safety penalty + collision penalty.

    奖励结构：
        r_total = r_sparse
                  + dense_reward_weight * r_dense
                  - obstacle_reward_weight * obstacle_penalty
                  - collision_penalty * collision_flag

    其中：
        r_dense = - ||achieved_goal - desired_goal||
        obstacle_penalty = max(0, safe_distance - distance_to_obstacle) / safe_distance
        collision_flag = 1 if distance_to_obstacle <= collision_distance else 0
    """

    def __init__(
        self,
        env,
        base_reward_func,
        weight=0.0,
        safe_distance=0.10,
        dense_reward_weight=0.0,
        collision_penalty=0.0,
        collision_distance=0.02,
    ):
        self.base_reward_func = base_reward_func

        # 障碍物安全距离软惩罚权重
        self.weight = float(weight)
        self.safe_distance = float(safe_distance)

        # 到目标距离的稠密奖励权重
        self.dense_reward_weight = float(dense_reward_weight)

        # 碰撞惩罚
        self.collision_penalty = float(collision_penalty)
        self.collision_distance = float(collision_distance)

        # 从 MuJoCo XML 中读取 obstacle1 / obstacle2 的 box 信息
        self.boxes = self._read_obstacle_boxes(env)

        # dynamic_obs_wrapper 中拼接的障碍物状态维度：
        # obstacle1_pos(3) + obstacle2_pos(3) + obstacle1_vel(3) + obstacle2_vel(3)
        self.dynamic_obs_dim = 12

    def _read_obstacle_boxes(self, env):
        boxes = []
        unwrapped = env.unwrapped
        if not hasattr(unwrapped, "model"):
            return boxes

        model = unwrapped.model
        names = getattr(unwrapped, "_model_names", None)

        for name in ("obstacle1", "obstacle2"):
            try:
                body_id = names.body_name2id[name] if names is not None else model.body(name).id
            except Exception:
                continue

            geom_adr = int(model.body_geomadr[body_id])
            geom_num = int(model.body_geomnum[body_id])
            if geom_num <= 0:
                continue

            boxes.append(
                {
                    "center": model.body_pos[body_id].copy(),
                    "half_size": model.geom_size[geom_adr].copy(),
                }
            )

        return boxes

    def _centers_from_obs_next(self, info, batch_size):
        """
        从 obs_next 末尾读取动态障碍物位置。

        obs_next 最后 12 维格式：
            [obs1_x, obs1_y, obs1_z, obs2_x, obs2_y, obs2_z,
             obs1_vx, obs1_vy, obs1_vz, obs2_vx, obs2_vy, obs2_vz]
        """
        if not info or "obs_next" not in info:
            return None

        obs_next = np.asarray(info["obs_next"])

        # HER batch 情况下 obs_next 是二维：(batch, obs_dim)
        if obs_next.ndim != 2 or obs_next.shape[1] < self.dynamic_obs_dim:
            return None

        obstacle_pos = obs_next[:, -self.dynamic_obs_dim : -self.dynamic_obs_dim + 6]
        if obstacle_pos.shape[0] != batch_size:
            return None

        return obstacle_pos.reshape(batch_size, 2, 3)

    def _aabb_distance(self, points, centers, half_size):
        """
        计算点到 AABB box 的距离。

        points:  (batch, 3)
        centers: (batch, 3)
        half_size: (3,)
        """
        delta = np.maximum(np.abs(points - centers) - half_size, 0.0)
        return np.linalg.norm(delta, axis=-1)

    def _obstacle_terms(self, points, info):
        """
        计算障碍物相关项：
        1. obstacle_penalty：安全距离内的软惩罚
        2. min_distances：每个点到最近障碍物的距离
        3. collision_flags：是否发生碰撞
        """
        batch_size = points.shape[0]

        obstacle_penalty = np.zeros(batch_size, dtype=np.float32)
        min_distances = np.full(batch_size, np.inf, dtype=np.float32)
        collision_flags = np.zeros(batch_size, dtype=np.float32)

        if not self.boxes:
            return obstacle_penalty, min_distances, collision_flags

        dynamic_centers = self._centers_from_obs_next(info, batch_size)

        for idx, box in enumerate(self.boxes):
            if dynamic_centers is not None and idx < dynamic_centers.shape[1]:
                centers = dynamic_centers[:, idx, :]
            else:
                centers = np.broadcast_to(box["center"], points.shape)

            distances = self._aabb_distance(points, centers, box["half_size"])

            min_distances = np.minimum(min_distances, distances)

            if self.safe_distance > 0:
                obstacle_penalty += np.clip(
                    (self.safe_distance - distances) / self.safe_distance,
                    0.0,
                    1.0,
                )

            collision_flags = np.maximum(
                collision_flags,
                (distances <= self.collision_distance).astype(np.float32),
            )

        return obstacle_penalty, min_distances, collision_flags

    def safety_metrics(self, achieved_goal, info=None):
        """
        给评估阶段使用的安全指标。

        返回：
            single 输入：
                {
                    "min_distance": float,
                    "collision": float
                }

            batch 输入：
                {
                    "min_distance": np.ndarray,
                    "collision": np.ndarray
                }
        """
        achieved_goal = np.asarray(achieved_goal)
        single = achieved_goal.ndim == 1
        points = achieved_goal.reshape(1, -1) if single else achieved_goal

        _, min_distances, collision_flags = self._obstacle_terms(points, info)

        if single:
            return {
                "min_distance": float(min_distances[0]),
                "collision": float(collision_flags[0]),
            }

        return {
            "min_distance": min_distances,
            "collision": collision_flags,
        }

    def __call__(self, achieved_goal, desired_goal, info):
        """
        保持 HER / Fetch compute_reward 的接口不变。

        achieved_goal:
            单条: (goal_dim,)
            batch: (batch, goal_dim)

        desired_goal:
            单条: (goal_dim,)
            batch: (batch, goal_dim)

        return:
            单条: scalar
            batch: (batch,)
        """
        base_reward = self.base_reward_func(achieved_goal, desired_goal, info)

        achieved_goal = np.asarray(achieved_goal, dtype=np.float32)
        desired_goal = np.asarray(desired_goal, dtype=np.float32)

        single = achieved_goal.ndim == 1

        points = achieved_goal.reshape(1, -1) if single else achieved_goal
        goals = desired_goal.reshape(1, -1) if desired_goal.ndim == 1 else desired_goal

        batch_size = points.shape[0]

        reward = np.asarray(base_reward, dtype=np.float32)

        if reward.ndim == 0:
            shaped_reward = np.full(batch_size, float(reward), dtype=np.float32)
        else:
            shaped_reward = reward.reshape(-1).astype(np.float32)

        if shaped_reward.shape[0] != batch_size:
            shaped_reward = np.broadcast_to(shaped_reward, (batch_size,)).copy()

        # 1. 到目标距离的稠密奖励：距离越远，惩罚越大
        if self.dense_reward_weight != 0.0:
            dense_reward = -np.linalg.norm(points - goals, axis=-1)
            shaped_reward += self.dense_reward_weight * dense_reward

        # 2. 障碍物安全距离惩罚 + 碰撞惩罚
        obstacle_penalty, _, collision_flags = self._obstacle_terms(points, info)

        if self.weight != 0.0:
            shaped_reward -= self.weight * obstacle_penalty

        if self.collision_penalty != 0.0:
            shaped_reward -= self.collision_penalty * collision_flags

        return shaped_reward[0] if single else shaped_reward


