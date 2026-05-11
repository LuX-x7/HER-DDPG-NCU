import numpy as np
import gymnasium as gym


class DynamicObsWrapper(gym.Wrapper):
    """
    Move obstacle1 and obstacle2 inside Fetch XML scenes.

    The 2Dynamicobs XML files define the obstacle bodies. This wrapper updates
    their body positions before each physics step to create deterministic
    back-and-forth motion.
    """

    def __init__(self, env):
        super().__init__(env)
        self.episode_step = 0
        self._last_obstacle_pos = None
        self._last_obstacle_vel = np.zeros(6, dtype=np.float32)
        self._init_obstacle_data()
        self._extend_observation_space()

    def _init_obstacle_data(self):
        """Cache obstacle body ids and original positions."""
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, "_model_names"):
            names = unwrapped._model_names
            self.bid1 = names.body_name2id["obstacle1"]
            self.bid2 = names.body_name2id["obstacle2"]
            model = unwrapped.model
            self.orig1 = model.body_pos[self.bid1].copy()
            self.orig2 = model.body_pos[self.bid2].copy()
            self._use_new_bindings = True
        else:
            sim = unwrapped.sim
            self.bid1 = sim.model.body_name2id("obstacle1")
            self.bid2 = sim.model.body_name2id("obstacle2")
            self.orig1 = sim.model.body_pos[self.bid1].copy()
            self.orig2 = sim.model.body_pos[self.bid2].copy()
            self._use_new_bindings = False

    def _extend_observation_space(self):
        base_space = self.env.observation_space
        obs_space = base_space.spaces["observation"]
        dtype = obs_space.dtype
        low = np.concatenate([obs_space.low, np.full(12, -np.inf, dtype=dtype)])
        high = np.concatenate([obs_space.high, np.full(12, np.inf, dtype=dtype)])
        spaces = dict(base_space.spaces)
        spaces["observation"] = gym.spaces.Box(low=low, high=high, dtype=dtype)
        self.observation_space = gym.spaces.Dict(spaces)

    def _obstacle_positions(self):
        if self._use_new_bindings:
            model = self.env.unwrapped.model
            pos1 = model.body_pos[self.bid1].copy()
            pos2 = model.body_pos[self.bid2].copy()
        else:
            model = self.env.unwrapped.sim.model
            pos1 = model.body_pos[self.bid1].copy()
            pos2 = model.body_pos[self.bid2].copy()
        return np.concatenate([pos1, pos2]).astype(np.float32)

    def _augment_observation(self, observation):
        observation = dict(observation)
        obstacle_state = np.concatenate([self._obstacle_positions(), self._last_obstacle_vel])
        observation["observation"] = np.concatenate(
            [observation["observation"], obstacle_state.astype(observation["observation"].dtype)]
        )
        return observation

    def _update_obstacles(self):
        """Update obstacle positions from the current episode step."""
        unwrapped = self.env.unwrapped
        dt = unwrapped.dt if hasattr(unwrapped, "dt") else unwrapped.sim.dt
        t = self.episode_step * dt

        # Obstacle 1 moves along y with a period of about 4 seconds.
        y1 = self.orig1[1] + 0.14 * np.sin(2 * np.pi * 0.25 * t)

        # Obstacle 2 moves along x with a period of about 5 seconds.
        x2 = self.orig2[0] + 0.16 * np.sin(2 * np.pi * 0.20 * t)

        if self._use_new_bindings:
            model = self.env.unwrapped.model
            model.body_pos[self.bid1][:] = [self.orig1[0], y1, self.orig1[2]]
            model.body_pos[self.bid2][:] = [x2, self.orig2[1], self.orig2[2]]
            import mujoco

            mujoco.mj_forward(self.env.unwrapped.model, self.env.unwrapped.data)
        else:
            sim = self.env.unwrapped.sim
            sim.model.body_pos[self.bid1][:] = [self.orig1[0], y1, self.orig1[2]]
            sim.model.body_pos[self.bid2][:] = [x2, self.orig2[1], self.orig2[2]]
            sim.forward()

        current_pos = self._obstacle_positions()
        if self._last_obstacle_pos is None:
            self._last_obstacle_vel = np.zeros_like(current_pos)
        else:
            dt = unwrapped.dt if hasattr(unwrapped, "dt") else unwrapped.sim.dt
            self._last_obstacle_vel = (current_pos - self._last_obstacle_pos) / max(dt, 1e-6)
        self._last_obstacle_pos = current_pos

    def reset(self, **kwargs):
        self.episode_step = 0
        self._last_obstacle_pos = None
        result = self.env.reset(**kwargs)
        self._update_obstacles()
        observation, info = result
        return self._augment_observation(observation), info

    def step(self, action):
        self._update_obstacles()
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_step += 1
        return self._augment_observation(obs), reward, terminated, truncated, info
