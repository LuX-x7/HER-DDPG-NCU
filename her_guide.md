# HER + DDPG 代码库 GPT 修改指南

> **用途**：将本文档配合具体代码片段发送给 GPT，让它帮你精准修改奖励函数、网络结构、训练逻辑等深层代码。
>
> **使用方法**：先阅读本文了解文件依赖关系；修改某部分时，把该文件的完整代码 + 本文的相关摘要一并粘贴给 GPT，并附上 Prompt 模板中的指令。

---

## 0. 项目架构总览

```text
train.py
  ├─ arguments.py           (超参数解析)
  ├─ env_utils.py           (XML 路径劫持 + 环境 ID 管理)
  │    └─ xml_envs/         (MuJoCo XML 场景定义)
  └─ rl_modules/ddpg_agent.py   (核心训练器)
       ├─ rl_modules/models.py          (Actor / Critic 网络)
       ├─ rl_modules/replay_buffer.py   (Episode 级经验回放缓冲)
       ├─ rl_modules/obstacle_reward.py (自定义奖励包装器)
       ├─ rl_modules/dynamic_obs_wrapper.py (动态障碍环境包装)
       ├─ her_modules/her.py            (HER 采样器)
       ├─ mpi_utils/mpi_utils.py        (MPI 参数/梯度同步)
       └─ mpi_utils/normalizer.py       (Running Mean/Std 归一化)
```

**数据流**：`train.py` 创建 `gym.Env` → 可选 `DynamicObsWrapper` 包裹 → `ddpg_agent` 持有 `actor/critic` + `replay_buffer` + `her_sampler` + `normalizer` → `learn()` 循环：采集 episode → `buffer.store_episode()` → `her_sampler.sample_her_transitions()` 替换目标并重算奖励 → `_update_network()` 更新网络 → MPI 同步梯度。

---

## 1. 文件索引：每个文件应该复制哪段代码给 GPT

### 1.1 `arguments.py` —— 超参数定义

**职责**：所有命令行参数。新增自定义参数（如新的奖励权重）必须在这里注册。

**关键代码**：

```python
# 核心参数速览（节选）
parser.add_argument('--env-name', type=str, default='FetchReach-v4')
parser.add_argument('--env-config', type=str, default='free', choices=["free", "2Staticobs", "2Dynamicobs"])
parser.add_argument('--n-epochs', type=int, default=50)
parser.add_argument('--n-cycles', type=int, default=50)
parser.add_argument('--n-batches', type=int, default=40)
parser.add_argument('--buffer-size', type=int, default=int(1e6))
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--lr-actor', type=float, default=0.001)
parser.add_argument('--lr-critic', type=float, default=0.001)
parser.add_argument('--gamma', type=float, default=0.98)
parser.add_argument('--polyak', type=float, default=0.95)          # target network soft-update coefficient
parser.add_argument('--noise-eps', type=float, default=0.2)        # Gaussian action noise scale
parser.add_argument('--random-eps', type=float, default=0.3)       # epsilon for random action exploration
parser.add_argument('--replay-strategy', type=str, default='future')
parser.add_argument('--replay-k', type=int, default=4)
parser.add_argument('--obstacle-reward-weight', type=float, default=0.0)
parser.add_argument('--obstacle-safe-distance', type=float, default=0.10)
parser.add_argument('--cuda', action='store_true')
```

**修改场景**：新增参数时，把 `arguments.py` 完整代码发给 GPT，指令示例：
> "在 arguments.py 中新增一个 `--dense-reward-weight` 参数，默认 0.0，类型 float，帮助文本写‘稠密距离奖励权重’。不要改动其他参数。"

---

### 1.2 `env_utils.py` —— XML 路径与环境 ID 工具

**职责**：将 Gymnasium Robotics 默认的 Fetch XML 路径替换为项目本地 `xml_envs/` 下的版本。

**关键函数签名**：

```python
def resolve_xml_dir(xml_dir=None, env_config=None) -> Path:
    """根据 env_config 返回 xml_envs/<config> 的绝对路径。"""

def patch_fetch_xml_paths(xml_dir):
    """劫持 reach/push/pick_and_place/slide 四个任务的 MODEL_XML_PATH。"""

def env_id_for(env_name: str, env_config: str) -> str:
    """生成运行唯一 ID，如 FetchPush-v4-dynobs。"""
```

**修改场景**：新增第四种环境配置（如 `3Randomobs`）时，需要修改 `ENV_CONFIGS` 和 `resolve_xml_dir`。把本文件完整代码发给 GPT 即可。

---

### 1.3 `rl_modules/models.py` —— Actor / Critic 网络结构

**职责**：定义 DDPG 的 Actor（策略网络）和 Critic（Q 网络）。

**完整类定义**（修改网络结构时，请将以下代码完整复制给 GPT）：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))
        return actions

class critic(nn.Module):
    def __init__(self, env_params):
        super(critic, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value
```

**输入维度说明**：
- `env_params['obs']`：observation 维度（Fetch 默认 25，DynamicObs 下为 25+12=37）
- `env_params['goal']`：目标维度（Fetch 任务中通常是 3）
- `env_params['action']`：动作维度（Fetch 为 4）

**修改场景 Prompt 模板**：
> 以下是 `rl_modules/models.py` 的完整代码。请把 Actor 和 Critic 都改成 4 层 hidden layer，每层 512 个神经元，并在第一层后加入 LayerNorm。保持输入输出维度和接口不变。

---

### 1.4 `rl_modules/ddpg_agent.py` —— 核心训练器（最重要）

**职责**：训练主循环、网络更新、动作选择、评估、GIF 生成、归一化维护。

**修改场景**：这是改动最频繁的文件。根据你想改的部分，把对应方法代码发给 GPT。

#### A. 初始化与模块组装（`__init__` 节选）

```python
class ddpg_agent:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params
        # 网络创建与同步
        self.actor_network = actor(env_params)
        self.critic_network = critic(env_params)
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        self.actor_target_network = actor(env_params)
        self.critic_target_network = critic(env_params)
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        if self.args.cuda:
            self.actor_network.cuda(); self.critic_network.cuda()
            self.actor_target_network.cuda(); self.critic_target_network.cuda()
        # 优化器
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        # 奖励函数包装器（关键！）
        reward_func = ObstacleAwareReward(
            self.env,
            self.env.unwrapped.compute_reward,
            weight=self.args.obstacle_reward_weight,
            safe_distance=self.args.obstacle_safe_distance,
        )
        # HER 采样器
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, reward_func)
        # 回放缓冲区
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        # 归一化器
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)
```

**关键依赖关系**：
- `reward_func` 被传给 `her_sampler`
- `her_sampler.sample_her_transitions` 被作为 `sample_func` 注入 `replay_buffer`
- 因此：**修改奖励函数后，HER  hindsight 样本的奖励会自动用新函数重算**，无需改动 buffer。

#### B. 训练循环（`learn()` 核心逻辑）

```python
    def learn(self):
        for epoch in range(self.args.n_epochs):
            for _ in range(self.args.n_cycles):
                # 1. 采集 num_rollouts_per_mpi 个 episode
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                for _ in range(self.args.num_rollouts_per_mpi):
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                    observation, _ = self.env.reset()
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            input_tensor = self._preproc_inputs(obs, g)
                            pi = self.actor_network(input_tensor)
                            action = self._select_actions(pi)
                        observation_new, reward, terminated, truncated, info = self.env.step(action)
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']
                        ep_obs.append(obs.copy()); ep_ag.append(ag.copy())
                        ep_g.append(g.copy()); ep_actions.append(action.copy())
                        obs = obs_new; ag = ag_new
                    ep_obs.append(obs.copy()); ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs); mb_ag.append(ep_ag)
                    mb_g.append(ep_g); mb_actions.append(ep_actions)
                # 2. 存入 buffer + 更新归一化统计量
                self.buffer.store_episode([np.array(mb_obs), np.array(mb_ag), np.array(mb_g), np.array(mb_actions)])
                self._update_normalizer([...])
                # 3. 网络更新 n_batches 次
                for _ in range(self.args.n_batches):
                    self._update_network()
                # 4. 软更新目标网络
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)
            # 5. 评估 + 日志 + 保存
            if self.args.eval:
                results = self._eval_agent(...)
                wandb.log(results)   # rank 0 only
                torch.save([o_norm.mean, o_norm.std, g_norm.mean, g_norm.std, actor_state_dict], model_path)
```

#### C. 网络更新（`_update_network()`，DDPG 核心）

```python
    def _update_network(self):
        transitions = self.buffer.sample(self.args.batch_size)
        # 预处理 + 归一化
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # Tensor 化 + CUDA
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
        # Critic target
        with torch.no_grad():
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next).detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # Critic loss (MSE)
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # Actor loss (maximize Q + L2 正则)
        actions_real = self.actor_network(inputs_norm_tensor)
        actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
        # Update
        self.actor_optim.zero_grad(); actor_loss.backward(); sync_grads(self.actor_network); self.actor_optim.step()
        self.critic_optim.zero_grad(); critic_loss.backward(); sync_grads(self.critic_network); self.critic_optim.step()
```

**Prompt 模板（改训练逻辑）**：
> 以下是 `rl_modules/ddpg_agent.py` 中 `_update_network()` 的完整代码。请把它改造成 **Twin Delayed DDPG (TD3)** 风格：
> 1. 新增第二个 Critic 网络（critic2）及其 target；
> 2. Critic target 取两个 target Q 的最小值；
> 3. Actor 更新频率降低为每 2 个 critic update 更新 1 次；
> 4. target policy 动作加入裁剪噪声（clip noise 0.5，std 0.2）。
> 保持变量命名风格和原有代码一致，不要改 buffer 接口。

#### D. 动作选择（`_select_actions()`，探索策略）

```python
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'],
                                            size=self.env_params['action'])
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action
```

**Prompt 模板（改探索策略）**：
> 以下是 `_select_actions()` 的代码。请将高斯噪声改为 **OU (Ornstein-Uhlenbeck) 噪声**，并实现 epsilon 随训练 epoch 衰减（从 0.3 线性降到 0.05）。传入当前 epoch 作为参数。

---

### 1.5 `rl_modules/obstacle_reward.py` —— 奖励函数（最常改）

**职责**：在原始稀疏 Fetch 奖励基础上叠加障碍物软惩罚。HER  hindsight 重算奖励时也会调用此对象。

**完整代码**（修改奖励时请完整复制给 GPT）：

```python
import numpy as np

class ObstacleAwareReward:
    """Sparse Fetch reward plus a soft safety penalty near obstacle boxes."""

    def __init__(self, env, base_reward_func, weight=0.0, safe_distance=0.10):
        self.base_reward_func = base_reward_func
        self.weight = float(weight)
        self.safe_distance = float(safe_distance)
        self.boxes = self._read_obstacle_boxes(env)
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
            boxes.append({
                "center": model.body_pos[body_id].copy(),
                "half_size": model.geom_size[geom_adr].copy(),
            })
        return boxes

    def _centers_from_obs_next(self, info, batch_size):
        if not info or "obs_next" not in info:
            return None
        obs_next = np.asarray(info["obs_next"])
        if obs_next.ndim != 2 or obs_next.shape[1] < self.dynamic_obs_dim:
            return None
        obstacle_pos = obs_next[:, -self.dynamic_obs_dim : -self.dynamic_obs_dim + 6]
        if obstacle_pos.shape[0] != batch_size:
            return None
        return obstacle_pos.reshape(batch_size, 2, 3)

    def _aabb_distance(self, points, centers, half_size):
        delta = np.maximum(np.abs(points - centers) - half_size, 0.0)
        return np.linalg.norm(delta, axis=-1)

    def __call__(self, achieved_goal, desired_goal, info):
        reward = self.base_reward_func(achieved_goal, desired_goal, info)
        if self.weight <= 0 or not self.boxes:
            return reward
        achieved_goal = np.asarray(achieved_goal)
        single = achieved_goal.ndim == 1
        points = achieved_goal.reshape(1, -1) if single else achieved_goal
        dynamic_centers = self._centers_from_obs_next(info, points.shape[0])
        penalty = np.zeros(points.shape[0], dtype=np.float32)
        for idx, box in enumerate(self.boxes):
            if dynamic_centers is not None and idx < dynamic_centers.shape[1]:
                centers = dynamic_centers[:, idx, :]
            else:
                centers = np.broadcast_to(box["center"], points.shape)
            distances = self._aabb_distance(points, centers, box["half_size"])
            penalty += np.clip((self.safe_distance - distances) / self.safe_distance, 0.0, 1.0)
        shaped = reward - self.weight * penalty
        return shaped[0] if single else shaped
```

**接口约束（改奖励时必须遵守）**：
- 输入 `achieved_goal`, `desired_goal` 可能是单条 `(dim,)` 或 batch `(batch, dim)`
- `info` 字典可能包含 `"obs_next"`（用于读取动态障碍物位置）；也可能为 `None`
- 返回值：单条时标量，batch 时 `(batch,)` 的 numpy array
- 不要修改 `base_reward_func`（这是 Fetch 环境原始稀疏奖励）

**Prompt 模板（改奖励函数）**：
> 以下是 `rl_modules/obstacle_reward.py` 的完整代码。请保留 `ObstacleAwareReward` 类结构和 `__call__` 的输入输出语义，但修改奖励计算逻辑：
> 1. 保留原有障碍物软惩罚；
> 2. 额外增加一项 **到目标的稠密距离奖励**：`dense = -np.linalg.norm(achieved_goal - desired_goal, axis=-1)`；
> 3. 使用 `--dense-reward-weight`（已在 args 中注册，值存在 `self.dense_reward_weight`）控制稠密奖励权重；
> 4. 最终奖励 = 原始稀疏奖励 - 障碍物惩罚 + dense_weight * dense。
> 请处理 single/batch 两种情况，保持 numpy 运算。

---

### 1.6 `rl_modules/dynamic_obs_wrapper.py` —— 动态障碍物环境包装

**职责**：`gym.Wrapper` 子类。每个 `step()` 前按正弦函数更新 obstacle1/obstacle2 的 body 位置，并把障碍物状态（位置+速度，12维）拼接到 observation。

**关键方法**（修改障碍物运动或 obs 拼接时复制给 GPT）：

```python
class DynamicObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.episode_step = 0
        self._last_obstacle_pos = None
        self._last_obstacle_vel = np.zeros(6, dtype=np.float32)
        self._init_obstacle_data()
        self._extend_observation_space()

    def _extend_observation_space(self):
        base_space = self.env.observation_space
        obs_space = base_space.spaces["observation"]
        dtype = obs_space.dtype
        low = np.concatenate([obs_space.low, np.full(12, -np.inf, dtype=dtype)])
        high = np.concatenate([obs_space.high, np.full(12, np.inf, dtype=dtype)])
        spaces = dict(base_space.spaces)
        spaces["observation"] = gym.spaces.Box(low=low, high=high, dtype=dtype)
        self.observation_space = gym.spaces.Dict(spaces)

    def _update_obstacles(self):
        unwrapped = self.env.unwrapped
        dt = unwrapped.dt if hasattr(unwrapped, "dt") else unwrapped.sim.dt
        t = self.episode_step * dt
        y1 = self.orig1[1] + 0.14 * np.sin(2 * np.pi * 0.25 * t)   # obs1 沿 y
        x2 = self.orig2[0] + 0.16 * np.sin(2 * np.pi * 0.20 * t)   # obs2 沿 x
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
            self._last_obstacle_vel = (current_pos - self._last_obstacle_pos) / max(dt, 1e-6)
        self._last_obstacle_pos = current_pos

    def _augment_observation(self, observation):
        observation = dict(observation)
        obstacle_state = np.concatenate([self._obstacle_positions(), self._last_obstacle_vel])
        observation["observation"] = np.concatenate(
            [observation["observation"], obstacle_state.astype(observation["observation"].dtype)]
        )
        return observation

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
```

**Prompt 模板（改障碍物运动）**：
> 以下是 `rl_modules/dynamic_obs_wrapper.py` 的完整代码。请将障碍物的运动改为 **Lissajous 曲线**（x = A sin(at), y = B sin(bt)），让两个障碍物都在 x-y 平面做椭圆运动，并保证它们不会相撞。observation 拼接维度可能需要调整，如果速度 still 拼接 6 维，请保持 `_extend_observation_space` 的维度计算正确。

---

### 1.7 `her_modules/her.py` —— HER 采样器

**职责**：从 episode buffer 中采样 transition，并以 `future_p` 概率将 desired_goal 替换为未来 achieved_goal，重新计算奖励。

**完整代码**（修改 HER 策略时复制）：

```python
import numpy as np

class her_sampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        # 随机选 episode 和 timestep
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        # HER 替换
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        # 重新计算奖励
        reward_info = {'obs_next': transitions.get('obs_next')}
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], reward_info), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}
        return transitions
```

**关键约束**：
- `reward_func(ag_next, g, info)` 必须同时支持单条和 batch 输入
- `episode_batch` 包含 keys：`obs, ag, g, actions, obs_next, ag_next`
- `transitions['r']` 最终被 reshape 为 `(batch_size, 1)`

**Prompt 模板（改 HER 策略）**：
> 以下是 `her_modules/her.py` 的完整代码。请扩展 `her_sampler`，使其支持 `replay_strategy='final'`（用 episode 最后一步的 achieved_goal 替换）和 `'episode'`（从 episode 任意时刻随机选 achieved_goal 替换）。保持 `sample_her_transitions` 的接口和返回值不变。

---

### 1.8 `rl_modules/replay_buffer.py` —— 经验回放缓冲区

**职责**：按 episode 存储数据，线程安全。

**关键接口**：

```python
class replay_buffer:
    def __init__(self, env_params, buffer_size, sample_func):
        self.T = env_params['max_timesteps']
        self.size = buffer_size // self.T
        self.current_size = 0
        self.sample_func = sample_func
        self.buffers = {
            'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
            'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
            'g': np.empty([self.size, self.T, self.env_params['goal']]),
            'actions': np.empty([self.size, self.T, self.env_params['action']]),
        }

    def store_episode(self, episode_batch):
        """episode_batch: [mb_obs, mb_ag, mb_g, mb_actions]"""

    def sample(self, batch_size):
        """调用 self.sample_func(temp_buffers, batch_size) 返回 transitions dict。"""
```

**修改场景**：如需实现 PER，需要修改 `store_episode` 维护优先级，以及 `sample` 的采样逻辑。把完整代码发给 GPT 即可。

---

### 1.9 `mpi_utils/mpi_utils.py` —— MPI 同步

**关键函数**：

```python
def sync_networks(network):
    """将 rank 0 的网络参数广播到所有 MPI 进程。"""

def sync_grads(network):
    """Allreduce 聚合所有进程的梯度，平均后写回。"""

def discounted_return(rewards, gamma, reward_offset=True):
    """计算折扣/无折扣回报。默认把奖励 +1 再算（Fetch 稀疏奖励为 -1/0）。"""
```

---

### 1.10 `mpi_utils/normalizer.py` —— 运行归一化

**职责**：维护 observation 和 goal 的 running mean/std，支持 MPI 跨进程同步。

**关键方法**：

```python
class normalizer:
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
    def update(self, v):
    def recompute_stats(self):
    def normalize(self, v, clip_range=None):
```

---

## 2. 按修改目标分类：该复制哪些文件、配什么 Prompt

### 场景 A：修改奖励函数（最常用）

**需要复制的文件**：
1. `rl_modules/obstacle_reward.py`（完整代码）
2. `arguments.py`（仅新增参数那几行）
3. `rl_modules/ddpg_agent.py`（仅 `__init__` 中 `ObstacleAwareReward` 初始化那几行）

**Prompt 模板**：
```
我正在一个 HER+DDPG Fetch 机器人项目上，使用稀疏奖励环境。下面是三个相关文件的关键代码。

[粘贴 obstacle_reward.py 完整代码]
[粘贴 arguments.py 中 parser.add_argument 相关段落]
[粘贴 ddpg_agent.py 中 ObstacleAwareReward 初始化段落]

请帮我修改奖励函数，要求：
1. ...（你的具体需求）
2. 保持 __call__ 接口不变，支持 single 和 batch 输入。
3. 如果需要在 arguments.py 加参数，请给出新增代码。
4. 如果需要在 ddpg_agent.py 中修改初始化传参，请给出修改后代码。
```

### 场景 B：修改网络结构

**需要复制的文件**：
1. `rl_modules/models.py`（完整代码）

**Prompt 模板**：
```
以下是 models.py 的完整代码。请修改 Actor 和 Critic 网络结构，要求：
1. ...
2. 保持 forward 输入输出维度不变。
3. 给出修改后的完整代码，可直接替换原文件。
```

### 场景 C：修改训练算法（如加入 TD3、SAC 等）

**需要复制的文件**：
1. `rl_modules/ddpg_agent.py`（完整代码）
2. `rl_modules/models.py`（完整代码，因为可能要新增网络）

**Prompt 模板**：
```
以下是 ddpg_agent.py 的完整代码，以及 models.py 的完整代码。
请将训练器改造为 TD3，要求：
1. ...
2. 保持 buffer 接口和 her_sampler 接口不变。
3. 给出两个文件的完整修改后代码。
```

### 场景 D：修改 HER 采样策略

**需要复制的文件**：
1. `her_modules/her.py`（完整代码）

### 场景 E：修改动态障碍物行为

**需要复制的文件**：
1. `rl_modules/dynamic_obs_wrapper.py`（完整代码）

### 场景 F：新增超参数

**需要复制的文件**：
1. `arguments.py`（完整代码）
2. 使用该参数的下游文件（如 `ddpg_agent.py` 中的初始化段）

---

## 3. 常见修改对照表

| 你想做什么 | 核心文件 | 是否需要改 arguments.py |
|-----------|---------|------------------------|
| 改奖励（障碍物惩罚、稠密奖励、碰撞惩罚） | `rl_modules/obstacle_reward.py` | 是（加 weight 参数） |
| 改网络层数/宽度/激活函数 | `rl_modules/models.py` | 否 |
| 改探索噪声（OU/衰减/自适应） | `rl_modules/ddpg_agent.py` (`_select_actions`) | 是（加参数） |
| 改 HER 目标选择策略 | `her_modules/her.py` | 是（`--replay-strategy` 扩展 choices） |
| 改训练算法（TD3/SAC/DDPG+PER） | `rl_modules/ddpg_agent.py` + `models.py` | 可能（如 alpha/lr 等） |
| 改障碍物运动方式 | `rl_modules/dynamic_obs_wrapper.py` | 可能 |
| 改环境物理布局 | `xml_envs/<config>/fetch/*.xml` | 否 |
| 改归一化 clip 或统计方式 | `mpi_utils/normalizer.py` | 是 |
| 新增评估指标 | `rl_modules/ddpg_agent.py` (`_eval_agent`) | 否 |
| 新增日志字段到 W&B | `rl_modules/ddpg_agent.py` (`learn`) | 否 |

---

## 4. 提交给 GPT 时的最佳实践

1. **尽量给完整文件代码**：GPT 处理 500 行以内的 Python 文件很轻松。给片段容易导致上下文缺失、接口不匹配。
2. **明确约束**：
   - "保持输入输出维度不变"
   - "不要改 buffer 接口"
   - "保持 numpy 运算，不要用 torch"
3. **分步修改**：如果改动涉及多个文件，先让 GPT 改核心算法文件，确认无误后再改参数文件和环境包装器。
4. **本地验证**：GPT 改完后，先用 `python -m py_compile <file>.py` 做语法检查，再运行 `python test_env.py --env-name FetchReach-v4 --env-config free --steps 5` 做集成测试。

---

*本文档基于项目当前代码结构生成。如后续新增模块，请同步在对应章节补充代码片段和 Prompt 模板。*
