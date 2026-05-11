import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from env_utils import run_output_dir
from mpi_utils.mpi_utils import sync_networks, sync_grads, discounted_return
from rl_modules.replay_buffer import replay_buffer
from rl_modules.models import actor, critic
from rl_modules.obstacle_reward import ObstacleAwareReward
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import imageio
except ImportError:
    imageio = None

"""
ddpg with HER (MPI-version)

"""
class ddpg_agent:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params
        # create the network
        self.actor_network = actor(env_params)
        self.critic_network = critic(env_params)
        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        # build up the target network
        self.actor_target_network = actor(env_params)
        self.critic_target_network = critic(env_params)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # if use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)



        # reward_func = ObstacleAwareReward(
        #     self.env,
        #     self.env.unwrapped.compute_reward,
        #     weight=self.args.obstacle_reward_weight,
        #     safe_distance=self.args.obstacle_safe_distance,
        # )
        # # her sampler
        # self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, reward_func)
        
        self.reward_func = ObstacleAwareReward(
            self.env,
            self.env.unwrapped.compute_reward,
            weight=self.args.obstacle_reward_weight,
            safe_distance=self.args.obstacle_safe_distance,
            dense_reward_weight=getattr(self.args, "dense_reward_weight", 0.0),
            collision_penalty=getattr(self.args, "collision_penalty", 0.0),
            collision_distance=getattr(self.args, "collision_distance", 0.02),
        )

        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.reward_func)



        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)
        # create the dict for store the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.model_path = os.path.join(
                self.args.save_dir,
                self.args.env_config,
                self.args.env_name,
                self.args.run_name,
            )
            os.makedirs(self.model_path, exist_ok=True)
            self.figure_path = run_output_dir(
                self.args.figures_dir,
                self.args.env_config,
                self.args.env_name,
                self.args.seed,
            )
            os.makedirs(self.figure_path, exist_ok=True)
            # local CSV for metrics (no wandb dependency)
            self.metrics_csv = os.path.join(self.model_path, 'metrics.csv')
            if not os.path.exists(self.metrics_csv):
                with open(self.metrics_csv, 'w') as f:
                    f.write('epoch,Test/success_rate,Test/final_distance,Test/discounted_return,Test/undiscounted_return,timestamp\n')

    def learn(self):
        """
        train the network
        """
        epoch_iter = range(self.args.n_epochs)
        if tqdm is not None and MPI.COMM_WORLD.Get_rank() == 0:
            epoch_iter = tqdm(epoch_iter, desc='Training')

        for epoch in epoch_iter:
            cycle_iter = range(self.args.n_cycles)
            if tqdm is not None and MPI.COMM_WORLD.Get_rank() == 0:
                cycle_iter = tqdm(cycle_iter, desc=f'Epoch {epoch}', leave=False)

            for _ in cycle_iter:
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                for _ in range(self.args.num_rollouts_per_mpi):
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                    # reset the environment
                    observation, _ = self.env.reset()
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            input_tensor = self._preproc_inputs(obs, g)
                            pi = self.actor_network(input_tensor)
                            action = self._select_actions(pi)
                        # feed the actions into the environment
                        observation_new, reward, terminated, truncated, info = self.env.step(action)
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']
                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())
                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                # store the episodes
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
                for _ in range(self.args.n_batches):
                    # train the network
                    self._update_network()
                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)

            # start to do the evaluation
            if self.args.eval:
                if MPI.COMM_WORLD.Get_rank() != 0:
                    results = self._eval_agent(make_gif=False, epoch=epoch)
                else:
                    make_gif = (epoch % self.args.save_gif_interval == 0) and (imageio is not None)
                    results = self._eval_agent(make_gif=make_gif, epoch=epoch)

                if MPI.COMM_WORLD.Get_rank() == 0:
                    results.update({'epoch': epoch})
                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb.log(results)
                    except Exception:
                        pass
                    print('[{}] epoch is: {}, eval success rate is: {:.3f}, final_distance is: {:.3f}'.format(
                        datetime.now(), epoch, results['Test/success_rate'], results['Test/final_distance']))
                    # write to local CSV
                    try:
                        with open(self.metrics_csv, 'a') as f:
                            f.write('{},{:.6f},{:.6f},{:.6f},{:.6f},{}\n'.format(
                                epoch,
                                results.get('Test/success_rate', 0),
                                results.get('Test/final_distance', 0),
                                results.get('Test/discounted_return', 0),
                                results.get('Test/undiscounted_return', 0),
                                datetime.now().isoformat(),
                            ))
                    except Exception:
                        pass
                    torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict()], 
                               self.model_path + '/model.pt')

        # end-of-training things
        # 所有 rank 都必须参与 _eval_agent，因为内部有 MPI allreduce
        make_gif = (MPI.COMM_WORLD.Get_rank() == 0) and (imageio is not None)
        results = self._eval_agent(make_gif=make_gif, epoch=self.args.n_epochs)
        if MPI.COMM_WORLD.Get_rank() == 0:
            torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict()], 
                       self.model_path + '/model.pt')
        self.env.close()

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs
    
    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs, 
                       'ag': mb_ag,
                       'g': mb_g, 
                       'actions': mb_actions, 
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32) 
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # the q loss
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # the actor loss
        actions_real = self.actor_network(inputs_norm_tensor)
        actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()

    # do the evaluation
    # def _eval_agent(self, make_gif=False, epoch=0):
    #     total_obs, total_g, total_ag, total_rewards, total_success_rate = [], [], [], [], []
    #     for i in range(self.args.n_test_rollouts):
    #         per_obs, per_g, per_ag, per_rewards, per_success_rate = [], [], [], [], []
    #         observation, _ = self.env.reset()
    #         obs = observation['observation']
    #         ag = observation['achieved_goal']
    #         g = observation['desired_goal']
    #         imgs = []
    #         for _ in range(self.env_params['max_timesteps']):
    #             with torch.no_grad():
    #                 input_tensor = self._preproc_inputs(obs, g)
    #                 pi = self.actor_network(input_tensor)
    #                 # convert the actions
    #                 actions = pi.detach().cpu().numpy().squeeze()
    #             observation_new, reward, terminated, truncated, info = self.env.step(actions)
    #             if make_gif:
    #                 img = self.env.render()
    #                 if img is not None:
    #                     imgs.append(img)
    #             per_obs.append(obs)
    #             per_g.append(g)
    #             per_ag.append(ag)
    #             per_rewards.append(reward)
    #             per_success_rate.append(info['is_success'])
    #             obs = observation_new['observation']
    #             ag = observation_new['achieved_goal']
    #             g = observation_new['desired_goal']
    #         total_obs.append(per_obs)
    #         total_g.append(per_g)
    #         total_ag.append(per_ag)
    #         total_rewards.append(per_rewards)
    #         total_success_rate.append(per_success_rate)

    #         if make_gif and len(imgs) > 0 and MPI.COMM_WORLD.Get_rank() == 0:
    #             gif_dir = self.figure_path / "policy_gifs"
    #             os.makedirs(gif_dir, exist_ok=True)
    #             gif_path = gif_dir / f"epoch{epoch}-test{i}.gif"
    #             imageio.mimsave(gif_path, imgs, duration=0.04)

    #     total_obs = np.array(total_obs)
    #     total_g = np.array(total_g)
    #     total_ag = np.array(total_ag)
    #     total_rewards = np.array(total_rewards)
    #     total_success_rate = np.array(total_success_rate)
    #     dis_return, undis_return = discounted_return(total_rewards, self.args.gamma)

    #     local_discounted_return = np.mean(dis_return)
    #     global_discounted_return = MPI.COMM_WORLD.allreduce(local_discounted_return, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()

    #     local_undiscounted_return = np.mean(undis_return)
    #     global_undiscounted_return = MPI.COMM_WORLD.allreduce(local_undiscounted_return, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()

    #     local_distances = np.mean(np.linalg.norm(total_ag[:, -1] - total_g[:, -1], axis=1))
    #     global_distances = MPI.COMM_WORLD.allreduce(local_distances, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()

    #     local_success_rate = np.mean(total_success_rate[:, -1])
    #     global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
    #     results = {'Test/final_distance': global_distances, 
    #                'Test/success_rate': global_success_rate,
    #                'Test/discounted_return': global_discounted_return,
    #                'Test/undiscounted_return': global_undiscounted_return}
    #     return results
    def _eval_agent(self, make_gif=False, epoch=0):
        total_obs = []
        total_g = []
        total_ag = []
        total_rewards = []
        total_success_rate = []
        total_min_obstacle_distances = []
        total_collision_flags = []

        for i in range(self.args.n_test_rollouts):
            per_obs = []
            per_g = []
            per_ag = []
            per_rewards = []
            per_success_rate = []
            per_min_obstacle_distances = []
            per_collision_flags = []

            observation, _ = self.env.reset()
            obs = observation['observation']
            ag = observation['achieved_goal']
            g = observation['desired_goal']

            imgs = []

            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
                    actions = pi.detach().cpu().numpy().squeeze()

                observation_new, env_reward, terminated, truncated, info = self.env.step(actions)

                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal']
                g_new = observation_new['desired_goal']

                # 用增强后的安全奖励重新计算 eval reward
                reward_info = {
                    'obs_next': obs_new.reshape(1, -1)
                }
                shaped_reward = self.reward_func(ag_new, g, reward_info)

                # 计算安全指标
                safety_info = self.reward_func.safety_metrics(ag_new, reward_info)

                if make_gif:
                    img = self.env.render()
                    if img is not None:
                        imgs.append(img)

                per_obs.append(obs_new)
                per_g.append(g)
                per_ag.append(ag_new)
                per_rewards.append(float(shaped_reward))
                per_success_rate.append(info.get('is_success', 0.0))
                per_min_obstacle_distances.append(safety_info["min_distance"])
                per_collision_flags.append(safety_info["collision"])

                obs = obs_new
                ag = ag_new
                g = g_new

            total_obs.append(per_obs)
            total_g.append(per_g)
            total_ag.append(per_ag)
            total_rewards.append(per_rewards)
            total_success_rate.append(per_success_rate)
            total_min_obstacle_distances.append(per_min_obstacle_distances)
            total_collision_flags.append(per_collision_flags)

            if make_gif and len(imgs) > 0 and MPI.COMM_WORLD.Get_rank() == 0:
                gif_dir = self.figure_path / "policy_gifs"
                os.makedirs(gif_dir, exist_ok=True)
                gif_path = gif_dir / f"epoch{epoch}-test{i}.gif"
                imageio.mimsave(gif_path, imgs, duration=0.04)

        total_obs = np.array(total_obs)
        total_g = np.array(total_g)
        total_ag = np.array(total_ag)
        total_rewards = np.array(total_rewards)
        total_success_rate = np.array(total_success_rate)
        total_min_obstacle_distances = np.array(total_min_obstacle_distances)
        total_collision_flags = np.array(total_collision_flags)

        # 由于现在 eval reward 已经不是单纯 Fetch sparse reward，
        # 这里建议 reward_offset=False，否则会把每步奖励额外 +1。
        dis_return, undis_return = discounted_return(
            total_rewards,
            self.args.gamma,
            reward_offset=False,
        )

        local_discounted_return = np.mean(dis_return)
        global_discounted_return = MPI.COMM_WORLD.allreduce(
            local_discounted_return,
            op=MPI.SUM,
        ) / MPI.COMM_WORLD.Get_size()

        local_undiscounted_return = np.mean(undis_return)
        global_undiscounted_return = MPI.COMM_WORLD.allreduce(
            local_undiscounted_return,
            op=MPI.SUM,
        ) / MPI.COMM_WORLD.Get_size()

        local_distances = np.mean(np.linalg.norm(total_ag[:, -1] - total_g[:, -1], axis=1))
        global_distances = MPI.COMM_WORLD.allreduce(
            local_distances,
            op=MPI.SUM,
        ) / MPI.COMM_WORLD.Get_size()

        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(
            local_success_rate,
            op=MPI.SUM,
        ) / MPI.COMM_WORLD.Get_size()

        # 最小障碍物距离：每条轨迹取最小距离，再对所有测试轨迹取平均
        if np.isfinite(total_min_obstacle_distances).any():
            local_min_obstacle_distance = np.mean(np.min(total_min_obstacle_distances, axis=1))
        else:
            # 无障碍环境下没有 obstacle box，记为 -1，避免 inf 写入日志
            local_min_obstacle_distance = -1.0

        global_min_obstacle_distance = MPI.COMM_WORLD.allreduce(
            local_min_obstacle_distance,
            op=MPI.SUM,
        ) / MPI.COMM_WORLD.Get_size()

        # 碰撞率：发生过至少一次碰撞的 episode 比例
        local_collision_rate = np.mean(np.any(total_collision_flags > 0, axis=1))
        global_collision_rate = MPI.COMM_WORLD.allreduce(
            local_collision_rate,
            op=MPI.SUM,
        ) / MPI.COMM_WORLD.Get_size()

        # 平均每条轨迹的碰撞步数
        local_collision_steps = np.mean(np.sum(total_collision_flags > 0, axis=1))
        global_collision_steps = MPI.COMM_WORLD.allreduce(
            local_collision_steps,
            op=MPI.SUM,
        ) / MPI.COMM_WORLD.Get_size()

        results = {
            'Test/final_distance': global_distances,
            'Test/success_rate': global_success_rate,
            'Test/discounted_return': global_discounted_return,
            'Test/undiscounted_return': global_undiscounted_return,
            'Test/min_obstacle_distance': global_min_obstacle_distance,
            'Test/collision_rate': global_collision_rate,
            'Test/collision_steps': global_collision_steps,
        }

        return results