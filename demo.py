import os
import torch
from rl_modules.models import actor
from arguments import get_args
import gymnasium as gym
import gymnasium_robotics
import numpy as np

# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

def convert_env_name(name):
    """Auto-upgrade old -v1 robotics envs to -v4 for native MuJoCo support."""
    if name.endswith('-v1') and 'Fetch' in name:
        return name[:-3] + '-v4'
    return name

if __name__ == '__main__':
    args = get_args()
    # register robotics envs
    gym.register_envs(gymnasium_robotics)
    args.env_name = convert_env_name(args.env_name)
    args.env_id = args.env_name
    args.run_name = f"{args.env_id}-ddpg-{args.seed}"
    # load the model param
    model_path = os.path.join(args.save_dir, args.env_name, args.run_name, 'model.pt')
    o_mean, o_std, g_mean, g_std, model = torch.load(model_path, map_location=lambda storage, loc: storage)
    # create the environment
    # auto-select render mode for headless environments
    render_mode = 'human' if os.environ.get('DISPLAY') else 'rgb_array'
    env = gym.make(args.env_name, render_mode=render_mode)
    # get the env param
    observation, _ = env.reset()
    # get the environment params
    env_params = {'obs': observation['observation'].shape[0], 
                  'goal': observation['desired_goal'].shape[0], 
                  'action': env.action_space.shape[0], 
                  'action_max': env.action_space.high[0],
                  }
    # create the actor network
    actor_network = actor(env_params)
    actor_network.load_state_dict(model)
    actor_network.eval()
    for i in range(args.demo_length):
        observation, _ = env.reset()
        # start to do the demo
        obs = observation['observation']
        g = observation['desired_goal']
        for t in range(env.spec.max_episode_steps):
            env.render()
            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            with torch.no_grad():
                pi = actor_network(inputs)
            action = pi.detach().numpy().squeeze()
            # put actions into the environment
            observation_new, reward, terminated, truncated, info = env.step(action)
            obs = observation_new['observation']
        print('the episode is: {}, is success: {}'.format(i, info['is_success']))
