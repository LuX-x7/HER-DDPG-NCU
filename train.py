import numpy as np
import gymnasium as gym
import gymnasium_robotics
import os
from arguments import get_args
from env_utils import env_id_for, infer_env_config, patch_fetch_xml_paths, resolve_xml_dir
from mpi4py import MPI
from rl_modules.ddpg_agent import ddpg_agent
import random
import torch

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""

def convert_env_name(name):
    """Auto-upgrade old -v1 robotics envs to -v4 for native MuJoCo support."""
    if name.endswith('-v1') and 'Fetch' in name:
        return name[:-3] + '-v4'
    return name

def get_env_params(env):
    obs, _ = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = env.spec.max_episode_steps
    return params

def launch(args):
    # register robotics envs
    gym.register_envs(gymnasium_robotics)
    # auto-upgrade env name if needed
    args.env_name = convert_env_name(args.env_name)

    xml_dir = resolve_xml_dir(args.xml_dir, args.env_config)
    patch_fetch_xml_paths(xml_dir)
    args.xml_dir = str(xml_dir)
    args.env_config = infer_env_config(xml_dir)
    args.env_id = env_id_for(args.env_name, args.env_config)

    args.run_name = f"{args.env_id}-ddpg-{args.seed}"

    # bind each MPI rank to a dedicated GPU for multi-GPU training
    rank = MPI.COMM_WORLD.Get_rank()
    if args.cuda and torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())

    if rank == 0:
        try:
            import wandb
            if args.wandb_mode != 'disabled':
                wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=args.run_name,
                    group=args.env_config,
                    tags=[args.env_config, args.env_name],
                    config=vars(args),
                    mode=args.wandb_mode,
                )
        except Exception as e:
            print(f"wandb init failed: {e}")

    # create the ddpg_agent (only rank 0 needs render_mode for gif capture)
    render_mode = 'rgb_array' if rank == 0 else None
    env = gym.make(args.env_name, render_mode=render_mode)
    # wrap before reading env params so dynamic obstacle state is included in obs dim
    if args.env_config == '2Dynamicobs':
        from rl_modules.dynamic_obs_wrapper import DynamicObsWrapper
        env = DynamicObsWrapper(env)
    # set random seeds for reproduce
    rank_seed = args.seed + rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if args.cuda:
        torch.cuda.manual_seed(rank_seed)
    # get the environment parameters
    env_params = get_env_params(env)

    # create the ddpg agent to interact with the environment 
    ddpg_trainer = ddpg_agent(args, env, env_params)
    ddpg_trainer.learn()

    # ensure wandb is properly closed
    if MPI.COMM_WORLD.Get_rank() == 0:
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    if not os.environ.get('DISPLAY'):
        os.environ['MUJOCO_GL'] = 'egl'
    # get the params
    args = get_args()
    launch(args)
