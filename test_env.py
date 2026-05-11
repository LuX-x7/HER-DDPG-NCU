import argparse
import os

if not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

import gymnasium as gym
import gymnasium_robotics

from env_utils import patch_fetch_xml_paths, resolve_xml_dir


def test_env(env_name, xml_dir=None, env_config="free", steps=50, save_dir="test_gifs", seed=123):
    gym.register_envs(gymnasium_robotics)

    xml_dir = resolve_xml_dir(xml_dir, env_config)
    patched = patch_fetch_xml_paths(xml_dir)
    for task, path in patched.items():
        print(f"[XML] {task}: {path}")

    env = gym.make(env_name, render_mode="rgb_array")
    if env_config == "2Dynamicobs":
        from rl_modules.dynamic_obs_wrapper import DynamicObsWrapper

        env = DynamicObsWrapper(env)

    print(f"\n[Env] {env_name}")
    print(f"[Env] fullpath: {env.unwrapped.fullpath}")

    obs, info = env.reset(seed=seed)
    imgs = []

    for _ in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        img = env.render()
        if img is not None:
            imgs.append(img)

        if terminated or truncated:
            break

    env.close()

    out_dir = os.path.join(save_dir, env_config, env_name)
    os.makedirs(out_dir, exist_ok=True)

    if imgs:
        gif_path = os.path.join(out_dir, f"test_seed{seed}.gif")
        try:
            import imageio

            imageio.mimsave(gif_path, imgs, duration=0.04)
            print(f"[GIF] Saved to: {gif_path}")
        except ImportError:
            print("[Warning] imageio not installed, skipping GIF save")

        key_indices = [0, len(imgs) // 2, len(imgs) - 1]
        for idx in key_indices:
            png_path = os.path.join(out_dir, f"test_seed{seed}_frame{idx:02d}.png")
            try:
                from PIL import Image

                Image.fromarray(imgs[idx]).save(png_path)
                print(f"[PNG] Saved to: {png_path}")
            except ImportError:
                print("[Warning] PIL not installed, skipping PNG save")
                break

    print("[Done] Test finished.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="FetchPush-v4")
    parser.add_argument("--env-config", type=str, default="free", choices=["free", "2Staticobs", "2Dynamicobs"])
    parser.add_argument("--xml-dir", type=str, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--save-dir", type=str, default="test_gifs")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    test_env(
        args.env_name,
        xml_dir=args.xml_dir,
        env_config=args.env_config,
        steps=args.steps,
        save_dir=args.save_dir,
        seed=args.seed,
    )
