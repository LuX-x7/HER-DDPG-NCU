#!/usr/bin/env python3
import argparse
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

from env_utils import ENV_CONFIGS, FETCH_ENVS


METRICS = [
    "Test/success_rate",
    "Test/final_distance",
    "Test/discounted_return",
    "Test/undiscounted_return",
]

TAG_SUFFIXES = {
    "free": "",
    "2Staticobs": "-2staticobs",
    "2Dynamicobs": "-dynobs",
}

COLORS = {
    "free": "C0",
    "2Staticobs": "C2",
    "2Dynamicobs": "C3",
}

LABELS = {
    "free": "Free",
    "2Staticobs": "2 Static Obs",
    "2Dynamicobs": "2 Dynamic Obs",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Export HER training curves from W&B to figures/.")
    parser.add_argument("--entity", type=str, default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--project", type=str, default=os.environ.get("WANDB_PROJECT", "her"))
    parser.add_argument(
        "--env-config",
        type=str,
        default="all",
        choices=["all", *ENV_CONFIGS],
        help="Which XML environment set to plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: figures/<timestamp>_<env-config>.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Maximum W&B history samples per run.",
    )
    return parser.parse_args()


def normalize_env_id(run):
    run_name = run.name
    env_id = str(run.config.get("env_id", run_name.split("-ddpg-")[0]))
    if env_id.endswith("-obs"):
        return env_id[:-4], "2Staticobs"
    for env_config, suffix in TAG_SUFFIXES.items():
        if suffix and env_id.endswith(suffix):
            return env_id[: -len(suffix)], env_config
    return env_id, "free"


def default_output_dir(env_config):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "all_envs" if env_config == "all" else env_config
    return os.path.join("figures", f"{stamp}_{suffix}")


def main():
    args = parse_args()
    output_dir = args.output_dir or default_output_dir(args.env_config)

    api = wandb.Api()
    project_path = f"{args.entity}/{args.project}" if args.entity else args.project
    runs = api.runs(project_path)

    env_configs = list(ENV_CONFIGS) if args.env_config == "all" else [args.env_config]
    data = {env: {env_config: {} for env_config in env_configs} for env in FETCH_ENVS}

    for run in runs:
        base_env, env_config = normalize_env_id(run)
        if env_config not in env_configs or base_env not in data:
            continue

        hist = run.history(samples=args.samples, keys=["epoch"] + METRICS)
        if hist.empty:
            continue

        for metric in METRICS:
            data[base_env][env_config].setdefault(metric, [])
            sub = hist[["epoch", metric]].dropna()
            data[base_env][env_config][metric].extend(zip(sub["epoch"].tolist(), sub[metric].tolist()))

    os.makedirs(output_dir, exist_ok=True)

    for metric in METRICS:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"{metric} (W&B)", fontsize=16)
        axes = axes.flatten()

        for ax, env in zip(axes, FETCH_ENVS):
            has_points = False
            for env_config in env_configs:
                points = data[env][env_config].get(metric, [])
                if not points:
                    continue

                points = sorted(points, key=lambda item: item[0])
                epochs, values = zip(*points)
                ax.plot(
                    epochs,
                    values,
                    marker="o",
                    markersize=3,
                    label=LABELS[env_config],
                    color=COLORS[env_config],
                )
                has_points = True

            ax.set_title(env)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.split("/")[-1])
            if has_points:
                ax.legend()
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fname = os.path.join(output_dir, f"{metric.replace('/', '_')}.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved: {fname}")

    print(f"All plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
