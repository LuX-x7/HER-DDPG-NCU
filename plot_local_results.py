#!/usr/bin/env python3
"""Plot training curves from local CSV files (no wandb required)."""
import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env_utils import ENV_CONFIGS, FETCH_ENVS


METRICS = [
    "Test/success_rate",
    "Test/final_distance",
    "Test/discounted_return",
    "Test/undiscounted_return",
]

COLORS = {
    "free": "#2ca02c",
    "2Staticobs": "C2",
    "2Dynamicobs": "C3",
}

LABELS = {
    "free": "Free",
    "2Staticobs": "2 Static Obs",
    "2Dynamicobs": "2 Dynamic Obs",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot HER training curves from local CSV files.")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="saved_models",
        help="Directory containing saved_models/ with metrics.csv files",
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default="all",
        choices=["all", *ENV_CONFIGS],
        help="Which environment config to plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: figures/<timestamp>_<env-config>_local",
    )
    return parser.parse_args()


def find_metrics_csv(base_dir, env_config):
    """Find all metrics.csv under base_dir/env_config/."""
    base = Path(base_dir)
    return sorted(base.glob(f"{env_config}/*/*/metrics.csv"))


def parse_run_name_from_path(csv_path):
    """Extract env_name and env_config from path like saved_models/2Staticobs/FetchReach-v4/FetchReach-v4-2staticobs-ddpg-123/metrics.csv"""
    parts = csv_path.parts
    # Find env_config index
    for env_config in ENV_CONFIGS:
        if env_config in parts:
            idx = parts.index(env_config)
            if idx + 2 < len(parts):
                env_name = parts[idx + 1]
                run_name = parts[idx + 2]
                return env_name, run_name
    return None, None


def read_csv(csv_path):
    """Read metrics.csv and return list of dicts."""
    data = []
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            try:
                row = {
                    "epoch": int(parts[0]),
                    "Test/success_rate": float(parts[1]),
                    "Test/final_distance": float(parts[2]),
                    "Test/discounted_return": float(parts[3]),
                    "Test/undiscounted_return": float(parts[4]),
                }
                data.append(row)
            except ValueError:
                continue
    return data


def default_output_dir(env_config):
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "all_envs" if env_config == "all" else env_config
    return os.path.join("figures", f"{stamp}_{suffix}_local")


def main():
    args = parse_args()
    output_dir = args.output_dir or default_output_dir(args.env_config)
    os.makedirs(output_dir, exist_ok=True)

    env_configs = list(ENV_CONFIGS) if args.env_config == "all" else [args.env_config]
    data = {env: {ec: [] for ec in env_configs} for env in FETCH_ENVS}

    for ec in env_configs:
        for csv_path in find_metrics_csv(args.save_dir, ec):
            env_name, run_name = parse_run_name_from_path(csv_path)
            if env_name not in data:
                continue
            rows = read_csv(csv_path)
            data[env_name][ec].extend(rows)

    for metric in METRICS:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"{metric} (Local CSV)", fontsize=16)
        axes = axes.flatten()

        for ax, env in zip(axes, FETCH_ENVS):
            has_points = False
            for ec in env_configs:
                rows = data[env][ec]
                if not rows:
                    continue

                rows_sorted = sorted(rows, key=lambda r: r["epoch"])
                epochs = [r["epoch"] for r in rows_sorted]
                values = [r[metric] for r in rows_sorted]
                ax.plot(
                    epochs,
                    values,
                    marker="o",
                    markersize=3,
                    label=LABELS[ec],
                    color=COLORS[ec],
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
