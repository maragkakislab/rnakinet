#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.ndimage import gaussian_filter1d


def read_experiment_metric(csv_paths: list[Path], metric_name: str) -> dict[str, np.ndarray]:
    experiment_to_values: dict[str, list[np.ndarray]] = {}
    for csv_path in csv_paths:
        metrics_df = pd.read_csv(csv_path, usecols=["experiment", metric_name])
        experiment_name = metrics_df["experiment"].iloc[0]
        experiment_to_values.setdefault(experiment_name, []).append(metrics_df[metric_name].to_numpy(dtype=float))
    return {exp: np.concatenate(chunks) for exp, chunks in experiment_to_values.items()}


def plot_distributions(
    experiment_to_values: dict[str, np.ndarray],
    metric_name: str,
    title: str,
    out_path: Path,
    normalize: bool,
    log10: bool,
    bins: int,
) -> None:
    plot_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "experiment": experiment_name,
                    "value": (np.log10(values[values > 0]) if log10 else values),
                }
            )
            for experiment_name, values in experiment_to_values.items()
        ],
        ignore_index=True,
    )
    x_min, x_max = float(plot_df["value"].min()), float(plot_df["value"].max())

    plt.figure(figsize=(10, 6))
    plt.rcParams['font.size'] = 14
    
    bin_edges = np.linspace(x_min, x_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for experiment_name, values in experiment_to_values.items():
        plot_values = np.log10(values[values > 0]) if log10 else values
        counts, _ = np.histogram(plot_values, bins=bin_edges)
        y = counts / counts.sum() if normalize else counts
        y_smooth = gaussian_filter1d(y, sigma=1.5)  # adjust sigma
        plt.plot(bin_centers, y_smooth, label=experiment_name)

    plt.xlabel(f"log10({metric_name})" if log10 else metric_name, fontsize=15)
    plt.ylabel("Proportion of reads" if normalize else "Read count", fontsize=15)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=13)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--title", required=True)
    p.add_argument("--metric", required=True)
    p.add_argument("--input-csvs", nargs="+", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--log10", action="store_true")
    p.add_argument("--bins", type=int, default=512)
    args = p.parse_args()

    experiment_to_values = read_experiment_metric(args.input_csvs, args.metric)
    plot_distributions(experiment_to_values, args.metric, args.title, args.output, args.normalize, args.log10, args.bins)


if __name__ == "__main__":
    main()