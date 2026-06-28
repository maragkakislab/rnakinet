import argparse
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt


def read_values(outputs_dir: Path, models: List[str], experiments: List[str]):
    pred_frac_matrix = np.full((len(experiments), len(models)), np.nan, dtype=float)

    for exp_idx, experiment in enumerate(experiments):
        for model_idx, model in enumerate(models):
            log_path = outputs_dir / "predictions" / model / experiment / "log.txt"
            lines = log_path.read_text().splitlines()
            value_line = next(line for line in lines if line.startswith("pred_frac_modified:"))
            pred_frac = float(value_line.split(":")[1].strip())
            pred_frac_matrix[exp_idx, model_idx] = pred_frac

    return pred_frac_matrix


def plot_grouped_bars(values: np.ndarray, models: List[str], experiments: List[str],
                      out_path: Path, title: str, ylabel: str, colors=None):
    n_experiments, n_models = values.shape

    experiment_positions = np.arange(n_experiments)
    group_width = 0.8
    bar_width = group_width / max(n_models, 1)
    model_offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * bar_width

    fig, ax = plt.subplots(figsize=(max(10, n_experiments * 0.6), 5))

    use_model_colors = (colors is not None) and (len(colors) == n_models)
    for model_idx, model_name in enumerate(models):
        bar_positions = experiment_positions + model_offsets[model_idx]
        bar_heights = values[:, model_idx]
        bar_color = colors[model_idx] if use_model_colors else None
        ax.bar(bar_positions, bar_heights, width=bar_width, label=model_name, color=bar_color)

    ax.set_xticks(experiment_positions)
    ax.set_xticklabels(experiments, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot percent-positive comparison bar chart.")
    p.add_argument("--models", nargs="+", required=True, help="Model names (order preserved).")
    p.add_argument("--experiments", nargs="+", required=True, help="Experiment names (order preserved).")
    p.add_argument("--out", required=True, help="Output image path (png/pdf/etc).")
    p.add_argument("--title", default="Percent Positive by Model and Experiment", help="Plot title.")
    p.add_argument("--outputs-dir", default="outputs", help="Outputs directory of Snakemake workflow (default: outputs).")
    p.add_argument("--colors", nargs="*", default=None, help="Optional list of colors, one per model (same order as --models).")
    p.add_argument("--ylabel", default="Predicted percent positive", help="Y-axis label.")
    args = p.parse_args()

    outputs_dir = Path(args.outputs_dir)
    out_path = Path(args.out)

    pct_pos_values = read_values(outputs_dir=outputs_dir, predictions_type=args.predictions_type, models=args.models, experiments=args.experiments)

    plot_grouped_bars(
        values=pct_pos_values,
        models=args.models,
        experiments=args.experiments,
        out_path=out_path,
        title=args.title,
        ylabel=args.ylabel,
        colors=args.colors,
    )


if __name__ == "__main__":
    main()
