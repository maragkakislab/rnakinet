import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def read_aurocs(auroc_tsv):
    stats_df = pd.read_csv(auroc_tsv, sep="\t")
    if "auroc" not in stats_df.columns:
        raise ValueError(f"{auroc_tsv} is missing required column: auroc")

    return stats_df.dropna(subset=["auroc"])


def plot_auroc_histogram(stats_df, output_path, title="AUROC Distribution"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(stats_df["auroc"], bins=50, edgecolor="black")
    ax.set_xlabel("AUROC")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main(args):
    stats_df = read_aurocs(args.auroc_tsv)
    if stats_df.empty:
        raise ValueError(f"{args.auroc_tsv} has no AUROC values to plot")

    plot_auroc_histogram(stats_df, args.output, args.title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a histogram of reference-level AUROCs")
    parser.add_argument("--auroc-tsv", required=True, help="Input TSV containing an auroc column")
    parser.add_argument("--output", required=True, help="Output plot path")
    parser.add_argument("--title", default="AUROC Distribution", help="Plot title")

    main(parser.parse_args())
