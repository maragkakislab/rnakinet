import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


READ_COUNT_COLUMN_PAIRS = [
    ("num_positive_reads", "num_negative_reads"),
    ("num_true_positives", "num_false_positives"),
]


def read_aurocs(auroc_tsv):
    stats_df = pd.read_csv(auroc_tsv, sep="\t")
    missing_columns = {"auroc"} - set(stats_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{auroc_tsv} is missing required columns: {missing}")

    for positive_col, negative_col in READ_COUNT_COLUMN_PAIRS:
        if {positive_col, negative_col}.issubset(stats_df.columns):
            stats_df = stats_df.copy()
            stats_df["total_reads"] = stats_df[positive_col] + stats_df[negative_col]
            return stats_df.dropna(subset=["auroc", "total_reads"])

    expected = " or ".join(f"{pos}/{neg}" for pos, neg in READ_COUNT_COLUMN_PAIRS)
    raise ValueError(f"{auroc_tsv} must contain read count columns: {expected}")


def plot_auroc_scatter(stats_df, output_path, title="Reads vs AUROC"):
    plot_df = stats_df[stats_df["total_reads"] > 0]
    if plot_df.empty:
        raise ValueError("No positive total read counts available to plot")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(plot_df["total_reads"], plot_df["auroc"], alpha=0.3, s=5)
    ax.set_xlabel("Total Reads")
    ax.set_ylabel("AUROC")
    ax.set_title(title)
    ax.set_xscale("log")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main(args):
    stats_df = read_aurocs(args.auroc_tsv)
    if stats_df.empty:
        raise ValueError(f"{args.auroc_tsv} has no AUROC/read-count rows to plot")

    plot_auroc_scatter(stats_df, args.output, args.title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot reference-level AUROC by total reads")
    parser.add_argument("--auroc-tsv", required=True, help="Input TSV containing AUROC stats")
    parser.add_argument("--output", required=True, help="Output plot path")
    parser.add_argument("--title", default="Reads vs AUROC", help="Plot title")

    main(parser.parse_args())
